#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from json import dumps
import datetime
import pickle
import os
import pandas as pd
import datetime
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from models.Attacks import DelayAttackManager, noise_attack, create_labelflip_mapping
from utils.attack_utils import select_malicious_nodes, is_malicious_node, is_lazy_node, select_fixed_lazy_nodes
import buildModels

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dateNow = datetime.datetime.now().strftime('%m%d%H%M%S')  # 简短时间戳：月日时分秒

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # # load dataset and split users
    # if args.dataset == 'mnist':
    #     trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    #     dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    #     # sample users
    #     if args.iid:
    #         # allocate the dataset index to users
    #         dict_users = mnist_iid(dataset_train, args.num_users)
    #     else:
    #         dict_users = mnist_noniid(dataset_train, args.num_users)
    # elif args.dataset == 'cifar':
    #     trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    #     dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    #     if args.iid:
    #         dict_users = cifar_iid(dataset_train, args.num_users)
    #     else:
    #         exit('Error: only consider IID setting in CIFAR10')
    # else:
    #     exit('Error: unrecognized dataset')
    # img_size = dataset_train[0][0].shape

    # # build model
    # if args.model == 'cnn' and args.dataset == 'cifar':
    #     net_glob = CNNCifar(args=args).to(args.device)
    # elif args.model == 'cnn' and args.dataset == 'mnist':
    #     net_glob = CNNMnist(args=args).to(args.device)
    # elif args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    # else:
    #     exit('Error: unrecognized model')
    # print(net_glob)
    # net_glob.train()

    # build network
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    
    # Load genesis model if exists, otherwise use newly initialized model
    genesis_file = './data/genesisForCNN.pkl'
    if os.path.exists(genesis_file):
        base_glob = torch.load(genesis_file)
        print('Loaded genesis model from: ' + genesis_file)
    else:
        # Use newly initialized model weights as genesis
        base_glob = copy.deepcopy(w_glob)
        # Ensure data directory exists
        os.makedirs('./data', exist_ok=True)
        torch.save(base_glob, genesis_file)
        print('Created genesis model file: ' + genesis_file)
    
    net_glob.load_state_dict(base_glob)

    allDeviceName = []
    for i in range(args.num_users):
        allDeviceName.append("device"+("{:0>5d}".format(i)))

    # training
    acc_test_list = []
    loss_test_list = []

    acc_train_list = []
    loss_train_list = []

    # with open('D:\\expRes\\dict_users.pkl', 'rb') as f:
    #     dict_users = pickle.load(f)
    # idxs_users = [5, 56, 76, 78, 68, 25, 47, 15, 61, 55, 60, 37, 27, 70, 79, 34, 18, 88, 57, 98, 48, 46, 33, 82, 4, 7, 6, 91, 92, 52]
    # print('Number of selected devices '+str(len(idxs_users)))
    # idxs_users = [ 7, 85, 14, 67, 88, 72, 20, 77, 89, 34, 82, 15, 26, 6, 42, 8, 60 ,49, 65, 46, 53, 24 ,31 ,98 ,64, 13, 56, 19, 74, 95]
    m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users

    # Initialize delay attack manager if delay attack is enabled
    delay_manager = None
    if 'delay' in args.attack_type:
        delay_manager = DelayAttackManager()

    # 初始化随机种子（如果提供了seed参数）
    base_seed = args.seed if hasattr(args, 'seed') and args.seed is not None else None
    if base_seed is not None:
        np.random.seed(base_seed)
        random.seed(base_seed)
    
    # 初始化固定的lazy节点集合（如果启用了fixed_lazy）
    fixed_lazy_nodes = set()
    if hasattr(args, 'fixed_lazy') and args.fixed_lazy and 'lazy' in args.attack_type:
        # 使用固定的lazy节点集合，lazy_frac相对于总用户数
        fixed_lazy_seed = base_seed * 100000 if base_seed is not None else None
        fixed_lazy_nodes = select_fixed_lazy_nodes(
            args.num_users, 
            args.malicious_frac,  # 使用malicious_frac作为lazy节点比例
            seed=fixed_lazy_seed
        )
        print(f"使用固定的lazy节点集合: {sorted(fixed_lazy_nodes)} (共{len(fixed_lazy_nodes)}个节点，占总用户的{len(fixed_lazy_nodes)/args.num_users*100:.1f}%)")

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        
        # 为节点选择设置固定种子（基于 base_seed + iteration），确保可重复性
        if base_seed is not None:
            # 使用 base_seed + iteration 作为节点选择的种子
            # 这样可以确保每轮选择的节点不同但可重复
            node_selection_seed = base_seed * 10000 + iter
            np.random.seed(node_selection_seed)
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Select malicious nodes for this round
        # 如果使用固定的lazy节点，则从选中的节点中筛选出在固定集合中的节点
        if hasattr(args, 'fixed_lazy') and args.fixed_lazy and 'lazy' in args.attack_type:
            # 使用固定的lazy节点集合：从选中的节点中找出在固定集合中的节点
            malicious_nodes = set(idx for idx in idxs_users if idx in fixed_lazy_nodes)
        else:
            # 使用原来的方式：每轮随机选择恶意节点
            malicious_seed = None
            if base_seed is not None:
                malicious_seed = base_seed * 10000 + iter + 50000  # 使用不同的偏移量避免冲突
            
            malicious_nodes = select_malicious_nodes(
                idxs_users.tolist(), 
                args.malicious_frac, 
                args.attack_type,
                seed=malicious_seed
            )

        for idx in idxs_users:
            is_mal = is_malicious_node(idx, malicious_nodes)
            # 如果使用固定lazy节点，直接检查是否在固定集合中
            if hasattr(args, 'fixed_lazy') and args.fixed_lazy and 'lazy' in args.attack_type:
                is_lazy = idx in fixed_lazy_nodes
            else:
                is_lazy = is_lazy_node(idx, malicious_nodes, args.attack_type)
            
            # Lazy attack: skip training and submission
            if is_lazy:
                print(f'Lazy node {allDeviceName[idx]} skipped in iteration {iter}')
                continue
            
            # Create label mapping for labelflip attack if enabled
            label_mapping = None
            if is_mal and 'labelflip' in args.attack_type:
                # Use node ID and iteration as seed for reproducibility
                labelflip_seed = (args.seed if hasattr(args, 'seed') else 1) * 1000 + idx * 100 + iter
                label_mapping = create_labelflip_mapping(
                    num_classes=args.num_classes,
                    mode=args.labelflip_mode,
                    target_label=args.labelflip_target,
                    seed=labelflip_seed
                )
                print(f'Labelflip attack: {allDeviceName[idx]} using {args.labelflip_mode} mode')
            
            # Delay attack or normal training
            if is_mal and 'delay' in args.attack_type:
                # Normal training for delay attack
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], label_mapping=label_mapping)
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                # Save current model to delay queue
                delay_manager.save_delayed_model(idx, iter, w)
                # Get delayed model (from delay_rounds rounds ago) if exists
                w_delayed = delay_manager.get_delayed_model(idx, iter, args.delay_rounds)
                if w_delayed is not None:
                    w = w_delayed
                    print(f'Delay attack: {allDeviceName[idx]} submitted delayed model from round {iter - args.delay_rounds}')
            else:
                # Normal training
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], label_mapping=label_mapping)
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            # Apply noise attack if enabled
            if is_mal and 'noise' in args.attack_type:
                w = noise_attack(w, args.noise_scale)
            
            w_locals.append(copy.deepcopy(w))
            if loss is not None:
                loss_locals.append(copy.deepcopy(loss))
            print('Training of '+str(allDeviceName[idx])+' in iteration '+str(iter)+' has done!')
        
        # Clean old delayed models
        if delay_manager is not None:
            delay_manager.clear_old_models(iter, max_delay=args.delay_rounds)
        
        # Update global weights (only if we have models)
        if len(w_locals) > 0:
            w_glob = FedAvg(w_locals)
            if len(loss_locals) > 0:
                loss_avg = sum(loss_locals) / len(loss_locals)
                loss_train_list.append(loss_avg)
            else:
                loss_train_list.append(0.0)
        else:
            print(f"Warning: No nodes responded in iteration {iter}, using previous global model")

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob.eval()


        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # acc_train, _ = test_img(net_glob, dataset_train, args)


        loss_test_list.append(loss_test)
        acc_test_list.append((acc_test.cpu().numpy().tolist()/100))

        # Generate simplified filename: method-attack-malFrac-timestamp
        # 只保留恶意节点比例
        attack_str = args.attack_type if args.attack_type != 'none' else 'none'
        mal_frac_str = f"{args.malicious_frac:.1f}" if args.malicious_frac > 0 else "0.0"
        filename_base = "baseline-{}-{}-{}".format(attack_str, mal_frac_str, dateNow)

        # Ensure save directory exists
        os.makedirs('./save', exist_ok=True)

        accDfTest = pd.DataFrame({'baseline':acc_test_list})
        accDfTest.to_csv("./save/{}.csv".format(filename_base), index=False, sep=',')

        # loss文件不再保存
        # lossDfTest = pd.DataFrame({'baseline':loss_test_list})
        # lossDfTest.to_csv("./save/{}_loss.csv".format(filename_base), index=False, sep=',')

        # loss_train_list.append(loss_train)
        # acc_train_list.append((acc_train.cpu().numpy().tolist())/100)

        # accDfTrain = pd.DataFrame({'baseline':acc_train_list})
        # accDfTrain.to_csv("D:\\ChainsFLexps\\ggFL\\GoogleFL-Train-iid{}-{}-{}localEpochs-{}users-{}Rounds_ACC_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

        # lossDfTrain = pd.DataFrame({'baseline':loss_train_list})
        # lossDfTrain.to_csv("D:\\ChainsFLexps\\ggFL\\GoogleFL-Train-iid{}-{}-{}localEpochs-{}users-{}Rounds_Loss_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

        # print('The acc in epoch '+str(iter)+' is '+str(acc_test.cpu().numpy().tolist()))
        # print('The loss in epoch '+str(iter)+' is '+str(loss_test))
        # # print('The content of w_glob', w_glob)
    
    print(acc_test_list)

