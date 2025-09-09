#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from json import dumps
import datetime
import os
import pandas as pd

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from models.Attacks import DelayAttackManager, noise_attack, create_labelflip_mapping
from utils.attack_utils import select_malicious_nodes, is_malicious_node, is_lazy_node


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            # allocate the dataset index to users
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)
    torch.save(w_glob, './data/genesisGPUForCNN.pkl')
    # training
    loss_train = []
    
    # Lists to store accuracy and loss for each round
    acc_test_list = []
    loss_test_list = []
    
    acc_train_list = []
    loss_train_list = []
    
    dateNow = datetime.datetime.now().strftime('%m%d%H%M%S')  # 简短时间戳：月日时分秒

    # Initialize delay attack manager if delay attack is enabled
    delay_manager = None
    if 'delay' in args.attack_type:
        delay_manager = DelayAttackManager()

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Select malicious nodes for this round
        malicious_nodes = select_malicious_nodes(
            idxs_users.tolist(), 
            args.malicious_frac, 
            args.attack_type,
            seed=args.seed if hasattr(args, 'seed') else None
        )
        
        for idx in idxs_users:
            is_mal = is_malicious_node(idx, malicious_nodes)
            is_lazy = is_lazy_node(idx, malicious_nodes, args.attack_type)
            
            # Lazy attack: skip training and submission
            if is_lazy:
                print(f'Lazy node {idx} skipped in iteration {iter}')
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
                print(f'Labelflip attack: node {idx} using {args.labelflip_mode} mode')
            
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
                    print(f'Delay attack: node {idx} submitted delayed model from round {iter - args.delay_rounds}')
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
            print(idx)
        
        # Clean old delayed models
        if delay_manager is not None:
            delay_manager.clear_old_models(iter, max_delay=args.delay_rounds)
        
        # update global weights (only if we have models)
        if len(w_locals) > 0:
            w_glob = FedAvg(w_locals)
        else:
            print(f"Warning: No nodes responded in iteration {iter}, using previous global model")
        # Ensure paras directory exists
        os.makedirs('./data/paras', exist_ok=True)
        torch.save(w_glob, './data/paras/'+str(iter)+'parameter.pkl')
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        print('The content of w_glob', w_glob)

        # Calculate loss (but don't print)
        if len(loss_locals) > 0:
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)
            loss_train_list.append(loss_avg)
        else:
            loss_train.append(0.0)
            loss_train_list.append(0.0)
        
        # Evaluate on test set after each round
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # acc_train, loss_train_eval = test_img(net_glob, dataset_train, args)
        
        loss_test_list.append(loss_test)
        acc_test_list.append((acc_test.cpu().numpy().tolist()/100))
        # acc_train_list.append((acc_train.cpu().numpy().tolist()/100))
        
        # Save to CSV files after each round (similar to baseline)
        os.makedirs('./save', exist_ok=True)
        
        # Generate simplified filename: method-attack-malFrac-timestamp
        # 只保留恶意节点比例
        attack_str = args.attack_type if args.attack_type != 'none' else 'none'
        mal_frac_str = f"{args.malicious_frac:.1f}" if args.malicious_frac > 0 else "0.0"
        filename_base = "fed-{}-{}-{}".format(attack_str, mal_frac_str, dateNow)
        
        accDfTest = pd.DataFrame({'fed':acc_test_list})
        accDfTest.to_csv("./save/{}.csv".format(filename_base), index=False, sep=',')
        
        lossDfTest = pd.DataFrame({'fed':loss_test_list})
        lossDfTest.to_csv("./save/{}_loss.csv".format(filename_base), index=False, sep=',')
        
        net_glob.train()

    # plot loss curve
    # Ensure save directory exists
    os.makedirs('./save', exist_ok=True)
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    # Generate simplified filename for plot
    # 只保留恶意节点比例
    attack_str = args.attack_type if args.attack_type != 'none' else 'none'
    mal_frac_str = f"{args.malicious_frac:.1f}" if args.malicious_frac > 0 else "0.0"
    plot_filename = './save/fed-{}-{}-{}.png'.format(
        attack_str, mal_frac_str, datetime.datetime.now().strftime('%m%d%H%M%S')
    )
    plt.savefig(plot_filename)

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

