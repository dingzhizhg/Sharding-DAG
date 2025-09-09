#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 暂时不能用

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from json import dumps
import datetime
import pandas as pd
import os
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Update import DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from models.Attacks import DelayAttackManager, noise_attack, create_labelflip_mapping
from utils.attack_utils import select_malicious_nodes, is_malicious_node, is_lazy_node


dateNow = datetime.datetime.now().strftime('%m%d%H%M%S')  # 简短时间戳：月日时分秒
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    
    # Load genesis model if exists, otherwise use newly initialized model
    if os.path.exists('./data/genesisGPUForCNN.pkl'):
        tmp_glob = torch.load('./data/genesisGPUForCNN.pkl')
    else:
        # Use newly initialized model weights as genesis
        tmp_glob = copy.deepcopy(w_glob)
        # Optionally save it for future use
        os.makedirs('./data', exist_ok=True)
        torch.save(tmp_glob, './data/genesisGPUForCNN.pkl')
        print('Created genesis model file: ./data/genesisGPUForCNN.pkl')


    # training
    loss_train = []

    m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # Initialize delay attack manager if delay attack is enabled
    delay_manager = None
    if 'delay' in args.attack_type:
        delay_manager = DelayAttackManager()

    workerIterIdx = {}
    
    for item in idxs_users:
        workerIterIdx[item] = 0

    realEpoch = 0
    currentEpoch = 0

    acc_test_list = []
    loss_test_list = []

    acc_train_list = []
    loss_train_list = []

    # Aggregation weight for new model (higher weight means more trust in new updates)
    alpha = 0.8  # Weight for new model, base model gets (1-alpha)
    
    while currentEpoch <= args.epochs:
        currentEpoch += 1
        base_glob = tmp_glob
        net_glob.load_state_dict(base_glob)

        print('# of current epoch is ' + str(currentEpoch))

        # Select multiple workers per round for better convergence (at least 2-3 workers)
        num_workers_per_round = max(2, min(int(m * 0.3), 5))  # 30% of selected users, at least 2, at most 5
        workers_now = np.random.choice(idxs_users, num_workers_per_round, replace=False).tolist()

        print('Selected workers for this round: ' + str(workers_now))
        
        # Select malicious nodes for this round
        malicious_nodes = select_malicious_nodes(
            workers_now, 
            args.malicious_frac, 
            args.attack_type,
            seed=args.seed if hasattr(args, 'seed') else None
        )
        
        w_locals = []
        for workerNow in workers_now:
            staleFlag = np.random.randint(-1, 4, size=1)[0]
            print('The staleFlag of worker ' + str(workerNow) + ' is ' + str(staleFlag))

            if staleFlag <= 4:
                is_mal = is_malicious_node(workerNow, malicious_nodes)
                is_lazy = is_lazy_node(workerNow, malicious_nodes, args.attack_type)
                
                # Lazy attack: skip training and submission
                if is_lazy:
                    print(f'Lazy node {workerNow} skipped in iteration {currentEpoch}')
                    continue
                
                # Create label mapping for labelflip attack if enabled
                label_mapping = None
                if is_mal and 'labelflip' in args.attack_type:
                    # Use node ID and epoch as seed for reproducibility
                    labelflip_seed = (args.seed if hasattr(args, 'seed') else 1) * 1000 + workerNow * 100 + currentEpoch
                    label_mapping = create_labelflip_mapping(
                        num_classes=args.num_classes,
                        mode=args.labelflip_mode,
                        target_label=args.labelflip_target,
                        seed=labelflip_seed
                    )
                    print(f'Labelflip attack: node {workerNow} using {args.labelflip_mode} mode')
                
                # Delay attack or normal training
                if is_mal and 'delay' in args.attack_type:
                    # Normal training for delay attack
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[workerNow], label_mapping=label_mapping)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    # Save current model to delay queue
                    delay_manager.save_delayed_model(workerNow, currentEpoch, w)
                    # Get delayed model (from delay_rounds rounds ago) if exists
                    w_delayed = delay_manager.get_delayed_model(workerNow, currentEpoch, args.delay_rounds)
                    if w_delayed is not None:
                        w = w_delayed
                        print(f'Delay attack: node {workerNow} submitted delayed model from round {currentEpoch - args.delay_rounds}')
                else:
                    # Normal training
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[workerNow], label_mapping=label_mapping)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                
                # Apply noise attack if enabled
                if is_mal and 'noise' in args.attack_type:
                    w = noise_attack(w, args.noise_scale)
                
                w_locals.append(copy.deepcopy(w))
                print('Training of node device '+str(workerNow)+' in iteration '+str(currentEpoch)+' has done!')
        
        # Clean old delayed models
        if delay_manager is not None:
            delay_manager.clear_old_models(currentEpoch, max_delay=args.delay_rounds)

        # Aggregate multiple workers' updates first
        if len(w_locals) > 0:
            w_aggregated = FedAvg(w_locals)
            
            # Then aggregate with base model using weighted average
            # Use adaptive weight: give more weight to new updates
            tmp_glob = copy.deepcopy(base_glob)
            for k in tmp_glob.keys():
                tmp_glob[k] = (1 - alpha) * base_glob[k] + alpha * w_aggregated[k]
        else:
            # If no workers updated, keep base model
            tmp_glob = base_glob

        net_glob.load_state_dict(tmp_glob)
        net_glob.eval()
        
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)

        acc_test_list.append(acc_test.cpu().numpy().tolist()/100)
        loss_test_list.append(loss_test)

        # acc_train_list.append(acc_train.cpu().numpy().tolist())
        # loss_train_list.append(loss_train)

        # Create save directory if it doesn't exist
        save_dir = './results'
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate simplified filename: method-attack-malFrac-timestamp
        # 只保留恶意节点比例
        attack_str = args.attack_type if args.attack_type != 'none' else 'none'
        mal_frac_str = f"{args.malicious_frac:.1f}" if args.malicious_frac > 0 else "0.0"
        filename_base = "asyn-{}-{}-{}".format(attack_str, mal_frac_str, dateNow)
        
        accDfTest = pd.DataFrame({'baseline':acc_test_list})
        accDfTest.to_csv("{}/{}.csv".format(save_dir, filename_base), index=False, sep=',')

        lossDfTest = pd.DataFrame({'baseline':loss_test_list})
        lossDfTest.to_csv("{}/{}_loss.csv".format(save_dir, filename_base), index=False, sep=',')


        # accDfTrain = pd.DataFrame({'baseline':acc_train_list})
        # accDfTrain.to_csv("D:\\ChainsFLexps\\asynFL\\AsynFL-Train-idd{}-{}-{}localEpochs-{}users-{}Rounds_ACC_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

        # lossDfTrain = pd.DataFrame({'baseline':loss_train_list})
        # lossDfTrain.to_csv("D:\\ChainsFLexps\\asynFL\\AsynFL-Train-idd{}-{}-{}localEpochs-{}users-{}Rounds_Loss_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

        print('# of real epoch is ' + str(realEpoch))
        print("Testing accuracy: {:.2f}".format(acc_test))

        # Update worker iteration indices
        for workerNow in workers_now:
            workerIterIdx[workerNow] += 1
        realEpoch += 1
