#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    # Attack arguments
    parser.add_argument('--attack_type', type=str, default='none',
                        choices=['none', 'lazy', 'delay', 'noise', 'labelflip', 
                                'lazy+noise', 'delay+noise', 'labelflip+noise', 
                                'lazy+labelflip', 'delay+labelflip'],
                        help='Type of attack: none, lazy, delay, noise, labelflip, or combinations')
    parser.add_argument('--malicious_frac', type=float, default=0.0,
                        help='Fraction of malicious nodes (0.0-1.0), relative to selected nodes per round')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                        help='Noise scale for noise attack (standard deviation of Gaussian noise). Default 0.1 for stronger attack effect.')
    parser.add_argument('--delay_rounds', type=int, default=3,
                        help='Number of rounds to delay for delay attack')
    parser.add_argument('--lazy_timeout', type=float, default=30.0,
                        help='Timeout in seconds for waiting lazy nodes (default: 30.0)')
    parser.add_argument('--fixed_lazy', action='store_true',
                        help='Use fixed lazy nodes across all rounds (default: False, lazy nodes selected per round)')
    parser.add_argument('--labelflip_mode', type=str, default='random',
                        choices=['random', 'fixed'],
                        help='Label flip mode: random (randomly flip to other labels) or fixed (flip all to target_label)')
    parser.add_argument('--labelflip_target', type=int, default=None,
                        help='Target label for fixed label flip mode (default: num_classes-1). Only used when labelflip_mode=fixed')
    
    # Node assignment arguments (for distributed deployment)
    parser.add_argument('--node_id', type=int, default=None,
                        help='Node ID for this client (0, 1, 2, ...). If None, client selects from all devices.')
    parser.add_argument('--devices_per_node', type=int, default=100,
                        help='Number of devices per node (default: 100). Used with --node_id to determine device range.')
    
    # Sharding arguments
    parser.add_argument('--shard_id', type=int, default=None,
                        help='Shard ID for distributed training (0, 1, 2, ...). If specified, loads shard-specific user split file.')
    
    # Network arguments
    parser.add_argument('--fabric_host', type=str, default=None,
                        help='Fabric network host address (e.g., 192.168.137.208). If not specified, uses FABRIC_HOST environment variable or localhost')
    
    args = parser.parse_args()
    return args
