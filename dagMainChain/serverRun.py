# -*- coding: utf-8 -*-

# 初始化DAG账本，提供DAG账本接口服务

import sys
from dagComps import transaction
from dagComps.dag import DAG
from dagSocket import dagServer
import socket
import os
import time
import shutil
import numpy as np
import glob
import torch
import subprocess

# Common Components
sys.path.append('../commonComponent')
import usefulTools

# FL related - import model building components
sys.path.append('../federatedLearning')
from models.Nets import CNNMnist, CNNCifar
from utils.options import args_parser

# The number of tips selected by the new transaction
alpha = 3
## The number of tips needs to be kept greater than 3
beta = 3

def main(arg=True, dataset='mnist', num_channels=1):
    # Clean up old DAG data when switching datasets
    # This ensures a fresh genesis block is created with the correct model
    if os.path.exists('./dagSS'):
        print("=" * 80)
        print("Cleaning up old DAG data to ensure correct model architecture...")
        print("=" * 80)
        shutil.rmtree('./dagSS')
        print("Removed old dagSS directory")
    
    os.mkdir('./dagSS')
    os.mkdir('./dagSS/dagPool')
    host_DAG = DAG(active_lst_addr='./dagSS/active_list.json',timespan=60)

# Generate the genesis block for DAG
    # Create a new model and upload to IPFS
    print("=" * 80)
    print("Creating genesis model and uploading to IPFS...")
    print("=" * 80)
    
    # Create a model based on dataset type
    class Args:
        def __init__(self, num_channels, num_classes):
            self.num_channels = num_channels
            self.num_classes = num_classes
    
    model_args = Args(num_channels, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset == 'cifar':
        print(f"Creating CNNCifar model for CIFAR-10 dataset (channels: {num_channels})")
        net_glob = CNNCifar(model_args).to(device)
    else:
        print(f"Creating CNNMnist model for MNIST dataset (channels: {num_channels})")
        net_glob = CNNMnist(model_args).to(device)
    
    # Get model state dict (weights)
    w_glob = net_glob.state_dict()
    
    # Save model to temporary file
    temp_model_file = './dagSS/genesis_model_temp.pkl'
    os.makedirs('./dagSS', exist_ok=True)
    torch.save(w_glob, temp_model_file)
    print(f"Model saved to temporary file: {temp_model_file}")
    
    # Upload model to IPFS
    print("Uploading model to IPFS network...")
    genesisInfo, upload_status = usefulTools.ipfsAddFile(temp_model_file)
    
    if upload_status == 0:
        print("=" * 80)
        print("Model successfully uploaded to IPFS!")
        print("Model file hash (IPFS):", genesisInfo)
        print("=" * 80)
        # Persist genesis CID (and optional IPFS peer id) for other nodes to auto-discover.
        # This enables automation on clients without manually copying the CID from logs.
        try:
            os.makedirs('./dagSS', exist_ok=True)
            with open('./dagSS/genesis.cid', 'w', encoding='utf-8') as f:
                f.write(str(genesisInfo).strip() + '\n')
            # Also store server IPFS PeerID if daemon is running (best-effort).
            peer_id = None
            try:
                ipfs_id = subprocess.check_output(['ipfs', 'id', '-f=<id>'], stderr=subprocess.STDOUT, text=True).strip()
                peer_id = ipfs_id if ipfs_id else None
            except Exception:
                peer_id = None
            if peer_id:
                with open('./dagSS/ipfs_peer_id', 'w', encoding='utf-8') as f:
                    f.write(peer_id + '\n')
        except Exception as e:
            print(f"Warning: Failed to persist genesis CID/peer id: {e}")
        # Clean up temporary file
        if os.path.exists(temp_model_file):
            os.remove(temp_model_file)
            print(f"Temporary file {temp_model_file} removed.")
    else:
        print("=" * 80)
        print("ERROR: Failed to upload model to IPFS!")
        print("Error message:", genesisInfo)
        print("=" * 80)
        # Use a default hash or exit
        raise Exception("Failed to upload genesis model to IPFS. Please check IPFS is running.")
    # Create genesis transaction
    ini_trans = transaction.Transaction(time.time(), 0, 0.0, genesisInfo, [])
    transaction.save_genesis(ini_trans, './dagSS/dagPool/')
    ini_trans.name = 'GenesisBlock'
    ini_trans_file_addr = './dagSS/dagPool/'+ ini_trans.name +'.json'
    host_DAG.DAG_publish(ini_trans, beta)
    host_DAG.DAG_genesisDel()

    while arg:
        # Listen on 0.0.0.0 to allow connections from other devices (e.g., Raspberry Pi)
        # Change to "127.0.0.1" if you only want local connections
        dagServer.socket_service("0.0.0.0", host_DAG, beta)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DAG Server for Federated Learning')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'],
                        help='Dataset type: mnist or cifar (default: mnist)')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='Number of channels: 1 for MNIST, 3 for CIFAR (default: 1)')
    
    args = parser.parse_args()
    
    # Auto-adjust num_channels based on dataset if not explicitly set
    if args.dataset == 'cifar' and args.num_channels == 1:
        args.num_channels = 3
        print(f"Auto-adjusted num_channels to 3 for CIFAR dataset")
    
    main(True, dataset=args.dataset, num_channels=args.num_channels)