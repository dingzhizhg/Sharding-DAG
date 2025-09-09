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
import time
import glob
import torch

# Common Components
sys.path.append('../commonComponent')
import usefulTools

# FL related (no longer needed - using existing model hash)
# sys.path.append('../federatedLearning')
# import buildModels

# The number of tips selected by the new transaction
alpha = 3
## The number of tips needs to be kept greater than 3
beta = 3

def main(arg=True):
    if os.path.exists('./dagSS') == False:
        os.mkdir('./dagSS')
    if os.path.exists('./dagSS/dagPool'):
        shutil.rmtree('./dagSS/dagPool')
    os.mkdir('./dagSS/dagPool')
    host_DAG = DAG(active_lst_addr='./dagSS/active_list.json',timespan=60)

# Generate the genesis block for DAG
    # Use the existing model hash instead of building a new model
    genesisInfo = 'QmabPyx31VoTPdwePRDe6vtsKtigepxXmXgBcYhgu7MEDk'
    print("=" * 80)
    print("Using existing model from IPFS for GenesisBlock!")
    print("Model file hash (IPFS):", genesisInfo)
    print("=" * 80)
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

    main(True)