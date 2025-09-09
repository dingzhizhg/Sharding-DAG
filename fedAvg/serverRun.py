# -*- coding: utf-8 -*-

"""
FedAvg Server (Aggregation Node)
负责初始化模型、发布任务、聚合客户端模型、发布聚合结果
"""

import sys
import os
import time
import shutil
import copy
import random
import subprocess
import threading
import json
import pickle
import pandas as pd
import numpy as np

# Common Components
sys.path.append('../commonComponent')
import usefulTools

# FL related
sys.path.append('../federatedLearning')
import torch
import buildModels
from models.Fed import FedAvg
from models.test import test_img
from utils.options import args_parser
from utils.attack_utils import select_malicious_nodes, is_lazy_node
import datetime


def main():
    # 创建必要的目录
    if os.path.exists('./serverS'):
        shutil.rmtree('./serverS')
    os.makedirs('./serverS', exist_ok=True)
    os.makedirs('./serverS/paras', exist_ok=True)
    os.makedirs('./serverS/paras/local', exist_ok=True)
    os.makedirs('./serverS/paras/agg', exist_ok=True)
    
    # 创建结果目录
    if not os.path.exists('../data/result'):
        os.makedirs('../data/result', exist_ok=True)
    
    # 解析参数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # 构建模型
    print("=" * 80)
    print("Building model...")
    print("=" * 80)
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()
    
    # 保存dict_users到文件以便客户端使用（使用数据集特定的文件名）
    dict_users_file = f'../commonComponent/dict_users_{args.dataset}.pkl'
    os.makedirs('../commonComponent', exist_ok=True)
    with open(dict_users_file, 'wb') as f:
        pickle.dump(dict_users, f)
    print(f"Saved dict_users to {dict_users_file}")
    
    # 上传dict_users到IPFS（可选，如果需要的话）
    dict_users_hash = None
    try:
        dict_users_hash, status = usefulTools.ipfsAddFile(dict_users_file)
        if status == 0:
            print(f"Uploaded dict_users to IPFS: {dict_users_hash}")
        else:
            print(f"Warning: Failed to upload dict_users to IPFS")
    except Exception as e:
        print(f"Warning: Could not upload dict_users to IPFS: {e}")
    
    # 初始化genesis模型
    w_glob = net_glob.state_dict()
    genesis_file = './serverS/genesis_model.pkl'
    torch.save(w_glob, genesis_file)
    print(f"Saved genesis model to {genesis_file}")
    
    # 上传genesis模型到IPFS
    print("=" * 80)
    print("Uploading genesis model to IPFS...")
    print("=" * 80)
    genesis_hash = None
    while genesis_hash is None:
        hash_value, status = usefulTools.ipfsAddFile(genesis_file)
        if status == 0:
            genesis_hash = hash_value.strip()
            print(f"Genesis model uploaded to IPFS: {genesis_hash}")
        else:
            print(f"Failed to upload genesis model, retrying...")
            time.sleep(2)
    
    # 生成时间戳
    dateNow = datetime.datetime.now().strftime('%m%d%H%M%S')
    
    # 初始化设备列表（用于识别device名称）
    allDeviceName = []
    for i in range(args.num_users):
        allDeviceName.append("device" + ("{:0>5d}".format(i)))
    
    # 训练循环
    iteration_count = 0
    acc_test_list = []
    loss_test_list = []
    
    while True:
        print('\n' + '=' * 80)
        print(f'Iteration {iteration_count} starting')
        print('=' * 80 + '\n')
        
        # Server不再指定device，由每个client自己随机选择
        print("Server mode: Waiting for clients to upload models (clients will randomly select devices)")
        
        # 生成任务ID
        taskID = 'task' + str(random.randint(1000, 9999))
        
        # 发布任务
        print(f"Releasing task {taskID}...")
        taskEpochs = args.epochs
        taskInitStatus = "start"
        taskUsersFrac = args.frac
        
        while True:
            taskRelease = subprocess.Popen(
                args=['../commonComponent/interRun.sh release ' + taskID + ' ' + str(taskEpochs) + ' ' + taskInitStatus + ' ' + str(taskUsersFrac)],
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'
            )
            trOuts, trErrs = taskRelease.communicate(timeout=10)
            if taskRelease.poll() == 0:
                print(f'Task {taskID} has been released!')
                print(f'Task details: {trOuts.strip()}\n')
                break
            else:
                print(f'Failed to release task {taskID}, retrying...')
                print(f'Error: {trErrs}')
                time.sleep(2)
        
        # 发布初始模型（epoch 0）
        print(f"Publishing initial aggregated model (epoch 0)...")
        while True:
            spcAggModelPublish = subprocess.Popen(
                args=['../commonComponent/interRun.sh aggregated ' + taskID + ' 0 training ' + genesis_hash],
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'
            )
            aggPubOuts, aggPubErrs = spcAggModelPublish.communicate(timeout=10)
            if spcAggModelPublish.poll() == 0:
                print(f'Initial aggregated model published!')
                print(f'Details: {aggPubOuts.strip()}\n')
                break
            else:
                print(f'Failed to publish initial model, retrying...')
                print(f'Error: {aggPubErrs}')
                time.sleep(2)
        
        # 等待客户端开始训练
        print("Waiting for clients to start training...")
        time.sleep(10)
        
        # 每个epoch的聚合循环
        currentEpoch = 1
        aggModelAcc = 50.0  # 初始准确率
        
        while currentEpoch <= args.epochs:
            print('\n' + '-' * 80)
            print(f'Epoch {currentEpoch}/{args.epochs}')
            print('-' * 80 + '\n')
            
            # Server不再知道哪些device会参与，需要扫描所有可能的device
            # 等待客户端上传模型，使用超时机制
            print(f"Waiting for clients to upload models (timeout: {args.lazy_timeout}s)...")
            start_time = time.time()
            w_locals = []
            received_devices = set()
            
            # 轮询所有可能的device，收集上传的模型
            # 使用超时机制，在超时后聚合已收到的模型
            while True:
                elapsed = time.time() - start_time
                
                # 扫描所有可能的device，查找新上传的模型
                for device_idx in range(args.num_users):
                    deviceID = allDeviceName[device_idx]
                    localFileName = './serverS/paras/local/' + taskID + '-' + deviceID + '-epoch-' + str(currentEpoch) + '.pkl'
                    
                    # 如果这个device的模型已经处理过，跳过
                    if deviceID in received_devices:
                        continue
                
                    # 检查是否有新模型上传
                    if os.path.exists(localFileName):
                        try:
                            canddts_dev_pas = torch.load(localFileName, map_location=torch.device('cpu'))
                            # 可选：检查模型质量
                            acc_canddts_dev, loss_canddts_dev = buildModels.evalua(net_glob, canddts_dev_pas, dataset_test, args)
                            acc_canddts_dev = acc_canddts_dev.cpu().numpy().tolist()
                            print(f"Received model from {deviceID}: accuracy {acc_canddts_dev:.2f}%")
                    
                            # 简单的质量检查：如果准确率低于聚合模型太多，认为是恶意节点
                            if len(w_locals) > 0 and (acc_canddts_dev - aggModelAcc) < -20:
                                print(f"{deviceID} detected as potentially malicious (low accuracy), skipping...")
                                received_devices.add(deviceID)  # 标记为已处理，但不加入聚合
                                continue
                    
                            w_locals.append(copy.deepcopy(canddts_dev_pas))
                            received_devices.add(deviceID)
                            print(f"✓ Added model from {deviceID} to aggregation list")
                        except Exception as e:
                            print(f"Error loading model from {deviceID}: {e}")
                            received_devices.add(deviceID)  # 标记为已处理，避免重复尝试
                
                # 检查超时或是否有足够的模型
                if elapsed >= args.lazy_timeout:
                    print(f"\nTimeout reached ({elapsed:.2f}s). Collected {len(w_locals)} models from clients.")
                    break
                
                # 如果收到了一些模型，可以继续等待更多，或者设置最小等待时间
                if len(w_locals) > 0 and elapsed >= 30:  # 至少等待30秒
                    print(f"Collected {len(w_locals)} models so far. Waiting a bit more for additional models...")
                    time.sleep(5)
                    if elapsed >= args.lazy_timeout:
                        break
                
                time.sleep(2)  # 轮询间隔
            
            # 验证收到的模型数量
            if len(w_locals) == 0:
                print(f"\nERROR: No models available for aggregation!")
                print(f"Waiting and retrying...")
                time.sleep(10)
                continue  # 重新尝试当前epoch，不增加currentEpoch
            
            # 聚合模型（使用所有收到的模型）
            print(f"\n✓ {len(w_locals)} models received. Aggregating using FedAvg...")
            w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)
            
            # 评估聚合后的模型
            net_glob.eval()
            aggModelAcc, aggModelLoss = test_img(net_glob, dataset_test, args)
            aggModelAcc_value = aggModelAcc.cpu().numpy().tolist()
            
            acc_test_list.append(aggModelAcc_value / 100)
            loss_test_list.append(aggModelLoss)
            
            print(f"\n✓ Aggregation completed!")
            print(f"Aggregated model accuracy: {aggModelAcc_value:.2f}%")
            print(f"Aggregated model loss: {aggModelLoss:.4f}")
            
            # 保存准确率到CSV
            attack_str = args.attack_type if hasattr(args, 'attack_type') and args.attack_type != 'none' else 'none'
            mal_frac_str = f"{args.malicious_frac:.1f}" if hasattr(args, 'malicious_frac') and args.malicious_frac > 0 else "0.0"
            # Map dataset name: 'cifar' -> 'cifar10' for filename
            dataset_raw = args.dataset if hasattr(args, 'dataset') else 'mnist'
            dataset_str = 'cifar10' if dataset_raw == 'cifar' else dataset_raw
            filename_base = "fedavg-{}-{}-{}-{}".format(dataset_str, attack_str, mal_frac_str, dateNow)
            
            accDf = pd.DataFrame({'fedavg': acc_test_list})
            accDf.to_csv("../data/result/{}.csv".format(filename_base), index=False, sep=',')
            
            lossDf = pd.DataFrame({'fedavg': loss_test_list})
            lossDf.to_csv("../data/result/{}_loss.csv".format(filename_base), index=False, sep=',')
            
            # 保存聚合后的模型
            aggEchoParasFile = './serverS/paras/agg/' + taskID + '-epoch-' + str(currentEpoch) + '.pkl'
            torch.save(w_glob, aggEchoParasFile)
            
            # 上传聚合模型到IPFS
            print(f"\nUploading aggregated model to IPFS...")
            aggEchoFileHash = None
            while aggEchoFileHash is None:
                hash_value, sttCodeAdd = usefulTools.ipfsAddFile(aggEchoParasFile)
                if sttCodeAdd == 0:
                    aggEchoFileHash = hash_value.strip()
                    print(f'Aggregated model uploaded to IPFS: {aggEchoFileHash}')
                else:
                    print(f'Failed to upload aggregated model, retrying...')
                    time.sleep(2)
            
            # 发布聚合模型
            taskStatus = 'training'
            if currentEpoch == args.epochs:
                taskStatus = 'done'
            
            print(f"Publishing aggregated model (epoch {currentEpoch}, status: {taskStatus})...")
            while True:
                epochAggModelPublish = subprocess.Popen(
                    args=['../commonComponent/interRun.sh aggregated ' + taskID + ' ' + str(currentEpoch) + ' ' + taskStatus + ' ' + aggEchoFileHash],
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'
                )
                aggPubOuts, aggPubErrs = epochAggModelPublish.communicate(timeout=10)
                if epochAggModelPublish.poll() == 0:
                    print(f'Aggregated model published!')
                    print(f'Details: {aggPubOuts.strip()}\n')
                    break
                else:
                    print(f'Failed to publish aggregated model, retrying...')
                    print(f'Error: {aggPubErrs}')
                    time.sleep(2)
            
            currentEpoch += 1
        
        print('\n' + '=' * 80)
        print(f'Iteration {iteration_count} completed')
        print('=' * 80 + '\n')
        
        iteration_count += 1
        
        # 可以选择是否继续下一轮迭代
        # 如果需要只运行一次，可以break
        # break


if __name__ == '__main__':
    main()

