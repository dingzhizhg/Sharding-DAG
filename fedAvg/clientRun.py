# -*- coding: utf-8 -*-

"""
FedAvg Client (Training Node)
每个客户端节点运行此脚本，负责：
1. 查询任务
2. 下载全局模型
3. 本地训练
4. 上传模型到IPFS和区块链
"""

import sys
import os
import time
import shutil
import copy
import pickle
import subprocess
import json
import numpy as np

# Common Components
sys.path.append('../commonComponent')
import usefulTools

# FL related
sys.path.append('../federatedLearning')
import torch
import buildModels
from models.Update import LocalUpdate
from models.test import test_img
from models.Attacks import DelayAttackManager, noise_attack, create_labelflip_mapping
from utils.attack_utils import select_malicious_nodes, is_malicious_node, is_lazy_node
from utils.options import args_parser


def main():
    """
    主函数
    每个client在每次iteration时会随机选择device，不再需要外部指定device列表
    """
    # 创建必要的目录
    if os.path.exists('./clientS'):
        shutil.rmtree('./clientS')
    os.makedirs('./clientS', exist_ok=True)
    os.makedirs('./clientS/paras', exist_ok=True)
    os.makedirs('./clientS/local', exist_ok=True)
    
    # 解析训练参数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # 构建模型
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()
    
    # 加载dict_users（应该与服务器端一致，使用数据集特定的文件名）
    dict_users_file = f'../commonComponent/dict_users_{args.dataset}.pkl'
    if os.path.exists(dict_users_file):
        with open(dict_users_file, 'rb') as f:
            dict_users = pickle.load(f)
        print(f"Loaded dict_users from {dict_users_file}")
    else:
        print(f"Warning: dict_users file not found at {dict_users_file}, using default split")
    
    # 初始化设备列表
    allDeviceName = []
    for i in range(args.num_users):
        allDeviceName.append("device" + ("{:0>5d}".format(i)))
    
    # 计算每个树莓派负责的设备范围
    device_start_idx = None
    device_end_idx = None
    if args.node_id is not None:
        device_start_idx = args.node_id * args.devices_per_node
        device_end_idx = device_start_idx + args.devices_per_node
        if device_end_idx > args.num_users:
            device_end_idx = args.num_users
        print(f"Client node {args.node_id} initialized. Device range: {device_start_idx} to {device_end_idx-1} (device{allDeviceName[device_start_idx]} to device{allDeviceName[device_end_idx-1]})")
        print(f"Will randomly select {int(args.frac * args.devices_per_node)} devices from this range per iteration.")
    else:
        print(f"Client node initialized. Will randomly select devices from all {args.num_users} devices for each iteration.")
    
    # 初始化delay attack manager（如果需要）
    delay_manager = None
    if 'delay' in args.attack_type:
        delay_manager = DelayAttackManager()
    
    # 检查是否是新任务
    checkTaskID = ''
    
    # 主循环：等待任务并训练
    iteration = 0
    while True:
        # 查询任务发布信息
        taskRelInfo = {}
        print('\n' + '=' * 80)
        print('Waiting for new task...')
        print('=' * 80)
        
        while True:
            taskRelQue, taskRelQueStt = usefulTools.simpleQuery('taskRelease')
            if taskRelQueStt == 0:
                taskRelInfo = json.loads(taskRelQue)
                print('\n' + '=' * 80)
                print('Latest task release status: %s' % taskRelQue.strip())
                print('=' * 80 + '\n')
                break
            time.sleep(5)
        
        taskRelEpoch = int(taskRelInfo['epoch'])
        taskID = taskRelInfo['taskID']
        
        print(f'Current task: {taskID}')
        
        # 查询任务信息
        taskInfo = {}
        while True:
            taskInQue, taskInQueStt = usefulTools.simpleQuery(taskID)
            if taskInQueStt == 0:
                taskInfo = json.loads(taskInQue)
                print('Latest task info: %s' % taskInQue.strip())
                print('=' * 80 + '\n')
                break
            time.sleep(2)
        
        # 如果任务已完成或是重复任务，跳过
        if taskInfo['status'] == 'done' or checkTaskID == taskID:
            print(f'*** {taskID} has been completed or is a duplicate! ***\n')
            time.sleep(5)
            continue
        
        # 开始新的迭代
        print('\n' + '=' * 80)
        print(f'Iteration #{iteration} starting')
        print('=' * 80 + '\n')
        
        # 每个iteration随机选择device
        if args.node_id is not None:
            # 从当前节点负责的设备范围内选择
            devices_in_range = args.devices_per_node
            m = max(int(args.frac * devices_in_range), 1)  # 从当前节点的设备中选择
            device_range = range(device_start_idx, device_end_idx)
            idxs_users = np.random.choice(list(device_range), m, replace=False)
        else:
            # 从全部设备中选择（默认行为）
            m = max(int(args.frac * args.num_users), 1)  # args.frac is the fraction of users
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        print('\n**************************** Selected devices for this iteration *****************************')
        print('Device indices: ', idxs_users.tolist())
        print('Device IDs: ', [allDeviceName[idx] for idx in idxs_users])
        print('**********************************************************************************************\n')
        
        currentEpoch = int(taskInfo['epoch']) + 1
        acc_test_list = []  # 存储每个epoch的测试准确率
        
        # 训练循环：每个epoch
        while currentEpoch <= taskRelEpoch:
            print('\n' + '-' * 80)
            print(f'Epoch {currentEpoch}/{taskRelEpoch}')
            print('-' * 80 + '\n')
            
            # 查询当前epoch的任务信息
            while True:
                taskInQueEpo, taskInQueEpoStt = usefulTools.simpleQuery(taskID)
                if taskInQueEpoStt == 0:
                    taskInfoEpo = json.loads(taskInQueEpo)
                    if int(taskInfoEpo['epoch']) == (currentEpoch - 1):
                        print(f'Task info for epoch {currentEpoch - 1}: {taskInQueEpo}')
                        break
                    else:
                        print(f'Waiting for epoch {currentEpoch - 1} to be published...')
                        time.sleep(10)
                else:
                    time.sleep(5)
            
            # 下载聚合模型
            aggBasModFil = './clientS/paras/aggModel-iter-' + str(iteration) + '-epoch-' + str(currentEpoch - 1) + '.pkl'
            print(f"Downloading aggregated model for epoch {currentEpoch - 1}...")
            while True:
                aggBasMod, aggBasModStt = usefulTools.ipfsGetFile(taskInfoEpo['paras'], aggBasModFil)
                if aggBasModStt == 0:
                    print(f'Aggregated model downloaded successfully!')
                    break
                else:
                    print(f'Failed to download aggregated model, retrying...')
                    time.sleep(2)
            
            # 加载模型
            w_glob = net_glob.state_dict()
            net_glob.load_state_dict(torch.load(aggBasModFil))
            
            # 计算上一个epoch的准确率（如果存在）
            if currentEpoch > 1:
                net_glob.eval()
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                acc_test_value = acc_test.cpu().numpy().tolist()
                acc_test_list.append(acc_test_value)
                print(f'Epoch {currentEpoch - 1} test accuracy: {acc_test_value:.2f}%')
            
            # 选择恶意节点（用于攻击实验）
            malicious_nodes = select_malicious_nodes(
                idxs_users,
                args.malicious_frac if hasattr(args, 'malicious_frac') else 0.0,
                args.attack_type if hasattr(args, 'attack_type') else 'none',
                seed=args.seed if hasattr(args, 'seed') else None
            )
            
            loss_locals = []
            
            # 遍历所有设备进行训练
            for idx in idxs_users:
                device_id = allDeviceName[idx]
                is_mal = is_malicious_node(idx, malicious_nodes)
                is_lazy = is_lazy_node(idx, malicious_nodes, 
                                      args.attack_type if hasattr(args, 'attack_type') else 'none')
                
                # Lazy攻击：跳过训练和上传
                if is_lazy:
                    print(f'Lazy node {device_id} skipped in epoch {currentEpoch}')
                    continue
                
                # 创建label mapping（用于labelflip攻击）
                label_mapping = None
                if is_mal and hasattr(args, 'attack_type') and 'labelflip' in args.attack_type:
                    labelflip_seed = (args.seed if hasattr(args, 'seed') else 1) * 1000 + idx * 100 + currentEpoch
                    label_mapping = create_labelflip_mapping(
                        num_classes=args.num_classes,
                        mode=args.labelflip_mode if hasattr(args, 'labelflip_mode') else 'random',
                        target_label=args.labelflip_target if hasattr(args, 'labelflip_target') else None,
                        seed=labelflip_seed
                    )
                    print(f'Labelflip attack: {device_id} using {args.labelflip_mode} mode')
                
                # 本地训练
                print(f"Training local model on {device_id}...")
                net_glob.train()
                
                # Delay攻击或正常训练
                if is_mal and hasattr(args, 'attack_type') and 'delay' in args.attack_type:
                    # 正常训练用于delay攻击
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], label_mapping=label_mapping)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    # 保存当前模型到delay队列
                    if delay_manager is not None:
                        delay_manager.save_delayed_model(idx, currentEpoch, w)
                        # 获取延迟的模型（来自delay_rounds轮之前）
                        w_delayed = delay_manager.get_delayed_model(idx, currentEpoch, 
                                                                   args.delay_rounds if hasattr(args, 'delay_rounds') else 2)
                        if w_delayed is not None:
                            w = w_delayed
                            print(f'Delay attack: {device_id} submitted delayed model from epoch {currentEpoch - args.delay_rounds}')
                else:
                    # 正常训练
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], label_mapping=label_mapping)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                
                # 应用noise攻击（如果启用）
                if is_mal and hasattr(args, 'attack_type') and 'noise' in args.attack_type:
                    w = noise_attack(w, args.noise_scale if hasattr(args, 'noise_scale') else 0.1)
                
                loss_locals.append(copy.deepcopy(loss))
                
                # 保存本地模型
                devLocFile = './clientS/local/' + taskID + '-' + device_id + '-epoch-' + str(currentEpoch) + '.pkl'
                torch.save(w, devLocFile)
                print(f'Local model saved: {devLocFile}')
                
                # 上传到IPFS
                print(f"Uploading {device_id} model to IPFS...")
                upload_success = False
                start_time = time.time()
                localAdd = None
                
                while not upload_success and (time.time() - start_time) < args.lazy_timeout:
                    localAdd, localAddStt = usefulTools.ipfsAddFile(devLocFile)
                    if localAddStt == 0:
                        print(f'{device_id} model uploaded to IPFS: {localAdd}')
                        upload_success = True
                        break
                    else:
                        print(f'Failed to upload {device_id} model to IPFS, retrying...')
                        time.sleep(1)
                
                if not upload_success:
                    print(f'Timeout: Failed to upload {device_id} model after {args.lazy_timeout}s')
                    continue
                
                # 发布到区块链
                print(f"Publishing {device_id} model to blockchain...")
                release_success = False
                start_time = time.time()
                
                while not release_success and (time.time() - start_time) < args.lazy_timeout:
                    try:
                        localRelease = subprocess.Popen(
                            args=['../commonComponent/interRun.sh local ' + device_id + ' ' + taskID + ' ' + str(currentEpoch) + ' ' + localAdd],
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'
                        )
                        localOuts, localErrs = localRelease.communicate(timeout=10)
                        if localRelease.poll() == 0:
                            print(f'{device_id} model published to blockchain!')
                            print(f'Details: {localOuts.strip()}')
                            release_success = True
                            break
                        else:
                            print(f'Failed to publish {device_id} model: {localErrs.strip()}')
                    except subprocess.TimeoutExpired:
                        localRelease.kill()
                        print(f'Timeout: Failed to publish {device_id} model')
                    except Exception as e:
                        print(f'Error publishing {device_id} model: {e}')
                    
                    time.sleep(2)
                
                if not release_success:
                    print(f'Timeout: Failed to publish {device_id} model after {args.lazy_timeout}s')
                
                print(f'Training of {device_id} in epoch {currentEpoch} has done!')
            
            # 清理旧的延迟模型
            if delay_manager is not None:
                delay_manager.clear_old_models(currentEpoch, 
                                              max_delay=args.delay_rounds if hasattr(args, 'delay_rounds') else 2)
            
            # 计算平均损失
            if len(loss_locals) > 0:
                loss_avg = sum(loss_locals) / len(loss_locals)
                print(f'Epoch {currentEpoch} average loss: {loss_avg:.4f}')
            
            currentEpoch += 1
        
        # 计算最后一个epoch的准确率
        print("\nWaiting for final epoch aggregated model...")
        while True:
            taskInQueFinal, taskInQueFinalStt = usefulTools.simpleQuery(taskID)
            if taskInQueFinalStt == 0:
                taskInfoFinal = json.loads(taskInQueFinal)
                if int(taskInfoFinal['epoch']) == taskRelEpoch:
                    # 下载最后一个epoch的聚合模型
                    finalAggModelFile = './clientS/paras/aggModel-iter-' + str(iteration) + '-epoch-' + str(taskRelEpoch) + '-final.pkl'
                    while True:
                        finalAggModel, finalAggModelStt = usefulTools.ipfsGetFile(taskInfoFinal['paras'], finalAggModelFile)
                        if finalAggModelStt == 0:
                            break
                        else:
                            time.sleep(1)
                    
                    # 加载模型并计算最后一个epoch的准确率
                    net_glob.load_state_dict(torch.load(finalAggModelFile))
                    net_glob.eval()
                    acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
                    acc_test_final_value = acc_test_final.cpu().numpy().tolist()
                    acc_test_list.append(acc_test_final_value)
                    print(f'Final epoch {taskRelEpoch} test accuracy: {acc_test_final_value:.2f}%')
                    break
                else:
                    time.sleep(5)
        
        checkTaskID = taskID
        
        print('\n' + '=' * 80)
        print(f'Iteration #{iteration} completed')
        print('=' * 80 + '\n')
        
        iteration += 1


if __name__ == '__main__':
    main()

