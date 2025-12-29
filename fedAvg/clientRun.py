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
import warnings

# 抑制 OpenBLAS 警告
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# 过滤 OpenBLAS 相关警告
warnings.filterwarnings('ignore', message='.*OpenBLAS.*')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

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


def main(aim_addr='192.168.137.208'):
    """
    主函数
    在shard模式下，设备列表会自动从dict_users文件中获取
    
    Args:
        aim_addr: 服务器地址（用于文档和一致性，FedAvg主要通过Fabric通信）
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
    
    # 设置Fabric网络主机地址
    # 客户端需要连接到运行Fabric网络的服务器地址（默认192.168.137.208）
    import re
    
    # 优先使用命令行参数
    if args.fabric_host is not None:
        fabric_host = args.fabric_host
        os.environ['FABRIC_HOST'] = fabric_host
        print(f"✓ Set FABRIC_HOST to {fabric_host} (from --fabric_host parameter)")
    elif os.environ.get('FABRIC_HOST'):
        fabric_host = os.environ.get('FABRIC_HOST')
        print(f"✓ Using FABRIC_HOST from environment: {fabric_host}")
    else:
        # 默认使用服务器地址（Fabric应该在服务器上运行）
        fabric_host = '192.168.137.208'
        os.environ['FABRIC_HOST'] = fabric_host
        print(f"✓ Using default FABRIC_HOST: {fabric_host} (server address)")
    
    # 如果使用IP地址，检查/etc/hosts配置和证书问题
    if re.match(r'^\d+\.\d+\.\d+\.\d+$', fabric_host) and fabric_host != '127.0.0.1':
        print(f"\n⚠ NOTE: Using IP address {fabric_host} for Fabric connection")
        print(f"  Checking /etc/hosts configuration...")
        
        # 检查/etc/hosts配置
        hosts_configured = False
        try:
            with open('/etc/hosts', 'r') as f:
                hosts_content = f.read()
                # 检查是否包含所需的域名映射
                required_domains = ['peer0.org1.example.com', 'peer0.org2.example.com', 'orderer.example.com']
                if all(domain in hosts_content for domain in required_domains):
                    # 检查IP地址是否匹配
                    pattern = rf'^{re.escape(fabric_host)}\s+.*peer0\.org1\.example\.com'
                    if re.search(pattern, hosts_content, re.MULTILINE):
                        hosts_configured = True
                        print(f"  ✓ /etc/hosts is properly configured")
                    else:
                        print(f"  ✗ /etc/hosts contains domains but IP address doesn't match")
                else:
                    print(f"  ✗ /etc/hosts missing required domain mappings")
        except PermissionError:
            print(f"  ⚠ Cannot read /etc/hosts (permission denied)")
        except Exception as e:
            print(f"  ⚠ Error checking /etc/hosts: {e}")
        
        if not hosts_configured:
            print(f"\n  REQUIRED ACTION: Configure /etc/hosts with:")
            print(f"    {fabric_host} peer0.org1.example.com peer0.org2.example.com orderer.example.com")
            print(f"  Run this command (requires sudo):")
            print(f"    sudo bash -c 'echo \"{fabric_host} peer0.org1.example.com peer0.org2.example.com orderer.example.com\" >> /etc/hosts'")
            print(f"  This is REQUIRED for TLS certificate validation.\n")
        else:
            print(f"\n  ⚠ IMPORTANT: TLS Certificate Issue")
            print(f"  When connecting to remote Fabric ({fabric_host}), you may encounter TLS certificate")
            print(f"  verification errors because client and server use different certificates.")
            print(f"\n  RECOMMENDED SOLUTION: Use SSH tunnel (recommended)")
            print(f"    1. In another terminal, run:")
            print(f"       ssh -L 7051:localhost:7051 -L 9051:localhost:9051 -L 7050:localhost:7050 pi@{fabric_host}")
            print(f"    2. Then run clientRun.py with --fabric_host localhost")
            print(f"       python3 clientRun.py --dataset mnist --model cnn --fabric_host localhost")
            print(f"\n  ALTERNATIVE: Copy certificates from server")
            print(f"    Copy Fabric certificates from server to client:")
            print(f"    scp -r pi@{fabric_host}:/home/pi/fabric/fabric-samples/test-network/organizations /home/pi/fabric/fabric-samples/test-network/")
            print(f"    (This ensures client uses the same certificates as server)\n")
    
    # 从环境变量读取shard_id（如果命令行未指定）
    if args.shard_id is None:
        env_shard_id = os.environ.get('SHARD_ID')
        if env_shard_id is not None:
            try:
                args.shard_id = int(env_shard_id)
                print(f"✓ Loaded shard_id from environment variable SHARD_ID: {args.shard_id}")
            except ValueError:
                print(f"Warning: Invalid SHARD_ID environment variable: {env_shard_id}")
    
    # shard_id必须指定（通过命令行参数或环境变量）
    if args.shard_id is None:
        print(f"✗ Error: shard_id must be specified (via --shard_id argument or SHARD_ID environment variable)!")
        return
    
    # 加载对应的 dict_users 文件以确定用户范围
    dict_users_file = f'../commonComponent/dict_users_{args.dataset}_shard{args.shard_id}.pkl'
    if not os.path.exists(dict_users_file):
        print(f"✗ Error: Shard file {dict_users_file} not found!")
        return
    
    with open(dict_users_file, 'rb') as f:
        dict_users = pickle.load(f)
    print(f"✓ Pre-loaded dict_users from shard-specific file: {dict_users_file}")
    user_ids = sorted(list(dict_users.keys()))
    print(f"  Shard {args.shard_id}: {len(dict_users)} users (user IDs: {user_ids[0]}-{user_ids[-1]})")
    # 根据dict_users中的最大用户ID设置num_users（确保覆盖所有用户）
    # 例如：shard0: 0-99, shard1: 100-199, shard2: 200-299
    max_user_id = max(user_ids)
    args.num_users = max_user_id + 1
    print(f"  Setting num_users={args.num_users} based on dict_users (max user ID: {max_user_id})")
    
    # 构建模型
    net_glob, args, dataset_train, dataset_test, dict_users_build = buildModels.modelBuild()
    net_glob.train()
    
    # 重新加载shard特定的dict_users（覆盖buildModels生成的）
    with open(dict_users_file, 'rb') as f:
        dict_users = pickle.load(f)
    
    # 初始化设备列表（确保覆盖dict_users中的所有用户ID）
    allDeviceName = []
    max_user_id = max(dict_users.keys())
    for i in range(max_user_id + 1):
        allDeviceName.append("device" + ("{:0>5d}".format(i)))
    
    # 从dict_users文件获取所有可用的设备列表（shard模式）
    # 这是所有可用的设备池，每个epoch会从中随机选择一部分进行训练
    all_available_idxs = sorted(list(dict_users.keys()))
    print(f"✓ Available devices from dict_users file: {len(all_available_idxs)} devices")
    print(f"  Device indices range: {all_available_idxs[0]}-{all_available_idxs[-1]}")
    
    if len(all_available_idxs) == 0:
        print(f"Error: No valid device IDs found in dict_users file!")
        return
    
    # 客户端将从Server查询每轮参与训练的设备列表
    print(f"Client node initialized: {len(all_available_idxs)} available devices in this shard")
    print(f"  Note: Device selection is controlled by Server, client will query device list from Fabric")
    
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
            
            # === 从Fabric查询Server选择的设备列表 ===
            print(f"Querying device list for epoch {currentEpoch} from Server...")
            selected_devices = None
            query_start = time.time()
            
            while time.time() - query_start < 120:  # 最多等待120秒
                try:
                    devices_key = f"{taskID}-epoch{currentEpoch}-devices"
                    result, status = usefulTools.simpleQuerySilent(devices_key)
                    if status == 0:
                        devices_info = json.loads(result)
                        if int(devices_info.get('epoch', -1)) == currentEpoch:
                            selected_devices = devices_info.get('devices', [])
                            print(f"✓ Received device list from Server: {len(selected_devices)} devices")
                            print(f"  Server selected devices: {selected_devices}")
                            break
                except Exception as e:
                    pass
                print(f"Waiting for Server to publish device list for epoch {currentEpoch}...")
                time.sleep(5)
            
            if not selected_devices:
                print(f"Error: Could not get device list from Server after 120s, skipping epoch...")
                time.sleep(10)
                currentEpoch += 1
                continue
            
            # 过滤：只训练属于本shard的设备
            # 从设备名转换为设备索引
            idxs_users = []
            for device_name in selected_devices:
                try:
                    idx = int(device_name.replace("device", ""))
                    if idx in all_available_idxs:  # 只处理本shard的设备
                        idxs_users.append(idx)
                except:
                    pass
            
            if len(idxs_users) == 0:
                print(f"No devices in this shard selected for epoch {currentEpoch}, skipping...")
                time.sleep(5)
                currentEpoch += 1
                continue
            
            print(f"\nEpoch {currentEpoch}: Training {len(idxs_users)} devices from this shard (out of {len(selected_devices)} total selected)")
            print(f"  Device indices to train: {idxs_users}")
            
            # 选择恶意节点（用于攻击实验）
            malicious_nodes = select_malicious_nodes(
                idxs_users,
                args.malicious_frac if hasattr(args, 'malicious_frac') else 0.0,
                args.attack_type if hasattr(args, 'attack_type') else 'none',
                seed=None  # 不使用seed，保证随机性
            )
            
            loss_locals = []
            
            # 遍历选中的设备进行训练
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
                    # 不使用固定的seed，保证随机性
                    label_mapping = create_labelflip_mapping(
                        num_classes=args.num_classes,
                        mode=args.labelflip_mode if hasattr(args, 'labelflip_mode') else 'random',
                        target_label=args.labelflip_target if hasattr(args, 'labelflip_target') else None,
                        seed=None  # 不使用seed，保证随机性
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
    # 参考DAG方法，支持指定服务器地址
    # 注意：FedAvg主要通过Fabric通信，aim_addr主要用于文档和一致性
    # Fabric地址通过--fabric_host参数或FABRIC_HOST环境变量设置
    # 默认使用与DAG方法相同的服务器地址
    main('192.168.137.208')

