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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

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
# from utils.attack_utils import select_malicious_nodes, is_lazy_node  # 注释：fedavg不需要恶意节点检测功能
import datetime


def query_device_model(deviceID, taskID, currentEpoch, lock, received_devices, failed_devices, w_locals, 
                       net_glob, dataset_test, args, aggModelAcc):
    """
    查询单个设备的模型信息（从Fabric）并下载模型文件（从IPFS）
    
    Args:
        deviceID: 设备ID
        taskID: 任务ID
        currentEpoch: 当前epoch
        lock: 线程锁
        received_devices: 已接收设备的集合
        failed_devices: 查询失败的设备集合（用于避免重复查询不存在的设备）
        w_locals: 本地模型权重列表
        net_glob: 全局模型
        dataset_test: 测试数据集
        args: 参数
        aggModelAcc: 当前聚合模型准确率
    
    Returns:
        bool: 是否成功获取模型
    """
    # 如果已经处理过，跳过
    with lock:
        if deviceID in received_devices:
            return False
        # 如果已经标记为查询失败，也跳过（避免重复查询不存在的设备）
        if deviceID in failed_devices:
            return False
    
    # 构建Fabric key: deviceID-taskID-epoch
    fabric_key = f"{deviceID}-{taskID}-{currentEpoch}"
    
    try:
        # 查询Fabric（使用静默模式，避免打印"设备不存在"的错误信息）
        model_info_str, status = usefulTools.simpleQuerySilent(fabric_key)
        
        if status != 0:
            # 模型尚未上传，标记为查询失败，避免重复查询
            with lock:
                failed_devices.add(deviceID)
            return False
        
        # 解析模型信息
        model_info = json.loads(model_info_str)
        
        # 验证任务ID和epoch是否匹配
        if model_info.get('taskID') != taskID or int(model_info.get('epoch', -1)) != currentEpoch:
            print(f"Warning: {deviceID} model info mismatch (expected taskID={taskID}, epoch={currentEpoch})")
            return False
        
        # 获取IPFS hash
        ipfs_hash = model_info.get('paras')
        if not ipfs_hash:
            print(f"Error: {deviceID} model info missing IPFS hash")
            return False
        
        # 下载模型文件
        localFileName = './serverS/paras/local/' + taskID + '-' + deviceID + '-epoch-' + str(currentEpoch) + '.pkl'
        os.makedirs(os.path.dirname(localFileName), exist_ok=True)
        
        download_success = False
        for retry in range(3):  # 最多重试3次
            fileGetStatus, sttCodeGet = usefulTools.ipfsGetFile(ipfs_hash, localFileName)
            if sttCodeGet == 0:
                download_success = True
                break
            else:
                if retry < 2:
                    time.sleep(1)
        
        if not download_success:
            print(f"Error: Failed to download {deviceID} model from IPFS after 3 retries")
            return False
        
        # 加载并验证模型
        try:
            canddts_dev_pas = torch.load(localFileName, map_location=torch.device('cpu'))
            acc_canddts_dev, loss_canddts_dev = buildModels.evalua(net_glob, canddts_dev_pas, dataset_test, args)
            acc_canddts_dev = acc_canddts_dev.cpu().numpy().tolist()
            
            # 质量检查：如果准确率低于聚合模型太多，认为是恶意节点
            # 注释：fedavg不需要恶意节点检测功能，所有模型都会被聚合
            # with lock:
            #     if len(w_locals) > 0 and (acc_canddts_dev - aggModelAcc) < -20:
            #         print(f"{deviceID} detected as potentially malicious (low accuracy: {acc_canddts_dev:.2f}%), skipping...")
            #         received_devices.add(deviceID)
            #         return False
            
            # 添加到聚合列表（fedavg接受所有模型）
            with lock:
                w_locals.append(copy.deepcopy(canddts_dev_pas))
                received_devices.add(deviceID)
                print(f"✓ Added model from {deviceID} (accuracy: {acc_canddts_dev:.2f}%)")
                return True
                
        except Exception as e:
            print(f"Error loading/validating {deviceID} model: {e}")
            with lock:
                received_devices.add(deviceID)  # 标记为已处理，避免重复尝试
            return False
            
    except json.JSONDecodeError as e:
        print(f"Error parsing {deviceID} model info from Fabric: {e}")
        return False
    except Exception as e:
        print(f"Error querying {deviceID} model: {e}")
        return False


def query_device_model_dag_style(deviceID, taskID, currentEpoch, lock, flagSet, w_locals, net_glob, dataset_test, args, aggModelAcc, timeout=15):
    """
    参考DAG方法的查询逻辑：查询单个设备模型（从Fabric）并下载模型文件（从IPFS）
    成功后将设备ID添加到flagSet中，表示已查询成功
    
    Args:
        deviceID: 设备ID
        taskID: 任务ID
        currentEpoch: 当前epoch
        lock: 线程锁
        flagSet: 成功查询的设备集合（参考DAG方法）
        w_locals: 本地模型权重列表
        net_glob: 全局模型
        dataset_test: 测试数据集
        args: 参数
        aggModelAcc: 当前聚合模型准确率
        timeout: 超时时间（秒）
    """
    # 构建Fabric key: deviceID-taskID-epoch
    fabric_key = f"{deviceID}-{taskID}-{currentEpoch}"
    
    try:
        # 查询Fabric（参考DAG方法的queryLocal实现）
        localQuery = subprocess.Popen(
            args=['../commonComponent/interRun.sh query ' + fabric_key], 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            encoding='utf-8'
        )
        outs, errs = localQuery.communicate(timeout=timeout)
        
        if localQuery.poll() == 0:
            # 解析模型信息
            localDetail = json.loads(outs.strip())
            
            # 验证任务ID和epoch是否匹配（参考DAG方法）
            if localDetail.get('taskID') == taskID and int(localDetail.get('epoch', -1)) == currentEpoch:
                print(f"The query result of the {deviceID} is {outs.strip()}")
                
                # 获取IPFS hash
                ipfs_hash = localDetail.get('paras')
                if not ipfs_hash:
                    print(f"Error: {deviceID} model info missing IPFS hash")
                    return
                
                # 下载模型文件（参考DAG方法，使用循环重试，但添加最大重试次数避免无限循环）
                localFileName = './serverS/paras/local/' + taskID + '-' + deviceID + '-epoch-' + str(currentEpoch) + '.pkl'
                os.makedirs(os.path.dirname(localFileName), exist_ok=True)
                
                # 添加最大重试次数（3次），避免无限循环
                download_success = False
                for retry in range(3):  # 最多重试3次
                    fileGetStatus, sttCodeGet = usefulTools.ipfsGetFile(ipfs_hash, localFileName)
                    if sttCodeGet == 0:
                        download_success = True
                        break
                    else:
                        if retry < 2:  # 不是最后一次重试，等待1秒
                            time.sleep(1)
                
                if not download_success:
                    print(f"Error: Failed to download {deviceID} model from IPFS after 3 retries")
                    return  # 下载失败，直接返回，不添加到flagSet
                
                # 加载并验证模型
                try:
                    canddts_dev_pas = torch.load(localFileName, map_location=torch.device('cpu'))
                    acc_canddts_dev, loss_canddts_dev = buildModels.evalua(net_glob, canddts_dev_pas, dataset_test, args)
                    acc_canddts_dev = acc_canddts_dev.cpu().numpy().tolist()
                    
                    # 添加到聚合列表（fedavg接受所有模型，不进行恶意节点检测）
                    with lock:
                        w_locals.append(copy.deepcopy(canddts_dev_pas))
                        flagSet.add(deviceID)  # 参考DAG方法，成功后将设备ID添加到flagSet
                        print(f"✓ Added model from {deviceID} (accuracy: {acc_canddts_dev:.2f}%)")
                except Exception as e:
                    print(f"Error loading/validating {deviceID} model: {e}")
        # else:
        #     print(f"Failed to query {deviceID}: {errs}")
    except subprocess.TimeoutExpired:
        localQuery.kill()
        print(f"Timeout: Failed to query {deviceID} within {timeout}s")
    except Exception as e:
        print(f"Error querying {deviceID}: {e}")


def publish_epoch_devices(taskID, currentEpoch, selected_devices, timeout=15):
    """
    发布当前epoch选择的设备列表到Fabric
    
    Args:
        taskID: 任务ID
        currentEpoch: 当前epoch
        selected_devices: 选中的设备名称列表
        timeout: 超时时间（秒）
    
    Returns:
        bool: 是否发布成功
    """
    # 构建JSON数组字符串（需要转义）
    devices_json = json.dumps(selected_devices).replace('"', '\\"')
    
    try:
        devices_cmd = f'../commonComponent/interRun.sh epochdevices {taskID} {currentEpoch} "{devices_json}"'
        devices_pub = subprocess.Popen(
            args=[devices_cmd],
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'
        )
        pub_outs, pub_errs = devices_pub.communicate(timeout=timeout)
        if devices_pub.poll() == 0:
            print(f"✓ Epoch {currentEpoch} device list published: {len(selected_devices)} devices")
            print(f"  Selected devices: {selected_devices}")
            return True
        else:
            print(f"Warning: Failed to publish device list: {pub_errs}")
            return False
    except subprocess.TimeoutExpired:
        devices_pub.kill()
        print(f"Warning: Timeout publishing device list")
        return False
    except Exception as e:
        print(f"Warning: Error publishing device list: {e}")
        return False


def collect_models_from_fabric(taskID, currentEpoch, target_devices, args, net_glob, dataset_test, aggModelAcc, timeout):
    """
    从Fabric查询并收集指定设备上传的模型
    
    Args:
        taskID: 任务ID
        currentEpoch: 当前epoch
        target_devices: 目标设备名称列表（来自Server选择）
        args: 参数
        net_glob: 全局模型
        dataset_test: 测试数据集
        aggModelAcc: 当前聚合模型准确率
        timeout: 超时时间（秒）
    
    Returns:
        list: 收集到的模型权重列表
    """
    w_locals = []
    flagList = set(target_devices)  # 使用传入的目标设备列表
    lock = threading.Lock()
    start_time = time.time()
    
    print(f"Querying {len(target_devices)} target devices...")
    
    # 参考DAG方法的查询逻辑：循环查询直到flagList为空或超时
    while len(flagList) != 0:
        elapsed = time.time() - start_time
        
        # 检查超时（参考DAG方法）
        if elapsed >= timeout:
            print(f"Timeout: Stopped waiting for nodes after {elapsed:.2f}s")
            print(f"Remaining nodes (possibly not uploaded yet): {sorted(list(flagList))}")
            break
        
        # 参考DAG方法：创建flagSet来跟踪本次查询成功的设备
        flagSet = set()
        ts = []
        
        # 当目标设备数量较少时（如10个），一次性全部查询
        # 当目标设备较多时，分批查询
        max_concurrent = min(len(flagList), 20)  # 增加并发数到20
        query_batch = list(flagList)[:max_concurrent]
        
        # 参考DAG方法：为每个设备创建线程进行查询
        query_timeout = 15  # 单个查询的超时时间
        for deviceID in query_batch:
            t = threading.Thread(
                target=query_device_model_dag_style,
                args=(deviceID, taskID, currentEpoch, lock, flagSet, w_locals, net_glob, dataset_test, args, aggModelAcc),
                kwargs={'timeout': query_timeout}
            )
            t.start()
            ts.append(t)
        
        # 参考DAG方法：等待所有线程完成（带超时）
        remaining_time_for_join = max(1, min(timeout - elapsed, 60))
        for t in ts:
            t.join(timeout=remaining_time_for_join)
        
        # 等待线程完成更新flagSet
        time.sleep(1)
        
        # 参考DAG方法：从flagList中移除已成功查询的设备
        flagList = flagList - flagSet
    
    print(f"\nCollected {len(w_locals)} models from clients.")
    return w_locals


def main(arg=True):
    """
    主函数
    服务器负责初始化模型、发布任务、聚合客户端模型、发布聚合结果
    
    Args:
        arg: 是否持续运行（默认True，持续运行）
    """
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
    
    # 设置Fabric网络主机地址
    # 服务器需要连接到运行Fabric网络的地址（默认192.168.137.208）
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
    
    # 如果使用IP地址，检查/etc/hosts配置
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
            print(f"\n  ℹ Server mode: Fabric should be running locally on this machine")
            print(f"  Clients connecting to this server should use SSH tunnel or copy certificates.\n")
    
    # 服务器不需要shard_id，它聚合所有客户端（不管来自哪个shard）的模型
    # shard_id只在客户端节点使用，用于加载对应shard的数据
    
    # 构建模型
    print("=" * 80)
    print("Building model...")
    print("Server mode: Will aggregate models from ALL clients (all shards)")
    print("=" * 80)
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()
    
    # 保存dict_users到文件以便客户端使用（使用默认的全局文件）
    dict_users_file = f'../commonComponent/dict_users_{args.dataset}.pkl'
    os.makedirs('../commonComponent', exist_ok=True)
    with open(dict_users_file, 'wb') as f:
        pickle.dump(dict_users, f)
    print(f"Saved dict_users to {dict_users_file}")
    print(f"Note: Server uses global dict_users (all users). Clients will use shard-specific dict_users.")
    
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
    
    # 初始化设备列表（用于识别所有可能的device名称）
    # 服务器需要知道所有可能的device ID范围，以便扫描和接收来自所有shard的客户端模型
    allDeviceName = []
    
    # 默认设备扫描范围设置
    # 默认情况：只扫描0-99的设备（用于测试，因为通常只有一个客户端节点代表0-99）
    # 如果需要扫描所有设备（生产环境），可以通过MAX_DEVICE_ID环境变量设置为更大的值或-1
    default_max_device_id = 99  # 默认只扫描0-99的设备（测试模式）
    
    env_max_device_id = os.environ.get('MAX_DEVICE_ID')
    if env_max_device_id is not None:
        try:
            if int(env_max_device_id) == -1:
                # -1 表示扫描所有设备
                max_device_id = args.num_users - 1
                print(f"⚠ Scanning ALL devices: 0-{max_device_id} (from MAX_DEVICE_ID=-1)")
            else:
                max_device_id = min(int(env_max_device_id), args.num_users - 1)
                print(f"⚠ Limited device scan range to 0-{max_device_id} (from MAX_DEVICE_ID environment variable)")
        except ValueError:
            print(f"Warning: Invalid MAX_DEVICE_ID environment variable: {env_max_device_id}, using default range 0-99")
            max_device_id = default_max_device_id
    else:
        # 默认情况：只扫描0-99的设备（测试模式）
        max_device_id = default_max_device_id
        print(f"ℹ Using default test mode: scanning devices 0-99 (set MAX_DEVICE_ID=-1 to scan all {args.num_users} devices)")
    
    # 原始逻辑：扫描所有设备（已注释，保留用于参考）
    # for i in range(args.num_users):
    #     allDeviceName.append("device" + ("{:0>5d}".format(i)))
    
    # 新逻辑：根据max_device_id限制扫描范围
    for i in range(max_device_id + 1):
        allDeviceName.append("device" + ("{:0>5d}".format(i)))
    
    print(f"Server will scan for models from devices: device00000 to {allDeviceName[-1]} (total {len(allDeviceName)} devices)")
    
    # 训练循环
    iteration_count = 0
    acc_test_list = []
    loss_test_list = []
    
    while arg:
        print('\n' + '=' * 80)
        print(f'Iteration {iteration_count} starting')
        print('=' * 80 + '\n')
        
        # Server选择设备，通过Fabric通知Client
        print("Server mode: Server will select devices for each epoch and publish to Fabric")
        
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
            
            # === Server选择本轮参与的设备 ===
            # 从所有设备中随机选择 frac 比例的设备
            m = max(int(args.frac * len(allDeviceName)), 1)
            selected_indices = np.random.choice(len(allDeviceName), size=m, replace=False)
            selected_devices = [allDeviceName[i] for i in sorted(selected_indices)]
            
            print(f"Server selected {len(selected_devices)} devices for epoch {currentEpoch}")
            
            # 发布设备列表到Fabric，让客户端知道要训练哪些设备
            publish_success = False
            for retry in range(3):
                if publish_epoch_devices(taskID, currentEpoch, selected_devices):
                    publish_success = True
                    break
                time.sleep(2)
            
            if not publish_success:
                print(f"ERROR: Failed to publish device list after 3 retries, retrying epoch...")
                time.sleep(5)
                continue
            
            # 等待客户端训练并上传模型
            print(f"\nWaiting for clients to train and upload models...")
            time.sleep(5)  # 给客户端一些时间开始
            
            # 通过Fabric查询收集客户端上传的模型（只查询选中的设备）
            print(f"Querying Fabric for client models (timeout: {args.lazy_timeout}s)...")
            print(f"Using key format: deviceID-{taskID}-{currentEpoch}")
            w_locals = collect_models_from_fabric(
                taskID, currentEpoch, selected_devices, args, net_glob, dataset_test, aggModelAcc, args.lazy_timeout
            )
            
            # 验证收到的模型数量
            if len(w_locals) == 0:
                print(f"\nERROR: No models available for aggregation in epoch {currentEpoch}!")
                print(f"This may happen if:")
                print(f"  1. Clients haven't uploaded models yet (wait longer)")
                print(f"  2. Clients failed to upload models (check client logs)")
                print(f"  3. Fabric network connectivity issues (check network)")
                print(f"Retrying epoch {currentEpoch}...")
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
    # 参考DAG方法，main函数接受arg参数控制是否持续运行
    # 参数解析在main()函数内部通过args_parser()完成
    main(True)

