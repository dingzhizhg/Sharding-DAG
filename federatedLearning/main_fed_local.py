# -*- coding: utf-8 -*-

import os
# Suppress OpenBLAS warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import copy
import numpy as np
import math
import time
import shutil
from torchvision import datasets, transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from json import dumps
import sys
import json
import pickle
import subprocess
import datetime

# Common Components
sys.path.append('../commonComponent')
import usefulTools

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from models.Attacks import DelayAttackManager, noise_attack, create_labelflip_mapping
from utils.attack_utils import select_malicious_nodes, is_malicious_node, is_lazy_node
import buildModels


if __name__ == '__main__':

    if os.path.exists('./data/local'):
        shutil.rmtree('./data/local')
    os.makedirs('./data/local', exist_ok=True)

    if os.path.exists('./data/paras'):
        shutil.rmtree('./data/paras')
    os.makedirs('./data/paras', exist_ok=True)

    # Create save directory if it doesn't exist
    if not os.path.exists('./save'):
        os.makedirs('./save', exist_ok=True)

    # build network
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()

    # Use dataset-specific dict_users file, or use the newly generated one
    dict_users_file = f'../commonComponent/dict_users_{args.dataset}.pkl'
    if os.path.exists(dict_users_file):
        try:
            with open(dict_users_file, 'rb') as f:
                dict_users = pickle.load(f)
            print(f"Loaded dict_users from {dict_users_file}")
        except Exception as e:
            print(f"Warning: Failed to load {dict_users_file}, using newly generated dict_users: {e}")
    else:
        # Save the newly generated dict_users for future use
        try:
            with open(dict_users_file, 'wb') as f:
                pickle.dump(dict_users, f)
            print(f"Saved dict_users to {dict_users_file}")
        except Exception as e:
            print(f"Warning: Failed to save dict_users to {dict_users_file}: {e}")
    
    ## Used to check whether it is a new task
    checkTaskID = ''

    # Initialize delay attack manager if delay attack is enabled
    delay_manager = None
    if 'delay' in args.attack_type:
        delay_manager = DelayAttackManager()

    iteration = 0
    while 1:
        taskRelInfo = {}
        # taskRelease info template {"taskID":"task1994","epoch":10,"status":"start","usersFrac":0.1}
        while 1:
            taskRelQue, taskRelQueStt = usefulTools.simpleQuery('taskRelease')
            if taskRelQueStt == 0:
                try:
                    # Try to parse JSON, handle potential formatting issues
                    taskRelQue_clean = taskRelQue.strip()
                    taskRelInfo = json.loads(taskRelQue_clean)
                    print('\n*************************************************************************************')
                    print('Latest task release status is %s!'%taskRelQue_clean)
                    print('*************************************************************************************\n')
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Raw query result: {repr(taskRelQue)}")
                    print(f"Query result length: {len(taskRelQue)}")
                    print(f"First 100 chars: {taskRelQue[:100]}")
                    print("Retrying query...")
                    time.sleep(2)
                    continue
        taskRelEpoch = int(taskRelInfo['epoch'])
        
        # task info template {"epoch":"0","status":training,"paras":"QmSaAhKsxELzzT1uTtnuBacgfRjhehkzz3ybwnrkhJDbZ2"}
        taskID = taskRelInfo['taskID']
        print('Current task is',taskID)
        taskInfo = {}
        while 1:
            taskInQue, taskInQueStt = usefulTools.simpleQuery(taskID)
            if taskInQueStt == 0:
                taskInfo = json.loads(taskInQue)
                print('Latest task info is %s!'%taskInQue.strip())
                print('*************************************************************************************\n')
                break
        if taskInfo['status'] == 'done' or checkTaskID == taskID:
            print('*** %s has been completed! ***\n'%taskID)
            time.sleep(5)
        else:
            print('\n******************************* Iteration #%d starting ********************************'%iteration+'\n')
            print('Iteration %d starting!'%iteration)
            print('\n*************************************************************************************\n')
            
            # 从任务信息中读取设备列表（由聚合节点发布）
            ## init the list of device name
            allDeviceName = []
            for i in range(args.num_users):
                allDeviceName.append("device"+("{:0>5d}".format(i)))
            
            # 从taskRelease信息中读取设备列表
            if 'deviceList' in taskRelInfo:
                # 设备列表以索引数组形式存储，转换为Python list
                idxs_users = list(taskRelInfo['deviceList'])  # 转换为list，避免numpy array的布尔值问题
                deviceSelected = [allDeviceName[idx] for idx in idxs_users]
                print('\n**************************** Selected devices from task (read from blockchain) *****************************')
                print('Device indices: ', idxs_users)
                print('Device IDs: ', deviceSelected)
                print('**********************************************************************************************\n')
            else:
                # 如果没有设备列表（向后兼容），使用随机选择
                print('Warning: No deviceList in taskRelease, falling back to random selection')
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False).tolist()  # 转换为list
                deviceSelected = [allDeviceName[idx] for idx in idxs_users]
                print('\n**************************** Selected devices for this iteration (random fallback) *****************************')
                print('Device indices: ', idxs_users)
                print('Device IDs: ', deviceSelected)
                print('**********************************************************************************************\n')
            
            # Select malicious nodes for this iteration (based on selected devices)
            malicious_nodes = select_malicious_nodes(
                idxs_users, 
                args.malicious_frac, 
                args.attack_type,
                seed=args.seed if hasattr(args, 'seed') else None
            )
            
            currentEpoch = int(taskInfo['epoch']) + 1
            loss_train = []
            acc_test_list = []  # 存储每个epoch的测试准确率
            while currentEpoch <= taskRelEpoch:
                ## query the task info of current epoch
                while 1:
                    taskInQueEpo, taskInQueEpoStt = usefulTools.simpleQuery(taskID)
                    if taskInQueEpoStt == 0:
                        taskInfoEpo = json.loads(taskInQueEpo)
                        if int(taskInfoEpo['epoch']) == (currentEpoch-1):
                            print('\n****************************** Latest status of %s ******************************'%taskID)
                            print('(In loop) Latest task info is \n %s!'%taskInQueEpo)
                            print('*************************************************************************************\n')
                            break
                        else:
                            print('\n*************************** %s has not been updated ***************************'%taskID)
                            print('(In loop) Latest task info is \n %s!'%taskInQueEpo)
                            print('*************************************************************************************\n')
                            time.sleep(10)
                ## download the paras file of aggregated model for training in current epoch 
                aggBasModFil = './data/paras/aggModel-iter-' + str(iteration) + '-epoch-' + str(currentEpoch-1) + '.pkl'
                while 1:
                    aggBasMod, aggBasModStt = usefulTools.ipfsGetFile(taskInfoEpo['paras'], aggBasModFil)
                    if aggBasModStt == 0:
                        print('\nThe paras file of aggregated model for epoch %d training has been downloaded!\n'%(int(taskInfoEpo['epoch'])+1))
                        break
                    else:
                        print('\nFailed to download the paras file of aggregated model for epoch %d training!\n'%(int(taskInfoEpo['epoch'])+1))

                # copy weights and load the base model paras
                w_glob = net_glob.state_dict()
                net_glob.load_state_dict(torch.load(aggBasModFil))
                
                # 计算上一个epoch的准确率（如果存在）
                # 对于第一个epoch（currentEpoch == 1），没有上一个epoch的聚合模型，跳过
                if currentEpoch > 1:
                    net_glob.eval()
                    acc_test, loss_test = test_img(net_glob, dataset_test, args)
                    acc_test_value = acc_test.cpu().numpy().tolist()
                    acc_test_list.append(acc_test_value)
                    print('Epoch %d test accuracy: %.2f%%' % (currentEpoch - 1, acc_test_value))

                # 使用在iteration开始时选择的设备列表（idxs_users和malicious_nodes已经在iteration开始时确定）
                # 在整个iteration的所有epoch中都使用相同的设备集合

                loss_locals = []
                for idx in idxs_users:
                    is_mal = is_malicious_node(idx, malicious_nodes)
                    is_lazy = is_lazy_node(idx, malicious_nodes, args.attack_type)
                    
                    # Lazy attack: skip training and upload
                    if is_lazy:
                        print(f'Lazy node {allDeviceName[idx]} skipped in epoch {currentEpoch}')
                        continue
                    
                    # Create label mapping for labelflip attack if enabled
                    label_mapping = None
                    if is_mal and 'labelflip' in args.attack_type:
                        # Use node ID and epoch as seed for reproducibility
                        labelflip_seed = (args.seed if hasattr(args, 'seed') else 1) * 1000 + idx * 100 + currentEpoch
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
                        delay_manager.save_delayed_model(idx, currentEpoch, w)
                        # Get delayed model (from delay_rounds rounds ago) if exists
                        w_delayed = delay_manager.get_delayed_model(idx, currentEpoch, args.delay_rounds)
                        if w_delayed is not None:
                            w = w_delayed
                            print(f'Delay attack: {allDeviceName[idx]} submitted delayed model from epoch {currentEpoch - args.delay_rounds}')
                    else:
                        # Normal training
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], label_mapping=label_mapping)
                        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    
                    # Apply noise attack if enabled
                    if is_mal and 'noise' in args.attack_type:
                        w = noise_attack(w, args.noise_scale)
                    
                    loss_locals.append(copy.deepcopy(loss))
                    devLocFile = './data/local/' + taskID + '-' + allDeviceName[idx] + '-epoch-' + str(currentEpoch) + '.pkl'
                    torch.save(w, devLocFile)
                    
                    # Upload to IPFS with timeout
                    upload_success = False
                    start_time = time.time()
                    localAdd = None
                    while not upload_success and (time.time() - start_time) < args.lazy_timeout:
                        localAdd, localAddStt = usefulTools.ipfsAddFile(devLocFile)
                        if localAddStt == 0:
                            print('%s has been added to the IPFS network!'%devLocFile)
                            print('And the hash value of this file is %s'%localAdd)
                            upload_success = True
                            break
                        else:
                            print('Failed to add %s to the IPFS network!'%devLocFile)
                            time.sleep(1)
                    
                    if not upload_success:
                        print(f'Timeout: Failed to upload {allDeviceName[idx]} model after {args.lazy_timeout}s')
                        continue
                    
                    # Release to blockchain with timeout
                    release_success = False
                    start_time = time.time()
                    while not release_success and (time.time() - start_time) < args.lazy_timeout:
                        try:
                            localRelease = subprocess.Popen(
                                args=['../commonComponent/interRun.sh local '+allDeviceName[idx]+' '+taskID+' '+str(currentEpoch)+' '+localAdd],
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'
                            )
                            localOuts, localErrs = localRelease.communicate(timeout=10)
                            if localRelease.poll() == 0:
                                print('*** Local model train in epoch ' + str(currentEpoch) + ' of ' + allDeviceName[idx] + ' has been uploaded! ***\n')
                                release_success = True
                                break
                            else:
                                print(localErrs.strip())
                                print('*** Failed to release Local model train in epoch ' + str(currentEpoch) + ' of ' + allDeviceName[idx] + '! ***\n')
                        except subprocess.TimeoutExpired:
                            localRelease.kill()
                            print(f'Timeout: Failed to release {allDeviceName[idx]} model')
                        
                        time.sleep(2)
                    
                    if not release_success:
                        print(f'Timeout: Failed to release {allDeviceName[idx]} model after {args.lazy_timeout}s')
                
                # Clean old delayed models
                if delay_manager is not None:
                    delay_manager.clear_old_models(currentEpoch, max_delay=args.delay_rounds)

                loss_avg = sum(loss_locals) / len(loss_locals) if len(loss_locals) > 0 else 0.0
                loss_train.append(loss_avg)
                
                currentEpoch += 1

            # 计算最后一个epoch的准确率
            # 等待最后一个epoch的聚合模型（优化等待逻辑）
            print(f"Waiting for final epoch {taskRelEpoch} aggregation model...")
            start_wait_time = time.time()
            max_wait_time = args.lazy_timeout  # 减少等待时间到1倍lazy_timeout
            final_model_loaded = False
            last_epoch_seen = -1
            last_status_seen = ''
            
            while (time.time() - start_wait_time) < max_wait_time:
                taskInQueFinal, taskInQueFinalStt = usefulTools.simpleQuery(taskID)
                if taskInQueFinalStt == 0:
                    taskInfoFinal = json.loads(taskInQueFinal)
                    current_epoch = int(taskInfoFinal.get('epoch', -1))
                    current_status = taskInfoFinal.get('status', 'unknown')
                    
                    # 检查是否已完成
                    if current_status == 'done':
                        # 如果已经通过epoch匹配获取了模型，直接退出
                        if final_model_loaded:
                            print(f"Status is 'done', final model already tested.")
                            break
                        
                        # 下载最后一个epoch的聚合模型
                        finalAggModelFile = './data/paras/aggModel-iter-' + str(iteration) + '-epoch-' + str(taskRelEpoch) + '-final.pkl'
                        download_start = time.time()
                        while (time.time() - download_start) < 30:  # 下载超时30秒（减少）
                            finalAggModel, finalAggModelStt = usefulTools.ipfsGetFile(taskInfoFinal['paras'], finalAggModelFile)
                            if finalAggModelStt == 0:
                                # 加载模型并计算最后一个epoch的准确率
                                try:
                                    net_glob.load_state_dict(torch.load(finalAggModelFile))
                                    net_glob.eval()
                                    acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
                                    acc_test_final_value = acc_test_final.cpu().numpy().tolist()
                                    acc_test_list.append(acc_test_final_value)
                                    print('Final epoch %d test accuracy: %.2f%%' % (taskRelEpoch, acc_test_final_value))
                                    final_model_loaded = True
                                    break
                                except Exception as e:
                                    print(f"Error loading final model: {e}")
                            else:
                                time.sleep(1)
                        
                        if final_model_loaded:
                            break
                        else:
                            print(f"Warning: Failed to download/load final model, continuing...")
                            break
                    # 检查epoch是否匹配（但status还不是done）
                    elif current_epoch == taskRelEpoch:
                        # epoch匹配，尝试下载并测试模型（即使status还不是done）
                        elapsed = time.time() - start_wait_time
                        if current_epoch != last_epoch_seen or current_status != last_status_seen:
                            print(f"Epoch {current_epoch} reached (status: {current_status}), downloading and testing model...")
                            last_epoch_seen = current_epoch
                            last_status_seen = current_status
                        
                        # 尝试下载并测试当前epoch的模型
                        if not final_model_loaded:
                            finalAggModelFile = './data/paras/aggModel-iter-' + str(iteration) + '-epoch-' + str(taskRelEpoch) + '-final.pkl'
                            finalAggModel, finalAggModelStt = usefulTools.ipfsGetFile(taskInfoFinal['paras'], finalAggModelFile)
                            if finalAggModelStt == 0:
                                # 加载模型并计算最后一个epoch的准确率
                                try:
                                    net_glob.load_state_dict(torch.load(finalAggModelFile))
                                    net_glob.eval()
                                    acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
                                    acc_test_final_value = acc_test_final.cpu().numpy().tolist()
                                    acc_test_list.append(acc_test_final_value)
                                    print('Final epoch %d test accuracy: %.2f%%' % (taskRelEpoch, acc_test_final_value))
                                    final_model_loaded = True
                                    # 如果status还不是done，继续等待status变成done（但已经获取了准确率）
                                    if current_status != 'done':
                                        print(f"Model tested, but status is still '{current_status}', waiting for 'done'...")
                                except Exception as e:
                                    print(f"Error loading final model: {e}")
                        
                        # 如果已经获取了模型，但status还不是done，继续等待
                        if final_model_loaded and current_status != 'done':
                            time.sleep(3)
                        elif not final_model_loaded:
                            time.sleep(3)  # 如果还没获取到模型，继续等待
                        else:
                            # 已经获取了模型且status是done，退出
                            break
                    else:
                        # epoch还没到，继续等待
                        elapsed = time.time() - start_wait_time
                        if current_epoch != last_epoch_seen or current_status != last_status_seen:
                            print(f"Waiting for final epoch... (current epoch: {current_epoch}, status: {current_status}, elapsed: {elapsed:.1f}s)")
                            last_epoch_seen = current_epoch
                            last_status_seen = current_status
                        time.sleep(3)  # 减少sleep时间从5秒到3秒
                else:
                    # 查询失败，短暂等待后重试
                    elapsed = time.time() - start_wait_time
                    if elapsed % 10 < 3:  # 每10秒打印一次
                        print(f"Query failed, retrying... (elapsed: {elapsed:.1f}s)")
                    time.sleep(2)  # 减少sleep时间从5秒到2秒
            
            if not final_model_loaded:
                print(f"Warning: Timeout waiting for final epoch {taskRelEpoch} model after {max_wait_time:.1f}s, continuing without final accuracy...")

            checkTaskID = taskID
            
            # 保存准确率到CSV文件（已禁用，不需要local-开头的文件）
            # if len(acc_test_list) > 0:
            #     # 确保data目录存在（项目根目录下的data文件夹）
            #     os.makedirs('../data', exist_ok=True)
            #     
            #     # 生成文件名：格式与示例文件一致 dag-{attack}-{mal_frac}-{timestamp}_basic.csv
            #     attack_str = args.attack_type if args.attack_type != 'none' else 'none'
            #     mal_frac_str = f"{args.malicious_frac:.1f}" if args.malicious_frac > 0 else "0.0"
            #     timestamp = int(time.time())
            #     csv_filename = '../data/local-{}-{}-{}_basic.csv'.format(
            #         attack_str, mal_frac_str, timestamp
            #     )
            #     
            #     # 按照示例CSV格式保存：第一行是shard名称，后面每行是准确率值
            #     with open(csv_filename, 'w') as f:
            #         f.write('shard_1\n')  # 第一行是shard名称
            #         for acc in acc_test_list:
            #             f.write(str(acc) + '\n')
            #     
            #     print('\nAccuracy results saved to: %s' % csv_filename)
            #     print('Total epochs: %d' % len(acc_test_list))
            
            print('\n*********************************** Iteration #%d ***********************************'%iteration+'\n')
            print('Current iteration %d has been completed!'%iteration)
            print('\n*************************************************************************************\n')
            iteration += 1

