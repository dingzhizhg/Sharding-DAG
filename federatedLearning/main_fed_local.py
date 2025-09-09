# -*- coding: utf-8 -*-

import copy
import numpy as np
import math
import time
import shutil
from torchvision import datasets, transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from json import dumps
import os
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
from models.Attacks import DelayAttackManager, noise_attack, crejljsdllllate_labelflip_mapping
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

    with open('../commonComponent/dict_users.pkl', 'rb') as f:
        dict_users = pickle.load(f)
    
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
                taskRelInfo = json.loads(taskRelQue)
                print('\n*************************************************************************************')
                print('Latest task release status is %s!'%taskRelQue.strip())
                break
        taskRelEpoch = int(taskRelInfo['epoch'])
        # 确保训练轮数为50
        taskRelEpoch = 50
        
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

                # Device name list
                # with open('../commonComponent/selectedDeviceIdxs.txt', 'rb') as f:
                #     idxs_users = pickle.load(f)

                idxs_users = [ 5, 56, 76, 78, 68, 25, 47, 15, 61, 55]

                ## init the list of device name
                allDeviceName = []
                for i in range(args.num_users):
                    allDeviceName.append("device"+("{:0>5d}".format(i)))

                # Select malicious nodes for this round
                malicious_nodes = select_malicious_nodes(
                    idxs_users, 
                    args.malicious_frac, 
                    args.attack_type,
                    seed=args.seed if hasattr(args, 'seed') else None
                )

                # print('\n**************************** All devices *****************************')
                # print(allDeviceName)
                # print('*************************************************************************************\n')

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
            # 等待最后一个epoch的聚合模型
            while 1:
                taskInQueFinal, taskInQueFinalStt = usefulTools.simpleQuery(taskID)
                if taskInQueFinalStt == 0:
                    taskInfoFinal = json.loads(taskInQueFinal)
                    if int(taskInfoFinal['epoch']) == taskRelEpoch:
                        # 下载最后一个epoch的聚合模型
                        finalAggModelFile = './data/paras/aggModel-iter-' + str(iteration) + '-epoch-' + str(taskRelEpoch) + '-final.pkl'
                        while 1:
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
                        print('Final epoch %d test accuracy: %.2f%%' % (taskRelEpoch, acc_test_final_value))
                        break
                    else:
                        time.sleep(5)

            checkTaskID = taskID
            
            # 保存准确率到CSV文件
            if len(acc_test_list) > 0:
                # 确保data目录存在（项目根目录下的data文件夹）
                os.makedirs('../data', exist_ok=True)
                
                # 生成文件名：格式与示例文件一致 dag-{attack}-{mal_frac}-{timestamp}_basic.csv
                attack_str = args.attack_type if args.attack_type != 'none' else 'none'
                mal_frac_str = f"{args.malicious_frac:.1f}" if args.malicious_frac > 0 else "0.0"
                timestamp = int(time.time())
                csv_filename = '../data/local-{}-{}-{}_basic.csv'.format(
                    attack_str, mal_frac_str, timestamp
                )
                
                # 按照示例CSV格式保存：第一行是shard名称，后面每行是准确率值
                with open(csv_filename, 'w') as f:
                    f.write('shard_1\n')  # 第一行是shard名称
                    for acc in acc_test_list:
                        f.write(str(acc) + '\n')
                
                print('\nAccuracy results saved to: %s' % csv_filename)
                print('Total epochs: %d' % len(acc_test_list))
            
            print('\n*********************************** Iteration #%d ***********************************'%iteration+'\n')
            print('Current iteration %d has been completed!'%iteration)
            print('\n*************************************************************************************\n')
            iteration += 1

