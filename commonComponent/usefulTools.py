# -*- coding: utf-8 -*-

import random
import sys
import json
import time
import subprocess

# two ipfs functions, understand
def ipfsGetFile(hashValue, fileName, timeout=60):
    """
    Use hashValue to download the file from IPFS network.
    
    Args:
        hashValue: IPFS hash of the file
        fileName: Local file path to save the downloaded file
        timeout: Download timeout in seconds (default 60s for distributed networks)
    """
    ipfsGet = subprocess.Popen(args=['ipfs get ' + hashValue + ' -o ' + fileName], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = ipfsGet.communicate(timeout=timeout)
    if ipfsGet.poll() == 0:
        return outs.strip(), ipfsGet.poll()
    else:
        return errs.strip(), ipfsGet.poll()

def ipfsAddFile(fileName, timeout=60):
    """
    Upload the file to IPFS network and return the exclusive fileHash value.
    
    Args:
        fileName: Local file path to upload
        timeout: Upload timeout in seconds (default 60s for distributed networks)
    """
    ipfsAdd = subprocess.Popen(args=['ipfs add ' + fileName + ' | tr \' \' \'\\n\' | grep Qm'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = ipfsAdd.communicate(timeout=timeout)
    if ipfsAdd.poll() == 0:
        return outs.strip(), ipfsAdd.poll()
    else:
        return errs.strip(), ipfsAdd.poll()

def queryLocal(lock, taskID, deviceID, currentEpoch, flagSet, localFileName, timeout=15):
    """
    Query and download the paras file of local model trained by the device.
    
    参数:
        lock: 线程锁
        taskID: 任务ID
        deviceID: 设备ID
        currentEpoch: 当前轮次
        flagSet: 标志集合
        localFileName: 本地文件名
        timeout: 超时时间（秒），默认15秒
    """
    try:
        # Use the correct key format: deviceID-taskID-epoch (matches the format used in localModelPub)
        deviceKey = f"{deviceID}-{taskID}-{currentEpoch}"
        localQuery = subprocess.Popen(args=['../commonComponent/interRun.sh query '+deviceKey], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        outs, errs = localQuery.communicate(timeout=timeout)
        if localQuery.poll() == 0:
            localDetail = json.loads(outs.strip())
            if localDetail['epoch'] == currentEpoch and localDetail['taskID'] == taskID:
                print("The query result of the " + deviceID + " is ", outs.strip())
                while 1:
                    # localFileName = './clientS/paras/' + taskID + '-' + deviceID + '-epoch-' + str(currentEpoch) + '.pkl'
                    outs, stt = ipfsGetFile(localDetail['paras'], localFileName)
                    if stt == 0:
                        break
                    # else:
                    #     print(outs.strip())
                lock.acquire()
                t1 = flagSet
                t1.add(deviceID)
                flagSet = t1
                lock.release()
            # else:
            #     print('*** This device %s has not updated its model! ***'%(deviceID))
        # else:
        #     print("Failed to query this device!", errs)
    except subprocess.TimeoutExpired:
        localQuery.kill()
        print(f"Timeout: Failed to query {deviceID} within {timeout}s")
    except Exception as e:
        print(f"Error querying {deviceID}: {e}")

def queryShell():
    """
    Use the shell envs to query info from fabric network.
    """
    # shell envs
    shellEnv1 = "export PATH=${PWD}/../bin:$PATH"
    shellEnv2 = "export FABRIC_CFG_PATH=$PWD/../config/"
    shellEnv3 = "export CORE_PEER_TLS_ENABLED=true"
    shellEnv4 = "export CORE_PEER_LOCALMSPID=\"Org1MSP\""
    shellEnv5 = "export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
    shellEnv6 = "export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
    shellEnv7 = "export CORE_PEER_ADDRESS=localhost:7051"
    oneKeyEnv = shellEnv1 + " && " + shellEnv2 + " && " + shellEnv3 + " && " + shellEnv4 + " && " + shellEnv5 + " && " + shellEnv6 + " && " + shellEnv7

def simpleQuery(key):
    """
    Use the only key to query info from fabric network.
    """
    infoQuery = subprocess.Popen(args=['../commonComponent/interRun.sh query '+key], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = infoQuery.communicate(timeout=15)
    if infoQuery.poll() == 0:
        return outs.strip(), infoQuery.poll()
    else:
        print("*** Failed to query the info of " + str(key) + "! ***" + errs.strip())
        return errs.strip(), infoQuery.poll()

def simpleQuerySilent(key):
    """
    Use the only key to query info from fabric network (silent mode - no error printing).
    This is useful when querying devices that may not exist yet, to avoid log spam.
    """
    infoQuery = subprocess.Popen(args=['../commonComponent/interRun.sh query '+key], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = infoQuery.communicate(timeout=15)
    if infoQuery.poll() == 0:
        return outs.strip(), infoQuery.poll()
    else:
        # Silently return error status without printing
        return errs.strip(), infoQuery.poll()


if __name__ == '__main__':
    taskInfo = {}
    while 1:
        outs, stt = simpleQuery("task109210")
        if stt == 0:
            print(outs)
            taskInfo = json.loads(outs)
            print('Latest task info is %s!\n'%outs)
            break
    print(type(taskInfo))
    print(taskInfo)
