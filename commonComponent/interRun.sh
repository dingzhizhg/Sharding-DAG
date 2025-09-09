#!/bin/bash

# Print the usage message
function printHelp() {
  echo "Usage: "
  echo "  allInOne.sh <Mode> [Variables]"
  echo "    <Mode>"
  echo "      - 'release'"
  echo "      - 'local'"
  echo "      - 'global'"
  echo "      - 'query'"
  echo "    [Variables]"
    echo "    for 'release' taskID epoch status usersFrac [deviceList]"
  echo "    for 'local' modelFileTrainedLocally deviceID taskID currentEpoch"
  echo "    for 'global' globalModelFileAggregated taskID currentEpoch status"
  echo "    for 'query' queryInfo"
  echo " Examples:"
  echo "  interRun.sh release task1234 10 start 0.1"
  echo "  interRun.sh local device00010 task1234 2 fileHash"
  echo "  interRun.sh aggregated task1234 2 training fileHash"
  echo "  interRun.sh query task1234"
}

# 跟随 Fabric 2.5 test-network 的目录结构
export FabricL=/home/pi/fabric/fabric-samples/test-network
export PATH=${FabricL}/../bin:$PATH
export FABRIC_CFG_PATH=${FabricL}/../config/
# Turn on the tls
export CORE_PEER_TLS_ENABLED=true

# Fabric network host address
# For Raspberry Pi connecting to WSL: set FABRIC_HOST to WSL IP (e.g., 172.28.223.132)
# For local connections: use localhost (default)
FABRIC_HOST=${FABRIC_HOST:-localhost}

# Set the environment variables
setEnvironments() {
  ORG=$1
  if [ $ORG -eq 1 ]; then
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    export CORE_PEER_ADDRESS=${FABRIC_HOST}:7051                                                                       
  elif [ $ORG -eq 2 ]; then                                                                                       
    export CORE_PEER_LOCALMSPID="Org2MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt 
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
    export CORE_PEER_ADDRESS=${FABRIC_HOST}:7051
  else  
    echo "================== ERROR !!! ORG Unknown =================="                                            
  fi
}

# Release the FL task from fabric
function taskRelease() {
  ORG=$1
  setEnvironments $ORG
  # If deviceList (6th parameter) is provided, include it in JSON
  # Use "$6" to preserve the entire JSON array string (with quotes from shlex.quote)
  if [ -n "$6" ]; then
    # $6 is already quoted by shlex.quote, so we need to remove the outer quotes
    # shlex.quote adds single quotes, so we remove them
    deviceList="${6#\'}"
    deviceList="${deviceList%\'}"
    # Now deviceList should be the raw JSON array like [5, 1, 55, 29]
    taskJson="{\\\"taskID\\\":\\\"$2\\\",\\\"epoch\\\":$3,\\\"status\\\":\\\"$4\\\",\\\"usersFrac\\\":$5,\\\"deviceList\\\":${deviceList}}"
  else
    taskJson="{\\\"taskID\\\":\\\"$2\\\",\\\"epoch\\\":$3,\\\"status\\\":\\\"$4\\\",\\\"usersFrac\\\":$5}"
  fi
  invoke="peer chaincode invoke \
    -o ${FABRIC_HOST}:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem \
    --peerAddresses ${FABRIC_HOST}:7051 \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses ${FABRIC_HOST}:9051 \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
    -C mychannel -n mycc \
    -c '{\"Args\":[\"set\",\"taskRelease\",\"${taskJson}\"]}'"
  eval ${invoke}
}

# Publish the global model file aggregated in current epoch
function aggModelPub() {
  ORG=$1
  setEnvironments $ORG
  invoke="peer chaincode invoke \
    -o ${FABRIC_HOST}:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem \
    --peerAddresses ${FABRIC_HOST}:7051 \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses ${FABRIC_HOST}:9051 \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
    -C mychannel -n mycc \
    -c '{\"Args\":[\"set\",\"${2}\",\"{\\\"epoch\\\":$3,\\\"status\\\":\\\"$4\\\",\\\"paras\\\":\\\"$5\\\"}\"]}'"
  eval ${invoke}
}

# Publish the local model file trained in current epoch
function localModelPub() {
  ORG=$1
  setEnvironments $ORG
  invoke="peer chaincode invoke \
    -o ${FABRIC_HOST}:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem \
    --peerAddresses ${FABRIC_HOST}:7051 \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses ${FABRIC_HOST}:9051 \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
    -C mychannel -n mycc \
    -c '{\"Args\":[\"set\",\"${2}\",\"{\\\"taskID\\\":\\\"$3\\\",\\\"epoch\\\":$4,\\\"paras\\\":\\\"${5}\\\"}\"]}'"
  eval ${invoke}
}

# Query the info from the chaincode
function devQuery(){
  ORG=$1
  setEnvironments $ORG
  query="peer chaincode query \
    -C mychannel -n mycc \
    -c '{\"Args\":[\"get\",\"${2}\"]}'"
  eval ${query}
}

# Parse commandline args

## Parse mode

if [[ $# -lt 1 ]] ; then
  printHelp
  exit 0
else
  MODE=$1
  shift
fi

if [ "${MODE}" == "release" ]; then
  taskRelease 1 $1 $2 $3 $4 "$5"
  sleep 2
  devQuery 1 'taskRelease'
elif [ "${MODE}" == "local" ]; then
  localModelPub 1 $1 $2 $3 $4
elif [ "${MODE}" == "aggregated" ]; then
  aggModelPub 1 $1 $2 $3 $4
  sleep 2
  devQuery 1 $1
elif [ "${MODE}" == "query" ]; then
  devQuery 1 $1
else
  printHelp
  exit 1
fi
