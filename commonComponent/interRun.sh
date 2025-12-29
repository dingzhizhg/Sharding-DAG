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

# Check if using IP address and warn about /etc/hosts requirement
if [[ ${FABRIC_HOST} =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && [ "${FABRIC_HOST}" != "127.0.0.1" ]; then
  # IP address detected - check if domain names resolve to this IP
  PEER1_RESOLVED=$(getent hosts peer0.org1.example.com 2>/dev/null | awk '{print $1}' | head -1)
  if [ -z "${PEER1_RESOLVED}" ] || [ "${PEER1_RESOLVED}" != "${FABRIC_HOST}" ]; then
    echo "===================================================================" >&2
    echo "ERROR: Using IP address ${FABRIC_HOST}, but domain name not configured!" >&2
    echo "===================================================================" >&2
    echo "" >&2
    echo "peer0.org1.example.com cannot be resolved or resolves to wrong IP." >&2
    echo "" >&2
    echo "REQUIRED ACTION: Add domain mapping to /etc/hosts file:" >&2
    echo "" >&2
    echo "  Run this command (requires sudo):" >&2
    echo "    sudo bash -c 'echo \"${FABRIC_HOST} peer0.org1.example.com peer0.org2.example.com orderer.example.com\" >> /etc/hosts'" >&2
    echo "" >&2
    echo "  Or manually edit /etc/hosts and add:" >&2
    echo "    ${FABRIC_HOST} peer0.org1.example.com peer0.org2.example.com orderer.example.com" >&2
    echo "" >&2
    echo "This is REQUIRED for TLS certificate validation." >&2
    echo "Fabric certificates are issued for domain names, not IP addresses." >&2
    echo "===================================================================" >&2
    exit 1
  fi
fi

# Set the environment variables
setEnvironments() {
  ORG=$1
  # Check if FABRIC_HOST is an IP address - if so, use domain name for peer addresses
  # to avoid TLS certificate validation issues (certificates are issued for domain names)
  if [[ ${FABRIC_HOST} =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # IP address detected - use domain names for TLS verification
    PEER1_DOMAIN="peer0.org1.example.com"
    PEER2_DOMAIN="peer0.org2.example.com"
    ORDERER_DOMAIN="orderer.example.com"
    PEER1_ADDRESS=${PEER1_DOMAIN}:7051
    PEER2_ADDRESS=${PEER2_DOMAIN}:9051
    ORDERER_ADDRESS=${ORDERER_DOMAIN}:7050
    if [ $ORG -eq 1 ]; then
      PEER_ADDRESS=${PEER1_ADDRESS}
    else
      PEER_ADDRESS=${PEER2_ADDRESS}
    fi
  else
    # Domain name or localhost - use as is
    PEER1_ADDRESS=${FABRIC_HOST}:7051
    PEER2_ADDRESS=${FABRIC_HOST}:9051
    ORDERER_ADDRESS=${FABRIC_HOST}:7050
    PEER_ADDRESS=${FABRIC_HOST}:7051
  fi
  
  # Export peer addresses for use in invoke commands
  export PEER1_ADDRESS
  export PEER2_ADDRESS
  export ORDERER_ADDRESS
  
  if [ $ORG -eq 1 ]; then
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    export CORE_PEER_ADDRESS=${PEER_ADDRESS}
  elif [ $ORG -eq 2 ]; then                                                                                       
    export CORE_PEER_LOCALMSPID="Org2MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt 
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
    export CORE_PEER_ADDRESS=${PEER_ADDRESS}
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
    -o ${ORDERER_ADDRESS} \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem \
    --peerAddresses ${PEER1_ADDRESS} \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses ${PEER2_ADDRESS} \
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
    -o ${ORDERER_ADDRESS} \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem \
    --peerAddresses ${PEER1_ADDRESS} \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses ${PEER2_ADDRESS} \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
    -C mychannel -n mycc \
    -c '{\"Args\":[\"set\",\"${2}\",\"{\\\"epoch\\\":$3,\\\"status\\\":\\\"$4\\\",\\\"paras\\\":\\\"$5\\\"}\"]}'"
  eval ${invoke}
}

# Publish the local model file trained in current epoch
# Key format: deviceID-taskID-epoch (e.g., device00010-task1234-1)
# This ensures uniqueness across different tasks and epochs
function localModelPub() {
  ORG=$1
  setEnvironments $ORG
  # Construct key as deviceID-taskID-epoch for uniqueness
  deviceKey="${2}-${3}-${4}"
  invoke="peer chaincode invoke \
    -o ${ORDERER_ADDRESS} \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem \
    --peerAddresses ${PEER1_ADDRESS} \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses ${PEER2_ADDRESS} \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
    -C mychannel -n mycc \
    -c '{\"Args\":[\"set\",\"${deviceKey}\",\"{\\\"taskID\\\":\\\"$3\\\",\\\"epoch\\\":$4,\\\"paras\\\":\\\"${5}\\\"}\"]}'"
  eval ${invoke}
}

# Publish the selected devices list for current epoch
# Key format: taskID-epochN-devices (e.g., task1234-epoch1-devices)
# This allows server to communicate which devices are selected for each epoch
function epochDevicesPub() {
  ORG=$1
  setEnvironments $ORG
  # Key format: taskID-epochN-devices
  devicesKey="${2}-epoch${3}-devices"
  # $4 is the JSON array of device names (already escaped)
  invoke="peer chaincode invoke \
    -o ${ORDERER_ADDRESS} \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem \
    --peerAddresses ${PEER1_ADDRESS} \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses ${PEER2_ADDRESS} \
    --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
    -C mychannel -n mycc \
    -c '{\"Args\":[\"set\",\"${devicesKey}\",\"{\\\"taskID\\\":\\\"$2\\\",\\\"epoch\\\":$3,\\\"devices\\\":$4}\"]}'"
  eval ${invoke}
}

# Query the info from the chaincode
function devQuery(){
  ORG=$1
  setEnvironments $ORG
  # When using IP address, setEnvironments sets CORE_PEER_ADDRESS to domain name
  # This allows TLS certificate verification to pass (certificates are for domain names)
  # The domain name should resolve to the IP via /etc/hosts or DNS
  # Use explicit --peerAddresses and --tlsRootCertFiles to ensure domain name is used for TLS verification
  # IMPORTANT: Use domain name in --peerAddresses (not IP) for TLS certificate validation
  # CORE_PEER_ADDRESS is already set to domain name by setEnvironments when FABRIC_HOST is an IP
  query="peer chaincode query \
    --peerAddresses ${CORE_PEER_ADDRESS} \
    --tlsRootCertFiles ${CORE_PEER_TLS_ROOTCERT_FILE} \
    --tls \
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
elif [ "${MODE}" == "epochdevices" ]; then
  epochDevicesPub 1 $1 $2 "$3"
elif [ "${MODE}" == "query" ]; then
  devQuery 1 $1
else
  printHelp
  exit 1
fi
