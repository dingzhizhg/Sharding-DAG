# 手动启动 Fabric 网络和部署 mycc Chaincode

## 环境变量设置

```bash
# 设置Fabric路径
export FabricL=/home/pi/fabric/fabric-samples/test-network
export PATH=${FabricL}/../bin:$PATH
export FABRIC_CFG_PATH=${FabricL}/../config/
export CORE_PEER_TLS_ENABLED=true
```

## 1. 启动 Fabric 网络（如果未运行）

### 启动网络基础设施（使用Docker Compose或手动启动）

```bash
cd /home/pi/fabric/fabric-samples/test-network

# 启动CA、Orderer和Peer节点
docker-compose -f docker/docker-compose-ca.yaml up -d
docker-compose -f docker/docker-compose-test-net.yaml up -d
```

或者检查网络是否已运行：
```bash
docker ps | grep -E "peer|orderer|ca"
```

## 2. 创建通道（如果不存在）

### 设置Org1环境变量
```bash
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051
```

### 创建通道
```bash
peer channel create -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  -c mychannel \
  -f ${FabricL}/channel-artifacts/mychannel.tx \
  --outputBlock ${FabricL}/channel-artifacts/mychannel.block \
  --tls \
  --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem
```

### 加入通道
```bash
# Org1加入通道
peer channel join -b ${FabricL}/channel-artifacts/mychannel.block

# Org2加入通道
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=localhost:9051

peer channel join -b ${FabricL}/channel-artifacts/mychannel.block
```

## 3. 部署 mycc Chaincode

### 3.1 打包 Chaincode
```bash
cd /home/pi/Sharding-DAG/chaincode/mycc

peer lifecycle chaincode package mycc.tar.gz \
  --path . \
  --lang golang \
  --label mycc_1.0
```

### 3.2 安装到 Org1 的 Peer
```bash
# 设置Org1环境
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

# 安装chaincode
peer lifecycle chaincode install mycc.tar.gz
```

### 3.3 安装到 Org2 的 Peer
```bash
# 设置Org2环境
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=localhost:9051

# 安装chaincode
peer lifecycle chaincode install mycc.tar.gz
```

### 3.4 查询包ID
```bash
# 使用Org1查询
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

peer lifecycle chaincode queryinstalled
```

记录返回的包ID，例如：`mycc_1.0:65fa9486b7b50011160074c293371fa5a2a4e4733b708b297173e96a86912e04`

### 3.5 批准 Chaincode 定义（Org1）
```bash
PACKAGE_ID="mycc_1.0:65fa9486b7b50011160074c293371fa5a2a4e4733b708b297173e96a86912e04"

peer lifecycle chaincode approveformyorg \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --channelID mychannel \
  --name mycc \
  --version 1.0 \
  --package-id $PACKAGE_ID \
  --sequence 1 \
  --tls \
  --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem
```

### 3.6 批准 Chaincode 定义（Org2）
```bash
# 设置Org2环境
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=localhost:9051

peer lifecycle chaincode approveformyorg \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --channelID mychannel \
  --name mycc \
  --version 1.0 \
  --package-id $PACKAGE_ID \
  --sequence 1 \
  --tls \
  --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem
```

### 3.7 提交 Chaincode 定义到通道
```bash
# 使用Org1提交
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

peer lifecycle chaincode commit \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --channelID mychannel \
  --name mycc \
  --version 1.0 \
  --sequence 1 \
  --peerAddresses localhost:7051 \
  --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
  --peerAddresses localhost:9051 \
  --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
  --tls \
  --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem
```

## 4. 验证 Chaincode

### 查询所有资产
```bash
peer chaincode query \
  -C mychannel \
  -n mycc \
  -c '{"Args":["GetAllAssets"]}'
```

### 设置一个值
```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls \
  --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem \
  --peerAddresses localhost:7051 \
  --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
  --peerAddresses localhost:9051 \
  --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
  -C mychannel \
  -n mycc \
  -c '{"Args":["Set","testKey","testValue"]}'
```

### 查询值
```bash
peer chaincode query \
  -C mychannel \
  -n mycc \
  -c '{"Args":["Get","testKey"]}'
```

## 当前状态

✅ Fabric 网络已运行
✅ 通道 mychannel 已创建并加入
✅ mycc chaincode 已安装并提交到通道
✅ Chaincode 版本: 1.0, Sequence: 1

## 注意事项

1. 确保所有Docker容器都在运行：`docker ps`
2. 确保环境变量正确设置
3. 如果网络已运行，可以跳过启动步骤
4. 如果chaincode已安装，可以跳过安装步骤，直接进行批准和提交


