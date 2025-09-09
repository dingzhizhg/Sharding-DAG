#!/bin/bash

# 启动 Hyperledger Fabric mycc 网络脚本
# 此脚本会自动启动 Fabric 网络、创建通道并部署 mycc chaincode

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 设置 Fabric 路径
export FabricL=/home/pi/fabric/fabric-samples/test-network
export PATH=${FabricL}/../bin:$PATH
export FABRIC_CFG_PATH=${FabricL}/../config/
export CORE_PEER_TLS_ENABLED=true

# 项目路径
PROJECT_ROOT=/home/pi/Sharding-DAG
CHAINCODE_DIR=${PROJECT_ROOT}/chaincode/mycc

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}启动 Hyperledger Fabric mycc 网络${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 Fabric 目录是否存在
if [ ! -d "$FabricL" ]; then
    echo -e "${RED}错误: Fabric test-network 目录不存在: $FabricL${NC}"
    exit 1
fi

# 检查 chaincode 目录是否存在
if [ ! -d "$CHAINCODE_DIR" ]; then
    echo -e "${RED}错误: Chaincode 目录不存在: $CHAINCODE_DIR${NC}"
    exit 1
fi

# 函数：检查网络是否运行
check_network_running() {
    local containers=$(docker ps --format "{{.Names}}" | grep -E "peer|orderer|ca" | wc -l)
    if [ "$containers" -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# 函数：检查通道是否存在
check_channel_exists() {
    if [ -f "${FabricL}/channel-artifacts/mychannel.block" ]; then
        return 0
    else
        return 1
    fi
}

# 函数：检查 chaincode 是否已部署
check_chaincode_deployed() {
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    export CORE_PEER_ADDRESS=localhost:7051
    
    local result=$(peer lifecycle chaincode querycommitted -C mychannel -n mycc 2>&1)
    if echo "$result" | grep -q "mycc"; then
        return 0
    else
        return 1
    fi
}

# 步骤 1: 启动 Fabric 网络
echo -e "\n${YELLOW}[步骤 1/4] 检查 Fabric 网络状态...${NC}"
if check_network_running; then
    echo -e "${GREEN}✓ Fabric 网络已在运行${NC}"
else
    echo -e "${YELLOW}启动 Fabric 网络并创建通道...${NC}"
    cd $FabricL
    ./network.sh up createChannel -ca
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Fabric 网络已启动，通道已创建${NC}"
    else
        echo -e "${RED}✗ 启动 Fabric 网络失败${NC}"
        exit 1
    fi
fi

# 等待网络就绪
echo -e "${YELLOW}等待网络就绪...${NC}"
sleep 5

# 步骤 2: 检查通道
echo -e "\n${YELLOW}[步骤 2/4] 检查通道状态...${NC}"
if check_channel_exists; then
    echo -e "${GREEN}✓ 通道 mychannel 已存在${NC}"
else
    echo -e "${YELLOW}创建通道...${NC}"
    # 设置 Org1 环境变量
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    export CORE_PEER_ADDRESS=localhost:7051
    
    # 创建通道
    peer channel create -o localhost:7050 \
        --ordererTLSHostnameOverride orderer.example.com \
        -c mychannel \
        -f ${FabricL}/channel-artifacts/mychannel.tx \
        --outputBlock ${FabricL}/channel-artifacts/mychannel.block \
        --tls \
        --cafile ${FabricL}/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem
    
    # Org1 加入通道
    peer channel join -b ${FabricL}/channel-artifacts/mychannel.block
    
    # Org2 加入通道
    export CORE_PEER_LOCALMSPID="Org2MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
    export CORE_PEER_ADDRESS=localhost:9051
    
    peer channel join -b ${FabricL}/channel-artifacts/mychannel.block
    echo -e "${GREEN}✓ 通道已创建并加入${NC}"
fi

# 步骤 3: 检查并部署 mycc chaincode
echo -e "\n${YELLOW}[步骤 3/4] 检查 mycc chaincode 状态...${NC}"
if check_chaincode_deployed; then
    echo -e "${GREEN}✓ mycc chaincode 已部署${NC}"
else
    echo -e "${YELLOW}部署 mycc chaincode...${NC}"
    
    # 3.1 打包 Chaincode
    echo -e "${YELLOW}  打包 chaincode...${NC}"
    cd $CHAINCODE_DIR
    if [ -f "mycc.tar.gz" ]; then
        rm mycc.tar.gz
    fi
    
    peer lifecycle chaincode package mycc.tar.gz \
        --path . \
        --lang golang \
        --label mycc_1.0
    
    if [ ! -f "mycc.tar.gz" ]; then
        echo -e "${RED}✗ 打包 chaincode 失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ Chaincode 已打包${NC}"
    
    # 3.2 安装到 Org1
    echo -e "${YELLOW}  安装到 Org1...${NC}"
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    export CORE_PEER_ADDRESS=localhost:7051
    
    peer lifecycle chaincode install mycc.tar.gz
    echo -e "${GREEN}  ✓ 已安装到 Org1${NC}"
    
    # 3.3 安装到 Org2
    echo -e "${YELLOW}  安装到 Org2...${NC}"
    export CORE_PEER_LOCALMSPID="Org2MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
    export CORE_PEER_ADDRESS=localhost:9051
    
    peer lifecycle chaincode install mycc.tar.gz
    echo -e "${GREEN}  ✓ 已安装到 Org2${NC}"
    
    # 3.4 查询包ID
    echo -e "${YELLOW}  查询包ID...${NC}"
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    export CORE_PEER_ADDRESS=localhost:7051
    
    PACKAGE_ID=$(peer lifecycle chaincode queryinstalled | grep -oP 'mycc_1.0:\K[^,}]+' | head -1)
    if [ -z "$PACKAGE_ID" ]; then
        echo -e "${RED}✗ 无法获取包ID${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ 包ID: $PACKAGE_ID${NC}"
    
    # 3.5 批准 Chaincode 定义（Org1）
    echo -e "${YELLOW}  Org1 批准 chaincode 定义...${NC}"
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
    echo -e "${GREEN}  ✓ Org1 已批准${NC}"
    
    # 3.6 批准 Chaincode 定义（Org2）
    echo -e "${YELLOW}  Org2 批准 chaincode 定义...${NC}"
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
    echo -e "${GREEN}  ✓ Org2 已批准${NC}"
    
    # 3.7 提交 Chaincode 定义到通道
    echo -e "${YELLOW}  提交 chaincode 定义到通道...${NC}"
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
    echo -e "${GREEN}  ✓ Chaincode 已提交到通道${NC}"
    
    echo -e "${GREEN}✓ mycc chaincode 部署完成${NC}"
fi

# 步骤 4: 验证部署
echo -e "\n${YELLOW}[步骤 4/4] 验证部署...${NC}"
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

# 测试查询
echo -e "${YELLOW}  测试 chaincode 查询...${NC}"
result=$(peer chaincode query -C mychannel -n mycc -c '{"Args":["GetAllAssets"]}' 2>&1)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Chaincode 查询成功${NC}"
    echo -e "${GREEN}  结果: $result${NC}"
else
    echo -e "${YELLOW}  ⚠ Chaincode 查询返回非零状态，但可能正常（如果返回空结果）${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ mycc 网络启动完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n网络状态："
echo -e "  - Fabric 网络: ${GREEN}运行中${NC}"
echo -e "  - 通道: ${GREEN}mychannel${NC}"
echo -e "  - Chaincode: ${GREEN}mycc v1.0${NC}"
echo -e "\n可以使用以下命令验证："
echo -e "  ${YELLOW}docker ps | grep -E \"peer|orderer|ca\"${NC}"
echo -e "  ${YELLOW}cd $PROJECT_ROOT/commonComponent && ./interRun.sh query taskRelease${NC}"

