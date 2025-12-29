# FedAvg Sharding Mode

FedAvg分布式联邦学习的Sharding（分片）模式实现。**完全基于Hyperledger Fabric和IPFS实现去中心化通信**。

## ⚠️ 重要提示

**FedAvg方法完全基于Fabric+IPFS，无需共享文件系统**：
- 服务器通过Fabric查询客户端模型（key格式：`deviceID-taskID-epoch`）
- 所有模型文件存储在IPFS上
- 所有元数据存储在Fabric区块链上
- 完全去中心化，支持任意数量的客户端

## 架构说明

**重要**：`serverRun.py` 和 `clientRun.py` **必须在不同的物理节点上运行**。

- **服务器节点**：运行 `serverRun.py`
  - 初始化模型并上传到IPFS
  - 通过Fabric发布任务
  - **通过Fabric查询客户端模型**（并行查询）
  - 从IPFS下载模型文件并聚合
  - 发布聚合结果到Fabric

- **客户端节点**：运行 `clientRun.py`
  - 通过Fabric查询任务信息
  - 从IPFS下载全局模型
  - 本地训练
  - 上传模型到IPFS
  - **通过Fabric发布模型信息**（key：`deviceID-taskID-epoch`）

## 运行方法

### 服务器节点

**前置条件**：Fabric网络在服务器节点运行，IPFS daemon已启动

```bash
cd /home/pi/Sharding-DAG/fedAvg
python3 serverRun.py --dataset mnist --model cnn --epochs 50 --frac 0.1 --num_users 300
```

**参数说明**：
- `--dataset`: 数据集类型 (mnist/cifar)
- `--model`: 模型类型 (cnn/mlp)
- `--epochs`: 训练轮数
- `--frac`: 每轮参与的客户端比例（默认0.1）
- `--num_users`: **总用户数量**（所有shard的总和）

**重要**：
- 服务器不需要指定`shard_id`，会自动聚合所有客户端模型
- `--num_users`应设置为所有shard的用户总数

### 客户端节点

**前置条件**：可以访问Fabric网络和IPFS

```bash
# 设置shard_id（通过环境变量或命令行参数）
export SHARD_ID=0
cd /home/pi/Sharding-DAG/fedAvg
python3 clientRun.py --dataset mnist --model cnn
```

**如果Fabric在远程服务器，使用SSH隧道**：
```bash
# 在单独终端运行（保持运行）
ssh -L 7051:localhost:7051 -L 9051:localhost:9051 -L 7050:localhost:7050 pi@192.168.137.208

# 在另一个终端运行
export SHARD_ID=0
python3 clientRun.py --dataset mnist --model cnn --fabric_host localhost
```

**参数说明**：
- `--shard_id`: Shard ID（必需，可通过环境变量`SHARD_ID`或命令行参数指定）
- `--fabric_host`: Fabric网络地址（可选，默认`192.168.137.208`）

## 前置条件

### 1. 环境变量（仅客户端需要）

```bash
# 客户端节点设置shard_id
echo 'export SHARD_ID=0' >> ~/.bashrc
source ~/.bashrc
```

### 2. dict_users文件

```bash
cd federatedLearning
python3 generate_shard_dict_users.py
```

生成的文件需要复制到所有节点：
- `commonComponent/dict_users_mnist_shard0.pkl` (用户ID: 0-99)
- `commonComponent/dict_users_mnist_shard1.pkl` (用户ID: 100-199)
- `commonComponent/dict_users_mnist_shard2.pkl` (用户ID: 200-299)

### 3. IPFS和Fabric网络

**IPFS配置**：
- 确保IPFS daemon已启动（可以在服务器或客户端节点上）
- 所有模型文件存储在IPFS上

**Fabric网络配置**：

1. **推荐方案：Fabric在服务器节点运行，客户端直接连接**
   ```bash
   # 服务器节点（192.168.137.208）
   cd /home/pi/fabric/fabric-samples/test-network
   ./network.sh up createChannel -ca
   cd /home/pi/Sharding-DAG
   ./start_mycc_network.sh
   
   # 客户端节点：配置/etc/hosts
   sudo bash -c 'echo "192.168.137.208 peer0.org1.example.com peer0.org2.example.com orderer.example.com" >> /etc/hosts'
   ```

2. **替代方案：使用SSH隧道（推荐用于测试）**
   ```bash
   # 在客户端节点上创建SSH隧道（保持运行）
   ssh -L 7051:localhost:7051 -L 9051:localhost:9051 -L 7050:localhost:7050 pi@192.168.137.208
   ```

## 工作原理

**完全基于Fabric+IPFS的去中心化架构**：

1. **任务发布**：服务器通过Fabric发布任务（key: `taskRelease`），客户端查询
2. **模型上传**：客户端训练后上传到IPFS，通过Fabric发布（key: `deviceID-taskID-epoch`）
3. **模型收集**：服务器通过Fabric并行查询所有设备，从IPFS下载模型文件
4. **结果发布**：服务器聚合后上传到IPFS，通过Fabric发布（key: `taskID`）

**Fabric Key设计**：
- `taskRelease`: 任务发布信息（全局唯一）
- `taskID`: 任务信息（每个任务一个key）
- `deviceID-taskID-epoch`: 设备模型信息（每个设备每个任务每个epoch唯一）

**关键特性**：
- ✅ 完全去中心化：无需共享文件系统
- ✅ 并行查询：多线程并行查询提高效率
- ✅ 容错机制：IPFS下载失败自动重试，Fabric查询失败自动跳过

## 部署示例

**服务器节点（192.168.137.208）**：
```bash
# 启动Fabric网络
cd /home/pi/fabric/fabric-samples/test-network
./network.sh up createChannel -ca
cd /home/pi/Sharding-DAG
./start_mycc_network.sh

# 启动IPFS daemon
ipfs daemon &

# 启动服务器
cd /home/pi/Sharding-DAG/fedAvg
python3 serverRun.py --dataset mnist --model cnn --epochs 50 --frac 0.1 --num_users 100
```

**客户端节点（alice）**：
```bash
# 配置/etc/hosts（如果直接连接）
sudo bash -c 'echo "192.168.137.208 peer0.org1.example.com peer0.org2.example.com orderer.example.com" >> /etc/hosts'

# 运行客户端
export SHARD_ID=0
cd /home/pi/Sharding-DAG/fedAvg
python3 clientRun.py --dataset mnist --model cnn
```

**重要提示**：
- 先启动服务器节点，再启动客户端节点
- 每个节点只运行一个程序（服务器运行serverRun.py，客户端运行clientRun.py）
- 所有节点必须可以访问IPFS和Fabric网络
- **完全去中心化**：无需共享文件系统，所有通信通过Fabric+IPFS

## 与DAG方法的对比

| 特性 | DAG方法 | FedAvg方法 |
|-----|---------|-----------|
| 通信方式 | Socket + Fabric | **完全Fabric** |
| 模型存储 | IPFS | IPFS |
| 元数据存储 | Socket + Fabric | Fabric |
| 去中心化 | 部分（需要socket） | **完全去中心化** |
| 文件系统依赖 | 部分 | **无** |

**优势**：
- ✅ 完全去中心化：无需Socket连接
- ✅ 更好的可扩展性：支持任意数量的客户端
- ✅ 更强的容错性：单个客户端故障不影响整体
- ✅ 数据一致性：所有数据存储在Fabric区块链上
