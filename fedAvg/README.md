# FedAvg 分布式联邦学习实现

这个文件夹包含FedAvg算法的分布式实现，使用Fabric区块链和IPFS进行通信。

## 文件结构

- `serverRun.py`: 聚合服务器节点，负责初始化模型、发布任务、聚合客户端模型、发布聚合结果
- `clientRun.py`: 客户端训练节点，负责查询任务、下载全局模型、本地训练、上传模型

## 架构说明

### Server (聚合节点)
- 初始化genesis模型并上传到IPFS
- 通过Fabric区块链发布任务
- 等待客户端上传训练后的模型
- 使用FedAvg算法聚合模型
- 将聚合后的模型上传到IPFS并发布到区块链

### Client (客户端节点)
- 每个客户端节点运行一个独立的进程
- 查询区块链上的任务信息
- 从IPFS下载全局聚合模型
- 使用本地数据训练模型
- 将训练后的模型上传到IPFS和区块链

## 使用方法

### 1. 启动服务器（聚合节点）

#### 场景：3个树莓派，每个负责100个设备，每轮收集30个设备模型

```bash
cd fedAvg
python serverRun.py \
    --num_users 300 \
    --frac 0.1 \
    --epochs 50 \
    --dataset mnist \
    --model cnn \
    --local_ep 5 \
    --lr 0.01 \
    --lazy_timeout 120
```

**参数说明**：
- `--num_users 300`: 总设备数 = 3个树莓派 × 100个设备/树莓派
- `--frac 0.1`: 每轮参与比例，服务器会收集 300 × 0.1 = 30个设备模型
- `--lazy_timeout 120`: 等待客户端上传模型的超时时间（秒）

服务器会：
1. 初始化模型
2. 上传genesis模型到IPFS
3. 发布任务到区块链
4. 等待客户端训练并聚合模型

### 2. 启动客户端（训练节点）

#### 方式一：每个树莓派负责固定的设备范围（推荐用于多树莓派部署）

假设有3个树莓派，每个树莓派负责100个设备：

**树莓派1（node_id=0）**：
```bash
cd fedAvg
python clientRun.py \
    --node_id 0 \
    --devices_per_node 100 \
    --num_users 300 \
    --frac 0.1 \
    --dataset mnist \
    --model cnn \
    --epochs 50
```
负责设备：device00000 到 device00099，每轮选择10个设备

**树莓派2（node_id=1）**：
```bash
cd fedAvg
python clientRun.py \
    --node_id 1 \
    --devices_per_node 100 \
    --num_users 300 \
    --frac 0.1 \
    --dataset mnist \
    --model cnn \
    --epochs 50
```
负责设备：device00100 到 device00199，每轮选择10个设备

**树莓派3（node_id=2）**：
```bash
cd fedAvg
python clientRun.py \
    --node_id 2 \
    --devices_per_node 100 \
    --num_users 300 \
    --frac 0.1 \
    --dataset mnist \
    --model cnn \
    --epochs 50
```
负责设备：device00200 到 device00299，每轮选择10个设备

**说明**：
- `--node_id`: 树莓派节点ID（0, 1, 2, ...）
- `--devices_per_node`: 每个树莓派负责的设备数（默认100）
- 每个树莓派从自己的设备范围内随机选择 `frac * devices_per_node` 个设备
- 服务器会收集所有树莓派上传的模型（理论上每轮30个设备）

#### 方式二：每个树莓派从全部设备中随机选择（默认行为）

```bash
cd fedAvg
python clientRun.py \
    --num_users 300 \
    --frac 0.033 \
    --dataset mnist \
    --model cnn \
    --epochs 50
```

**重要**：如果不指定`--node_id`，客户端会从全部设备中随机选择

### 3. 参数说明

服务器和客户端共享的参数（通过args_parser）：
- `--dataset`: 数据集类型 (mnist/cifar)
- `--model`: 模型类型 (cnn/mlp)
- `--epochs`: 训练轮数
- `--num_users`: 总用户数（服务器端：所有设备总数；客户端：必须与服务器一致）
- `--frac`: 每轮参与的客户端比例
  - 服务器端：从全部设备中选择的比例（例如：300 × 0.1 = 30个设备）
  - 客户端（使用--node_id时）：从当前节点负责的设备中选择的比例（例如：100 × 0.1 = 10个设备）
- `--node_id`: 树莓派节点ID（0, 1, 2, ...），仅客户端使用
- `--devices_per_node`: 每个树莓派负责的设备数（默认100），仅客户端使用
- `--iid`: 是否IID数据分布
- `--local_ep`: 本地训练epoch数
- `--lr`: 学习率
- `--attack_type`: 攻击类型 (none/lazy/noise/labelflip/delay)
- `--malicious_frac`: 恶意节点比例
- `--lazy_timeout`: 等待客户端的超时时间（秒）

## 工作流程

1. **服务器初始化**
   - 创建模型
   - 上传genesis模型到IPFS
   - 保存dict_users到文件

2. **任务发布**
   - 服务器通过Fabric区块链发布任务
   - 任务包含：taskID, epochs, status, usersFrac

3. **客户端训练**
   - 客户端查询任务
   - 下载全局模型（从IPFS）
   - 本地训练
   - 上传模型到IPFS
   - 发布模型哈希到区块链

4. **服务器聚合**
   - 查询区块链获取所有客户端模型
   - 从IPFS下载模型文件
   - 使用FedAvg聚合
   - 上传聚合模型到IPFS
   - 发布聚合结果到区块链

5. **重复步骤3-4**直到所有epoch完成

## 目录结构

运行后会在以下位置生成文件：

```
fedAvg/
├── serverS/          # 服务器端文件
│   ├── paras/        # 模型参数
│   │   ├── local/    # 客户端模型
│   │   └── agg/      # 聚合模型
│   └── genesis_model.pkl
│
└── clientS/          # 客户端文件（每个节点）
    ├── paras/        # 下载的聚合模型
    └── local/        # 本地训练的模型
```

结果文件保存在：
```
data/result/
├── fedavg-{attack}-{mal_frac}-{timestamp}.csv
└── fedavg-{attack}-{mal_frac}-{timestamp}_loss.csv
```

## 注意事项

1. **IPFS和Fabric网络**：确保IPFS daemon和Fabric网络已启动
2. **设备ID**：每个客户端节点必须使用唯一的device_id
3. **dict_users**：服务器和客户端必须使用相同的dict_users文件（通过commonComponent/dict_users.pkl共享）
4. **网络连接**：确保所有节点可以访问IPFS和Fabric网络
5. **超时设置**：根据网络情况调整lazy_timeout参数

## 与main_fed_baseline.py的区别

- `main_fed_baseline.py`: 中心化实现，在单进程中模拟所有客户端
- `fedAvg/`: 分布式实现，客户端和服务器分离，支持多节点部署

## 故障排查

1. **客户端无法连接**：检查IPFS和Fabric网络是否正常运行
2. **模型下载失败**：检查IPFS哈希是否正确，IPFS daemon是否运行
3. **任务查询超时**：检查Fabric网络连接
4. **设备ID冲突**：确保每个客户端使用不同的device_id

