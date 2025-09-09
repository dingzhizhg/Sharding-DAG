# 参数汇总表

## 联邦学习参数 (Federated Learning Parameters)

| 参数名称 | 默认值 | 说明 |
|---------|--------|------|
| `epochs` | 10 | 训练轮数 (rounds of training) |
| `num_users` | 100 | 用户数量 (number of users: K) |
| `frac` | 0.1 | 客户端参与比例 (the fraction of clients: C) |
| `local_ep` | 5 | 本地训练轮数 (the number of local epochs: E) |
| `local_bs` | 10 | 本地批次大小 (local batch size: B) |
| `bs` | 128 | 测试批次大小 (test batch size) |
| `lr` | 0.01 | 学习率 (learning rate) |



| `num_shards` (MNIST non-IID) | 200 | MNIST非IID场景下的数据分片数量 (Number of shards for MNIST non-IID) |
| `num_imgs` (MNIST non-IID) | 300 | 每个分片的图片数量 (Number of images per shard) |
| `shards_per_user` (MNIST non-IID) | 2 | 每个用户分配的分片数量 (Number of shards assigned to each user) |



