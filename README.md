# Sharding-DAG
Implementation of Sharding-DAG.

## Requirments

- [Hyperledge Fabric 2.1](https://hyperledger-fabric.readthedocs.io/en/release-2.1/test_network.html#before-you-begin)
- Python3
- Pytorch
- Torchvision
- [IPFS](https://docs.ipfs.io/install/command-line/#official-distributions)

**Attention**: The configs of Fabric in `./commonComponent/interRun.sh` should be modified to access the Fabric deployed above. Besides, this file should be authorized with the right of *writer/read/run*.

## Deployment of DAG

The DAG could be deployed on a personal computer or the cloud server.

Copy all the files of this repository to the PC or cloud server, and then run following commands in the root path of this repository.

```
# For MNIST dataset (default)
cd dagMainChain
python3 serverRun.py

# For CIFAR-10 dataset
cd dagMainChain
python3 serverRun.py --dataset cifar --num_channels 3
``` 

## Run one shard

The shard also could be deployed on a personal computer or the cloud server.

Copy all the files of this repository to the deployment location, and modify `line 466` of `dagMainChain/clientRun.py` for the real address of the DAG server deployed above. Then run following commands in the root path of this repository.

```
# run the DAG client (MNIST dataset)
cd /home/pi/Sharding-DAG/dagMainChain
python3 clientRun.py --epochs 1 --frac 0.1 --gpu -1 --model cnn --dataset mnist --num_channels 1 --iid --attack_type none --malicious_frac 0.0 --lazy_timeout 180

python3 clientRun.py --epochs 1 --frac 0.1 --gpu -1 --model cnn --dataset mnist --num_channels 1 --iid --attack_type lazy --malicious_frac 0.1 --lazy_timeout 180

# run the FL task (MNIST dataset)
cd /home/pi/Sharding-DAG/federatedLearning
python3 main_fed_local.py --epochs 1 --frac 0.1 --gpu -1 --model cnn --dataset mnist --num_channels 1 --iid --attack_type none --malicious_frac 0.0 --lazy_timeout 180

python3 main_fed_local.py --epochs 1 --frac 0.1 --gpu -1 --model cnn --dataset mnist --num_channels 1 --iid --attack_type lazy --malicious_frac 0.1 --lazy_timeout 180

# For CIFAR-10 dataset, use:
# 运行 clientRun.py 时
cd /home/pi/Sharding-DAG/dagMainChain
python3 clientRun.py --epochs 1 --frac 0.1 --gpu -1 --model cnn --dataset cifar --num_channels 3 --iid --attack_type none --malicious_frac 0.0 --lazy_timeout 180 --lr 0.005

# 运行 main_fed_local.py 时
cd /home/pi/Sharding-DAG/federatedLearning
python3 main_fed_local.py --epochs 1 --frac 0.1 --gpu -1 --model cnn --dataset cifar --num_channels 3 --iid --attack_type none --malicious_frac 0.0 --lazy_timeout 180 --lr 0.005
```

The details of these parameters could be found in file `federatedLearning/utils/options.py`. 
It should be noted that the `--epochs` configured in command with `clientRun.py` represents the number of rounds run in each shard.
And the `--epochs` configured in command with `main_fed_local.py` represents the number of epochs run on each local device.

## Dataset Configuration

### MNIST Dataset
- Use `--dataset mnist`
- Set `--num_channels 1` (grayscale images)
- Supports both IID (`--iid`) and non-IID settings

### CIFAR-10 Dataset
- Use `--dataset cifar`
- Set `--num_channels 3` (RGB images)
- **Only supports IID setting** (`--iid` flag is required)
- The dataset will be automatically downloaded to `../data/cifar/` on first run
- Automatically uses `CNNCifar` model when `--model cnn` and `--dataset cifar` are specified

## Run multiple shards

Similar to the above, copy all the files of this repository and then modify the files and execute the commands presented above.

Besides, the para of `nodeNum` in `line 58` of `dagMainChain/clientRun.py` indicates the shard index which should be modified.

## Acknowledgments

Acknowledgments give to [shaoxiongji](https://github.com/shaoxiongji/federated-learning) and [AshwinRJ](https://github.com/AshwinRJ/Federated-Learning-PyTorch) for the basic codes of the FL module.

