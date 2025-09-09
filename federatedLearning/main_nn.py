#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import datetime
import os

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar


def test(net_g, data_loader, device):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # 准备测试数据集
    if args.dataset == 'mnist':
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    # 记录 loss 和 accuracy 的列表
    dateNow = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    acc_test_list = []
    loss_test_list = []
    loss_train_list = []

    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        loss_train_list.append(loss_avg)

        # 测试并记录
        test_acc, test_loss = test(net_glob, test_loader, args.device)
        loss_test_list.append(test_loss)
        acc_test_list.append(float(test_acc.item()) / float(len(dataset_test)))

        # 保存到 CSV 文件
        os.makedirs('./log', exist_ok=True)
        accDfTest = pd.DataFrame({'nn': acc_test_list})
        accDfTest.to_csv('./log/nn_{}_{}_{}epochs_ACC_{}.csv'.format(args.dataset, args.model, args.epochs, dateNow), index=False, sep=',', float_format='%.10f')

        lossDfTest = pd.DataFrame({'nn': loss_test_list})
        lossDfTest.to_csv('./log/nn_{}_{}_{}epochs_Loss_{}.csv'.format(args.dataset, args.model, args.epochs, dateNow), index=False, sep=',', float_format='%.10f')

        lossDfTrain = pd.DataFrame({'nn': loss_train_list})
        lossDfTrain.to_csv('./log/nn_{}_{}_{}epochs_Train_Loss_{}.csv'.format(args.dataset, args.model, args.epochs, dateNow), index=False, sep=',', float_format='%.10f')

        net_glob.train()

    # plot loss
    plt.figure()
    plt.plot(range(len(loss_train_list)), loss_train_list)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    print('\nFinal results:')
    print('Test accuracy list:', acc_test_list)
    print('Test loss list:', loss_test_list)
    print('Train loss list:', loss_train_list)
