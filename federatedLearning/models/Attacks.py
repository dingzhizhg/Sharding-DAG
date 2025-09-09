#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np


class DelayAttackManager:
    """
    延迟攻击管理器：管理延迟队列，实现延迟提交攻击
    """
    def __init__(self):
        self.delay_queue = {}  # {node_id: {round: model_weights}}
    
    def save_delayed_model(self, node_id, round, model_weights):
        """
        保存当前轮次的模型到延迟队列
        
        参数:
            node_id: 节点ID
            round: 当前轮次
            model_weights: 模型权重字典
        """
        if node_id not in self.delay_queue:
            self.delay_queue[node_id] = {}
        self.delay_queue[node_id][round] = copy.deepcopy(model_weights)
    
    def get_delayed_model(self, node_id, current_round, delay_rounds=1):
        """
        获取延迟的模型（返回delay_rounds轮之前的模型）
        
        参数:
            node_id: 节点ID
            current_round: 当前轮次
            delay_rounds: 延迟轮数（默认1）
        
        返回:
            model_weights: delay_rounds轮之前的模型权重，如果不存在则返回None
        """
        if node_id not in self.delay_queue:
            return None
        
        delayed_round = current_round - delay_rounds
        if delayed_round >= 0 and delayed_round in self.delay_queue[node_id]:
            return copy.deepcopy(self.delay_queue[node_id][delayed_round])
        
        return None
    
    def clear_old_models(self, current_round, max_delay=1):
        """
        清理过期的延迟模型
        
        参数:
            current_round: 当前轮次
            max_delay: 最大延迟轮数，超过此轮数的模型将被清理
        """
        for node_id in list(self.delay_queue.keys()):
            rounds_to_remove = [
                r for r in self.delay_queue[node_id].keys()
                if r < current_round - max_delay
            ]
            for r in rounds_to_remove:
                del self.delay_queue[node_id][r]
            
            # 如果节点没有延迟模型了，删除该节点
            if not self.delay_queue[node_id]:
                del self.delay_queue[node_id]


def noise_attack(w, noise_scale=0.01):
    """
    噪声攻击：在模型参数上添加高斯噪声
    
    参数:
        w: 模型权重字典
        noise_scale: 噪声标准差（默认0.01）
    
    返回:
        w_poisoned: 被污染的模型权重字典
    """
    w_poisoned = copy.deepcopy(w)
    
    for k in w_poisoned.keys():
        # 对每个参数添加高斯噪声
        noise = torch.randn_like(w_poisoned[k]) * noise_scale
        w_poisoned[k] = w_poisoned[k] + noise
    
    return w_poisoned


def create_labelflip_mapping(num_classes, mode='random', target_label=None, seed=None):
    """
    创建标签翻转映射表
    
    参数:
        num_classes: 类别数量
        mode: 翻转模式，'random' 为随机翻转，'fixed' 为固定翻转到 target_label
        target_label: 固定翻转模式下的目标标签（如果为 None，则使用 num_classes-1）
        seed: 随机种子（用于可复现性）
    
    返回:
        label_mapping: 字典，{原始标签: 翻转后的标签}
    """
    if seed is not None:
        np.random.seed(seed)
    
    label_mapping = {}
    
    if mode == 'fixed':
        # 固定翻转模式：所有标签都翻转到 target_label
        if target_label is None:
            target_label = num_classes - 1
        target_label = int(target_label) % num_classes
        
        for i in range(num_classes):
            if i == target_label:
                # 如果原始标签就是目标标签，随机选择另一个标签
                other_labels = [j for j in range(num_classes) if j != target_label]
                label_mapping[i] = int(np.random.choice(other_labels))
            else:
                label_mapping[i] = target_label
    else:
        # 随机翻转模式：每个标签随机翻转到其他标签
        for i in range(num_classes):
            # 随机选择一个不同于原始标签的标签
            other_labels = [j for j in range(num_classes) if j != i]
            label_mapping[i] = int(np.random.choice(other_labels))
    
    return label_mapping

