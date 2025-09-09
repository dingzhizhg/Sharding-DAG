#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from typing import Set, List, Optional


def select_malicious_nodes(selected_nodes: List[int], malicious_frac: float, 
                          attack_type: str, seed: Optional[int] = None) -> Set[int]:
    """
    从选择的节点中按比例选择恶意节点
    
    参数:
        selected_nodes: 当前轮选择的节点ID列表
        malicious_frac: 恶意节点比例 (0.0-1.0)
        attack_type: 攻击类型
        seed: 随机种子（可选）
    
    返回:
        Set[int]: 恶意节点ID集合
    """
    if malicious_frac <= 0 or attack_type == 'none' or not selected_nodes:
        return set()
    
    # 计算恶意节点数量
    num_malicious = max(1, int(malicious_frac * len(selected_nodes)))
    num_malicious = min(num_malicious, len(selected_nodes))
    
    # 使用局部随机状态，避免影响全局随机状态
    if seed is not None:
        # 保存当前随机状态
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        
        # 设置局部随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 随机选择恶意节点
        malicious_nodes = set(random.sample(selected_nodes, num_malicious))
        
        # 恢复原始随机状态
        random.setstate(random_state)
        np.random.set_state(np_random_state)
    else:
        # 如果没有提供种子，直接随机选择
        malicious_nodes = set(random.sample(selected_nodes, num_malicious))
    
    return malicious_nodes


def is_malicious_node(node_id: int, malicious_nodes: Set[int]) -> bool:
    """
    判断节点是否为恶意节点
    
    参数:
        node_id: 节点ID
        malicious_nodes: 恶意节点集合
    
    返回:
        bool: 是否为恶意节点
    """
    return node_id in malicious_nodes


def select_fixed_lazy_nodes(num_users: int, lazy_frac: float, seed: Optional[int] = None) -> Set[int]:
    """
    选择固定的lazy节点集合，在所有轮次中都保持lazy
    
    参数:
        num_users: 总用户数量
        lazy_frac: lazy节点比例 (0.0-1.0)，相对于总用户数
        seed: 随机种子（可选）
    
    返回:
        Set[int]: 固定的lazy节点ID集合
    """
    if lazy_frac <= 0:
        return set()
    
    # 计算lazy节点数量（相对于总用户数）
    num_lazy = max(1, int(lazy_frac * num_users))
    num_lazy = min(num_lazy, num_users)
    
    # 使用局部随机状态，避免影响全局随机状态
    if seed is not None:
        # 保存当前随机状态
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        
        # 设置局部随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 随机选择固定的lazy节点
        lazy_nodes = set(np.random.choice(range(num_users), num_lazy, replace=False))
        
        # 恢复原始随机状态
        random.setstate(random_state)
        np.random.set_state(np_random_state)
    else:
        # 如果没有提供种子，直接随机选择
        lazy_nodes = set(np.random.choice(range(num_users), num_lazy, replace=False))
    
    return lazy_nodes


def is_lazy_node(node_id: int, malicious_nodes: Set[int], attack_type: str) -> bool:
    """
    判断节点是否为懒惰节点
    
    参数:
        node_id: 节点ID
        malicious_nodes: 恶意节点集合
        attack_type: 攻击类型
    
    返回:
        bool: 是否为懒惰节点
    """
    if 'lazy' not in attack_type:
        return False
    return node_id in malicious_nodes

