#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量运行实验脚本
运行参数组合：
- method: 'baseline', 'load', 'nn' - 选择运行的方法
- attack_type: none (只运行一次), lazy, delay, labelflip (每个运行多个恶意节点比例)
- malicious_frac (恶意节点比例): 0.1, 0.2, 0.3 (none只使用0.1，其他运行所有值)
- epochs: 50
- frac (客户端参与比例): 使用默认值0.1
- labelflip_mode: random (默认) 或 fixed
- seed: none使用默认值，其他攻击类型使用123
- 数据分布: 非IID (NIID) - 默认模式，不添加--iid参数
- 其他参数使用默认值 
"""

import subprocess
import sys
import os

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ==================== 配置参数 ====================
# 选择运行的方法：'baseline', 'load', 'nn'
METHOD = 'baseline'  # 修改这里来选择运行的方法

# Conda环境名称
conda_env = 'sharding-dag'

# 实验参数配置
# attack_types = ['none', 'lazy', 'delay', 'labelflip']  # 原始配置：包含所有攻击类型
attack_types = ['none', 'delay', 'labelflip']  # 当前配置：只运行none, delay, labelflip
# 'lazy' 和 'noise' 已注释，不运行

malicious_frac_values = [0.1, 0.3, 0.5]  # 恶意节点比例：0.1, 0.3, 0.5
# malicious_frac_values = [0.1, 0.2, 0.3]  # 原始配置：0.1, 0.2, 0.3
epochs = 50
client_frac = 0.1  # 客户端参与比例（固定值）
labelflip_mode = 'fixed'  # labelflip模式：使用'fixed'而不是'random'
# labelflip_mode = 'random'  # 原始配置：'random' 或 'fixed'
seed_for_attacks = 123  # 除none以外的攻击类型使用的随机种子

# 根据方法选择主程序文件
method_files = {
    'baseline': 'main_fed_baseline.py',
    'load': 'main_fed_load.py',
    'nn': 'main_nn.py'
}

if METHOD not in method_files:
    print(f"错误：未知的方法 '{METHOD}'。请选择 'baseline', 'load', 或 'nn'")
    sys.exit(1)

main_file = method_files[METHOD]

# 基础命令 - 使用conda run在指定环境中运行
base_cmd = [
    'conda', 'run', '-n', conda_env, '--no-capture-output',
    'python',
    main_file,
    '--epochs', str(epochs),
    '--frac', str(client_frac),  # 固定客户端参与比例
]

# 计算总实验数：none运行1次，其他攻击类型各运行3次
total_experiments = 1 + (len(attack_types) - 1) * len(malicious_frac_values)
current_experiment = 0

print(f"开始运行 {total_experiments} 个实验（方法: {METHOD}）...")
print(f"运行方法: {METHOD} ({main_file})")
print(f"数据分布模式: 非IID (NIID) - 客户端数据非独立同分布")
print(f"攻击类型: {', '.join(attack_types)} (lazy和noise已注释，不运行)")
print(f"恶意节点比例: {malicious_frac_values}")
print(f"客户端参与比例 (frac): {client_frac} (固定)")
print(f"延迟轮数 (delay_rounds): 3 (默认)")
# print(f"噪声尺度 (noise_scale): 0.1 (默认)")  # noise攻击已注释
print(f"标签翻转模式 (labelflip_mode): {labelflip_mode} (使用fixed模式)")
print(f"随机种子 (seed): none使用默认值，其他攻击类型使用 {seed_for_attacks}")
print("=" * 60)

for attack_type in attack_types:
    # none只运行一次（使用默认恶意节点比例），其他攻击类型运行所有恶意节点比例
    if attack_type == 'none':
        malicious_frac_list = [0.1]  # none只使用0.1
    else:
        malicious_frac_list = malicious_frac_values
    
    for malicious_frac in malicious_frac_list:
        current_experiment += 1
        print(f"\n[{current_experiment}/{total_experiments}] 运行实验: method={METHOD}, attack_type={attack_type}, malicious_frac={malicious_frac}, epochs={epochs}, iid=False (NIID)")
        print("-" * 60)
        
        # 构建命令（设置恶意节点比例）
        cmd = base_cmd + [
            '--attack_type', attack_type,
            '--malicious_frac', str(malicious_frac),
        ]
        
        # 除none以外，添加seed参数
        if attack_type != 'none':
            cmd.extend(['--seed', str(seed_for_attacks)])
        
        # 如果是labelflip攻击，添加labelflip相关参数
        if 'labelflip' in attack_type:
            cmd.extend(['--labelflip_mode', labelflip_mode])
        
        # 执行命令
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✓ 实验完成: method={METHOD}, attack_type={attack_type}, malicious_frac={malicious_frac}")
        except subprocess.CalledProcessError as e:
            print(f"✗ 实验失败: method={METHOD}, attack_type={attack_type}, malicious_frac={malicious_frac}")
            print(f"错误代码: {e.returncode}")
            continue

print("\n" + "=" * 60)
print("所有实验运行完成！")

