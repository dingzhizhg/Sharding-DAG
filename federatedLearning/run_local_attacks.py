#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量运行local方法的攻击实验脚本
运行参数组合参考 plot_baseline.py：
- attack_type: none (只运行一次), lazy, delay, labelflip (每个运行多个恶意节点比例)
- malicious_frac (恶意节点比例): lazy只运行0.1, 0.3, 0.5；其他攻击类型也运行0.1, 0.3, 0.5
- epochs: 50 (默认，由main_fed_local.py从taskRelease获取)
- frac (客户端参与比例): 使用默认值0.1
- labelflip_mode: fixed (参考plot_baseline.py的配置)
- seed: none使用默认值，其他攻击类型使用123
- 数据分布: 非IID (NIID) - 默认模式，不添加--iid参数
- 其他参数使用默认值

注意：运行前请确保IPFS和server都已经启动
"""

import subprocess
import sys
import os

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ==================== 配置参数 ====================
# 实验参数配置 - 参考 plot_baseline.py
attack_types = ['none', 'lazy', 'delay', 'labelflip']  # 所有攻击类型

# 对于lazy攻击，只运行0.1, 0.3, 0.5三个比例（参考plot_baseline.py第84行）
# 对于其他攻击，也运行0.1, 0.3, 0.5（从CSV文件名可以看出）
lazy_frac_values = [0.1, 0.3, 0.5]  # lazy攻击的恶意节点比例
other_frac_values = [0.1, 0.3, 0.5]  # 其他攻击的恶意节点比例

client_frac = 0.1  # 客户端参与比例（固定值）
labelflip_mode = 'fixed'  # labelflip模式：使用'fixed'
seed_for_attacks = 123  # 除none以外的攻击类型使用的随机种子

main_file = 'main_fed_local.py'  # local方法的主程序文件

# 基础命令 - 直接运行python（不使用conda，因为用户说默认环境已配置好）
base_cmd = [
    'python',
    main_file,
    '--frac', str(client_frac),  # 固定客户端参与比例
]

# 计算总实验数：none运行1次，lazy运行3次，delay运行3次，labelflip运行3次
total_experiments = 1 + len(lazy_frac_values) + len(other_frac_values) * 2
current_experiment = 0

print("=" * 80)
print("开始运行 local 方法的攻击实验...")
print("=" * 80)
print(f"运行方法: local ({main_file})")
print(f"数据分布模式: 非IID (NIID) - 客户端数据非独立同分布")
print(f"攻击类型: {', '.join(attack_types)}")
print(f"Lazy攻击恶意节点比例: {lazy_frac_values}")
print(f"其他攻击恶意节点比例: {other_frac_values}")
print(f"客户端参与比例 (frac): {client_frac} (固定)")
print(f"延迟轮数 (delay_rounds): 3 (默认)")
print(f"标签翻转模式 (labelflip_mode): {labelflip_mode}")
print(f"随机种子 (seed): none使用默认值，其他攻击类型使用 {seed_for_attacks}")
print(f"总实验数: {total_experiments}")
print("=" * 80)
print("\n注意：请确保IPFS和server都已经启动！")
print("=" * 80)

for attack_type in attack_types:
    # none只运行一次（使用0.1），其他攻击类型根据类型选择比例
    if attack_type == 'none':
        malicious_frac_list = [0.1]  # none只使用0.1
    elif attack_type == 'lazy':
        malicious_frac_list = lazy_frac_values  # lazy只运行0.1, 0.3, 0.5
    else:
        malicious_frac_list = other_frac_values  # delay和labelflip运行0.1, 0.3, 0.5
    
    for malicious_frac in malicious_frac_list:
        current_experiment += 1
        print(f"\n[{current_experiment}/{total_experiments}] 运行实验: method=local, attack_type={attack_type}, malicious_frac={malicious_frac}")
        print("-" * 80)
        
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
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✓ 实验完成: method=local, attack_type={attack_type}, malicious_frac={malicious_frac}")
        except subprocess.CalledProcessError as e:
            print(f"✗ 实验失败: method=local, attack_type={attack_type}, malicious_frac={malicious_frac}")
            print(f"错误代码: {e.returncode}")
            continue
        except KeyboardInterrupt:
            print("\n\n用户中断实验！")
            sys.exit(1)

print("\n" + "=" * 80)
print("所有实验运行完成！")
print("=" * 80)

