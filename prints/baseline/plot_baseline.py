import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
import glob
import re

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))  # prints/baseline/
baseline_dir = base_dir  # CSV文件与脚本在同一目录

# 存储所有数据
acc_data = {}

# 读取baseline文件夹中的所有CSV文件
csv_files = glob.glob(os.path.join(baseline_dir, 'baseline-*.csv'))

# 解析文件名并读取数据
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    # 解析文件名：新格式 baseline-{attack_type}-{malicious_frac}-{timestamp}.csv
    # 也兼容旧格式 baseline-{attack_type}-{malicious_frac}-{client_frac}-{timestamp}.csv
    match_new = re.match(r'baseline-(\w+)-([\d.]+)-\d+\.csv', filename)  # 新格式：attack_type-malicious_frac-timestamp
    match_old = re.match(r'baseline-(\w+)-([\d.]+)-([\d.]+)-\d+\.csv', filename)  # 旧格式：attack_type-malicious_frac-client_frac-timestamp
    
    if match_new:
        # 新格式：attack_type, malicious_frac
        attack_type = match_new.group(1)
        malicious_frac = match_new.group(2)
        label = f'{attack_type}-{malicious_frac}'
    elif match_old:
        # 旧格式：attack_type, malicious_frac, client_frac
        attack_type = match_old.group(1)
        malicious_frac = match_old.group(2)  # 第二个是malicious_frac
        label = f'{attack_type}-{malicious_frac}'
    else:
        continue
    
    # 读取acc数据
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        # 跳过第一行（方法名），读取第2-51行（50轮数据）
        acc_values = []
        for i in range(1, min(52, len(lines))):  # 读取50轮
            line = lines[i].strip()
            if line:
                try:
                    value = float(line)
                    acc_values.append(value)
                except ValueError:
                    continue
        if acc_values:
            acc_data[label] = {
                'values': acc_values,
                'attack_type': attack_type,
                'frac': float(malicious_frac)  # 使用malicious_frac
            }

# 获取none方法的数据
none_data = None
for label in acc_data.keys():
    if acc_data[label]['attack_type'] == 'none':
        none_data = acc_data[label]['values']
        break

# 定义四组对比：lazy, delay, noise, labelflip
# attack_groups = ['lazy', 'delay', 'noise', 'labelflip']
attack_groups = ['lazy', 'delay', 'labelflip']
# 为每组生成一张图
for attack_type in attack_groups:
    plt.figure(figsize=(8, 5))
    
    # 先绘制none方法（如果存在）
    if none_data:
        epochs = range(1, len(none_data) + 1)
        plt.plot(epochs, none_data, label='none', linewidth=2, marker='o', markersize=3, linestyle='--')
    
    # 绘制该attack_type的所有恶意节点比例值
    # 对于lazy攻击，只显示0.1, 0.3, 0.5三个比例
    lazy_allowed_fracs = [0.1, 0.3, 0.5] if attack_type == 'lazy' else None
    
    attack_labels = []
    for label in sorted(acc_data.keys(), key=lambda x: acc_data[x]['frac']):  # frac字段存储的是malicious_frac
        if acc_data[label]['attack_type'] == attack_type:
            # 如果是lazy攻击，检查frac是否在允许的列表中
            if lazy_allowed_fracs is not None:
                if acc_data[label]['frac'] not in lazy_allowed_fracs:
                    continue
            attack_labels.append(label)
            values = acc_data[label]['values']
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=label, linewidth=2, marker='o', markersize=3)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Baseline Accuracy Comparison: none vs {attack_type} (50 Epochs)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 50)
    plt.tight_layout()
    
    # 图像保存在当前目录（与CSV文件同一目录）
    acc_plot_path = os.path.join(baseline_dir, f'baseline_{attack_type}.png')
    plt.savefig(acc_plot_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f'Baseline accuracy comparison plot ({attack_type}) saved to: {acc_plot_path}')
    plt.close()

print('\nAll plots generated successfully!')
print(f'Generated {len(attack_groups)} comparison plots: {", ".join(attack_groups)} (each with none baseline)')

