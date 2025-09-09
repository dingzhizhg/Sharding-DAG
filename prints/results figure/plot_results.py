import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义方法列表和对应的文件夹
methods = {
    'nn': 'nn',
    'local': 'local',
    'fed': 'fed',
    'baseline': 'baseline',
    # 'asyn': 'asyn'  # 注释掉asyn方法
}

# 定义方法标签映射（按顺序：central, sharding-dag, fedavg, fedasync）
method_labels = {
    'nn': 'centralized',
    'local': 'sharding-dag',
    'fed': 'fedavg',
    'baseline': 'fedasync'
}

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))  # prints/results figure/
# 数据文件在上一级目录（prints/）的子文件夹中
data_dir = os.path.dirname(base_dir)  # prints/

# 存储所有方法的数据
acc_data = {}
loss_data = {}

# 读取每个方法的数据
for method_name, folder_name in methods.items():
    acc_file = os.path.join(data_dir, folder_name, 'acc.csv')
    loss_file = os.path.join(data_dir, folder_name, 'loss.csv')
    
    # 读取acc数据
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            lines = f.readlines()
            # 跳过第一行（方法名），读取第2-51行（50轮数据）
            acc_values = []
            for i in range(1, min(52, len(lines))):  # 读取50轮
                line = lines[i].strip()
                if line:
                    try:
                        value = float(line)
                        # 如果是local方法且值大于1，则除以100（百分比转小数）
                        if method_name == 'local' and value > 1:
                            value = value / 100.0
                        acc_values.append(value)
                    except ValueError:
                        continue
            if acc_values:
                acc_data[method_name] = acc_values
    
    # 读取loss数据
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            lines = f.readlines()
            # 跳过第一行（方法名），读取第2-51行（50轮数据）
            loss_values = []
            for i in range(1, min(52, len(lines))):  # 读取50轮
                line = lines[i].strip()
                if line:
                    try:
                        value = float(line)
                        loss_values.append(value)
                    except ValueError:
                        continue
            if loss_values:
                loss_data[method_name] = loss_values

# 绘制acc对比图
plt.figure(figsize=(8, 4))
# 按顺序绘制：nn, local, fed, baseline
for method_name in ['nn', 'local', 'fed', 'baseline']:
    if method_name in acc_data:
        values = acc_data[method_name]
        epochs = range(1, len(values) + 1)
        label = method_labels.get(method_name, method_name)
        plt.plot(epochs, values, label=label, linewidth=2, marker='o', markersize=3)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Comparison (50 Epochs)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(1, 50)
plt.tight_layout()
acc_plot_path = os.path.join(base_dir, 'acc_comparison.png')
plt.savefig(acc_plot_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
print(f'Accuracy comparison plot saved to: {acc_plot_path}')
plt.close()

# 绘制loss对比图
plt.figure(figsize=(8, 4))
# 按顺序绘制：nn, local, fed, baseline
for method_name in ['nn', 'local', 'fed', 'baseline']:
    if method_name in loss_data:
        values = loss_data[method_name]
        epochs = range(1, len(values) + 1)
        label = method_labels.get(method_name, method_name)
        plt.plot(epochs, values, label=label, linewidth=2, marker='o', markersize=3)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Comparison (50 Epochs)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(1, 50)
plt.tight_layout()
loss_plot_path = os.path.join(base_dir, 'loss_comparison.png')
plt.savefig(loss_plot_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
print(f'Loss comparison plot saved to: {loss_plot_path}')
plt.close()

print('\nAll plots generated successfully!')

