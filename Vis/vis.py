import matplotlib.pyplot as plt
import numpy as np

def Vis_Data(data=None, bar_labels=None, groups=None, figsize=(12, 8), bar_width=0.25, 
             show_values=True, title='Comparison of the GNB, RF, and SVM three models'):
    """
    可视化多组数据的条形图
    
    参数:
    data: 二维列表，包含要可视化的数据
    bar_labels: 每组内条形的标签列表
    groups: 组标签列表
    figsize: 图形大小
    bar_width: 条形宽度
    show_values: 是否在条形上方显示数值
    title: 图表标题
    """
    
    # 设置中文字体（如果需要显示中文）
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 设置图形大小
    plt.figure(figsize=figsize)

    # 设置条形位置
    x_pos = np.arange(len(groups))

    # 创建条形图
    for i in range(len(bar_labels)):  # 根据bar_labels数量确定循环次数
        # 计算每个条形的位置
        positions = x_pos + i * bar_width
        # 绘制条形
        bars = plt.bar(positions, [group[i] for group in data], 
                       width=bar_width, label=bar_labels[i])

    # 自定义图表
    plt.xlabel('Models Evaluation', fontsize=12)
    plt.ylabel('Data', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # 设置x轴标签位置
    plt.xticks(x_pos + bar_width * (len(bar_labels)-1)/2, groups)

    # 添加图例
    plt.legend()

    # 添加数值标签在条形上方
    if show_values:
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}', ha='center', va='bottom')
        
        '''# 为所有条形添加数值标签
        for i in range(len(bar_labels)):
            positions = x_pos + i * bar_width
            bars = plt.bar(positions, [group[i] for group in data], width=bar_width)
            add_value_labels(bars)'''

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()

# 使用示例
if __name__ == "__main__":
    
    # 使用自定义数据
    model_data = [
            [25, 30, 35],  # 第1组
            [40, 35, 28],  # 第2组
            [22, 45, 33],  # 第3组
            [38, 42, 29],  # 第4组
            [31, 27, 39],  # 第5组
            [36, 33, 41]   # 第6组
        ]
    model_groups = ['Accuracy', 'Time Used', 'Precision', 'Recall', 'F1', 'N Samples']
    model_labels = ['GNB', 'RF', 'SVM']
    
    Vis_Data(data=model_data, groups=model_groups, bar_labels=model_labels, 
             title='Custom Model Comparison')