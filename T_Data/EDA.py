import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 从CSV文件读取数据
def load_and_visualize(filename):
    # 读取数据
    df = pd.read_csv(filename)
    
    print("数据前5行:")
    print(df.head())
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为不同标签设置颜色
    color_map = {'B': 'red', 'G': 'green', 'M': 'blue'}
    
    # 绘制散点图
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        ax.scatter(subset['speed'], subset['taste'], subset['service'], 
                  c=color_map[label], label=label, s=60, alpha=0.7)
    
    # 设置标签和标题
    ax.set_xlabel('Speed')
    ax.set_ylabel('Taste')
    ax.set_zlabel('Service')
    ax.set_title('3D Data Visualization')
    ax.legend()
    
    plt.show()

load_and_visualize('./T_Data/all_data.csv')