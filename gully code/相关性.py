import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
def load_data(file_path):
    # 读取Excel文件中的指定工作簿
    data = pd.read_excel(r'C:\Users\huang\Desktop\训练.xlsx', sheet_name='14相关性')

    return data

# 计算相关性矩阵
def compute_correlation(data):
    correlation_matrix = data.corr()
    return correlation_matrix

# 绘制左侧的散点图矩阵
def plot_scatter_matrix(data):
    sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle("散点图矩阵", size=16)
    plt.show()

# 绘制右侧的相关性热力图
def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title("相关性热力图", size=16)
    plt.show()

# 主程序
def main(file_path):
    # 加载数据
    data = load_data(file_path)
    
    # 计算相关性矩阵
    correlation_matrix = compute_correlation(data)
    
    # 绘制散点图矩阵
    plot_scatter_matrix(data)
    
    # 绘制相关性热力图
    plot_correlation_heatmap(correlation_matrix)

# 执行程序
file_path = r'C:\Users\huang\Desktop\训练.xlsx'  # 您的Excel文件路径
main(file_path)
