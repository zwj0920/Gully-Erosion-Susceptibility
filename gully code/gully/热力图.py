import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取Excel文件
file_path = r"C:\Users\huang\Desktop\训练.xlsx"  
data = pd.read_excel(file_path, sheet_name=9)  

# 变量名统一处理：大写化的变量名和首字母大写的标签
upper_case_vars = ['ndvi', 'spi', 'twi', 'dem', 'cm']
columns = data.columns
formatted_columns = [col.upper() if col in upper_case_vars else col.capitalize() for col in columns]

# 重新设置数据列名以用于后续操作
data.columns = formatted_columns

# 指定两个集合
gully_parameters = ['Perimeter', 'Area', 'Length', 'Width', 'Depth', 'Volume']
environmental_factors = ['Ndvi', 'Spi', 'Twi', 'Cm', 'Aspect', 'Slope', 'Dem', 'Artificial']

# 提取两个集合的数据
gully_data = data[gully_parameters]
env_data = data[environmental_factors]

# 计算两个集合之间的相关性
correlation_matrix = pd.DataFrame(index=gully_parameters, columns=environmental_factors, dtype=float)
p_values = pd.DataFrame(index=gully_parameters, columns=environmental_factors, dtype=float)

for i, gully_var in enumerate(gully_parameters):
    for j, env_var in enumerate(environmental_factors):
        corr, p_val = pearsonr(data[gully_var], data[env_var])
        correlation_matrix.loc[gully_var, env_var] = corr
        p_values.loc[gully_var, env_var] = p_val

# 创建显著性标识
significance_marks = np.where(p_values < 0.01, '**', 
                              np.where((p_values >= 0.01) & (p_values < 0.05), '*', ''))

# 绘制热力图
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', 
                      xticklabels=environmental_factors, yticklabels=gully_parameters, 
                      linewidths=0.3, linecolor='white', cbar_kws={"shrink": 0.8})

# 添加数字和显著性标识
for i in range(len(gully_parameters)):
    for j in range(len(environmental_factors)):
        value = correlation_matrix.iloc[i, j]
        mark = significance_marks[i, j]
        # 显示数字
        plt.text(j + 0.5, i + 0.5, f'{value:.2f}', color='black', 
                 ha='center', va='center', fontsize=12)  # 扩大字号但不加粗
        # 显示星号，放置在数字下方
        if mark:
            plt.text(j + 0.5, i + 0.35, mark, color='black', 
                     ha='center', va='center', fontsize=10)

# 设置标题、标签和布局
plt.title('2024 Gully Parameters and Environmental Factors Heatmap', fontsize=16, pad=20)
plt.xlabel('Environmental Factors', fontsize=14, labelpad=10)
plt.ylabel('Gully Parameters', fontsize=14, labelpad=10)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# 保存图片
output_path = r"C:\Users\huang\Desktop\论文图片\相关性热力图\2024 Gully Parameters and Environmental Factors Heatmap.png"
plt.savefig(output_path, dpi=600)
plt.show()
