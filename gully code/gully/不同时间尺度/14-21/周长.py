import pandas as pd  
import statsmodels.api as sm  
from statsmodels.formula.api import mixedlm  
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_absolute_error, r2_score  
from sklearn.preprocessing import StandardScaler
import matplotlib
import seaborn as sns
import numpy as np

# 读取数据  
file_path = r"C:\Users\huang\Desktop\训练.xlsx"
data = pd.read_excel(file_path, sheet_name=5)  # 使用 sheet_name=5，即周长的数据

# 标准化数据
scaler = StandardScaler()
normalized_data = data.copy()
normalized_data[['ndvi', 'spi', 'twi', 'cm', 'aspect', 'slope', 'dem', 'artificial']] = scaler.fit_transform(
    data[['ndvi', 'spi', 'twi', 'cm', 'aspect', 'slope', 'dem', 'artificial']]
)

# 定义模型公式  
formula = '周长 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial'  

# 构建GLMM模型，使用id和time作为随机效应  
model = mixedlm(formula, normalized_data, groups=normalized_data["id"], re_formula="~time")  
result = model.fit()  

# 查看结果  
print(result.summary())

# 添加预测值和残差到数据框  
normalized_data['Predicted'] = result.fittedvalues  
normalized_data['Residuals'] = result.resid  

# 计算MAE和R²  
train_mae = mean_absolute_error(normalized_data['周长'], normalized_data['Predicted'])  
train_r2 = r2_score(normalized_data['周长'], normalized_data['Predicted'])  
print(f"Training set MAE: {train_mae:.4f}")  
print(f"Training set R²: {train_r2:.4f}")  

# 计算各因素对周长的贡献应用平方根构构的素中值。
contribution_ratio = np.abs(result.params[1:9]) / np.sum(np.abs(result.params[1:9]))
contribution_ratio_df = pd.DataFrame({'Factor': ['ndvi', 'spi', 'twi', 'cm', 'aspect', 'slope', 'dem', 'artificial'], 'Contribution': contribution_ratio})

# 按照贡献比例从大到小排序，并将具体贡献数值以百分比显示
contribution_ratio_df = contribution_ratio_df.sort_values(by='Contribution', ascending=False)
contribution_ratio_df['Contribution'] = (contribution_ratio_df['Contribution'] * 100).round(2)

# 打印排序后的贡献比例
print(contribution_ratio_df)

# 绘制因素贡献柱状图，颜色从大到小蓝色渐变（深色表示贡献较大）
plt.figure(figsize=(12, 8))
gradient_palette = sns.color_palette("Blues", len(contribution_ratio_df))[::-1]  # 贡献越大颜色越深
bar_plot = sns.barplot(x='Contribution', y='Factor', data=contribution_ratio_df, palette=gradient_palette, orient='h')

# 在每个柱子末尾添加具体百分比数值
for index, value in enumerate(contribution_ratio_df['Contribution']):
    bar_plot.text(value + 0.5, index, f'{value}%', color='black', va="center")

plt.xlabel('Contribution Proportion (%)', fontsize=14)
plt.ylabel('Factor', fontsize=14)
plt.title('Factor Contribution to Gully Circumference', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('14-21 Factor_Contribution.png', dpi=300)
plt.show()
