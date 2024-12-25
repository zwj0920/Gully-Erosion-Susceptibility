import pandas as pd  
import statsmodels.api as sm  
from statsmodels.formula.api import mixedlm  
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_absolute_error, r2_score  
from sklearn.preprocessing import StandardScaler
import matplotlib
import seaborn as sns
import scipy.stats as stats

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

# 读取数据  
file_path = r"C:\Users\huang\Desktop\训练.xlsx"
data = pd.read_excel(file_path, sheet_name=3)  # 使用 sheet_name=3，即周长的数据

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

# Plot comparison of predicted and actual values 
plt.figure(figsize=(10, 6))  
sns.scatterplot(x='周长', y='Predicted', data=normalized_data, color='blue', alpha=0.6)  
plt.plot([normalized_data['周长'].min(), normalized_data['周长'].max()],   
         [normalized_data['周长'].min(), normalized_data['周长'].max()],   
         'r--', linewidth=2, label='y=x')  
plt.xlabel('Actual', fontsize=18)  
plt.ylabel('Predicted', fontsize=18)  
plt.title('Comparison of Actual and Predicted Gully Circumference', fontsize=20)  
plt.legend(title=f'MAE: {train_mae:.4f}\nR$^2$: {train_r2:.4f}', fontsize=16)  
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)  
plt.tight_layout()  
plt.savefig('Comparison_of_Actual_and_Predicted.png', dpi=300)  # Save image
plt.show()  

# 绘制残差分布图
plt.figure(figsize=(10, 6))
sns.histplot(normalized_data['Residuals'], kde=True, color='purple', bins=30)
plt.xlabel('Residuals', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.title('Residual Distribution of Gully Circumference', fontsize=20)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('Residual_Distribution.png', dpi=300)
plt.show()

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(normalized_data['Residuals'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Gully Circumference Residuals', fontsize=20)
plt.ylabel('Quantiles', fontsize=18)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('Q-Q_Plot.png', dpi=300)
plt.show()
