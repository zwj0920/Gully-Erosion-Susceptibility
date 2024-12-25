import pandas as pd  
import statsmodels.api as sm  
from statsmodels.formula.api import mixedlm  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.metrics import mean_absolute_error, r2_score  
import matplotlib
import scipy.stats as stats

# 读取数据  
file_path = r"C:\Users\huang\Desktop\训练.xlsx"  
data = pd.read_excel(file_path, sheet_name=4)  

# 定义模型公式  
formula = '周长 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial'  

# 构建GLMM，使用id和time作为随机效应  
model = mixedlm(formula, data, groups=data["id"], re_formula="~time")  
result = model.fit()  

# 查看结果  
print(result.summary())  

# 计算AIC和BIC
aic = result.aic
bic = result.bic
print(f"AIC: {aic:.4f}, BIC: {bic:.4f}")

# 添加预测值和残差到数据框  
data['预测值'] = result.fittedvalues  
data['残差'] = result.resid  

# 计算MAE和R²  
train_mae = mean_absolute_error(data['周长'], data['预测值'])  
train_r2 = r2_score(data['周长'], data['预测值'])  
print(f"训练集平均绝对误差（MAE）: {train_mae:.4f}")  
print(f"训练集R²: {train_r2:.4f}")  

# 计算各类影响因素的贡献占比
coef = result.params[1:]  # 排除截距项
importance = (coef / coef.sum()).abs()  # 计算标准化系数的绝对值
importance = importance / importance.sum()  # 归一化

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False  

# 选择特定的因素
selected_factors = ['ndvi', 'spi', 'twi', 'cm', 'aspect', 'slope', 'dem', 'artificial']
importance = importance[selected_factors]

# 绘制影响因素贡献占比图
plt.figure(figsize=(10, 6))
bars = importance.plot(kind='bar', color='skyblue')
plt.title('周长影响因素贡献占比', fontsize=16)
plt.xlabel('因素', fontsize=14)
plt.ylabel('贡献占比', fontsize=14)
plt.xticks(rotation=45)

# 显示每项因素的具体占比数值
for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height(), 
             f'{bar.get_height():.2%}',  
             ha='center', va='bottom')

plt.grid(axis='y')
plt.tight_layout()
plt.savefig('周长因素贡献占比.png')  # 保存图像
plt.show()

# 绘制预测值与实际值的比较图  
plt.figure(figsize=(10, 6))  
sns.scatterplot(x='周长', y='预测值', data=data, color='blue', alpha=0.6)  
plt.plot([data['周长'].min(), data['周长'].max()],   
         [data['周长'].min(), data['周长'].max()],   
         'r--', linewidth=2, label='y=x')  
plt.xlabel('实际值', fontsize=14)  
plt.ylabel('预测值', fontsize=14)  
plt.title('周长实际值与预测值比较', fontsize=16)  
plt.legend(title=f'MAE: {train_mae:.4f}\nR$^2$: {train_r2:.4f}')  
plt.grid(True)  
plt.tight_layout()  
plt.savefig('周长实际值与预测值比较.png')  # 保存图像
plt.show()  

# 绘制残差图
plt.figure(figsize=(10, 6))
sns.histplot(data['残差'], kde=True, color='purple', bins=30)
plt.xlabel('残差', fontsize=14)
plt.title('周长残差分布', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('周长残差分布.png')  # 保存图像
plt.show()

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(data['残差'], dist="norm", plot=plt)
plt.title('周长残差的Q-Q图', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('周长残差Q-Q图.png')  # 保存图像
plt.show()