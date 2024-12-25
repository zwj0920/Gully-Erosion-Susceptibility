import pandas as pd  
import statsmodels.api as sm  
from statsmodels.formula.api import mixedlm  
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_absolute_error, r2_score  
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 读取数据  
file_path = r"C:\Users\huang\Desktop\训练.xlsx"
data = pd.read_excel(file_path, sheet_name=3)  # 使用 sheet_name=3，即沟宽的数据

# 标准化数据
scaler = StandardScaler()
normalized_data = data.copy()
normalized_data[['ndvi', 'spi', 'twi', 'cm', 'aspect', 'slope', 'dem', 'artificial']] = scaler.fit_transform(
    data[['ndvi', 'spi', 'twi', 'cm', 'aspect', 'slope', 'dem', 'artificial']]
)

# 定义模型公式  
formula = '沟宽 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial'  

# 构建GLMM模型，使用id和time作为随机效应  
model = mixedlm(formula, normalized_data, groups=normalized_data["id"], re_formula="~time")  
result = model.fit()  

# 查看结果  
print(result.summary())  

# 添加预测值和残差到数据框  
normalized_data['Predicted'] = result.fittedvalues  
normalized_data['Residuals'] = result.resid  

# 计算MAE和R²  
train_mae = mean_absolute_error(normalized_data['沟宽'], normalized_data['Predicted'])  
train_r2 = r2_score(normalized_data['沟宽'], normalized_data['Predicted'])  
print(f"Training set MAE: {train_mae:.4f}")  
print(f"Training set R²: {train_r2:.4f}")  

# 绘制预测与实际值的比较图并添加正比例函数线
plt.figure(figsize=(10, 6))
# 将 RGB 值转换为 0 到 1 的范围
color_rgb = (1/255, 117/255, 109/255)
sns.scatterplot(x='沟宽', y='Predicted', data=normalized_data, color=color_rgb, alpha=0.6)

# 绘制 y=x 线，使用黑色
plt.plot([0, normalized_data['沟宽'].max()],   
         [0, normalized_data['沟宽'].max()],   
         'k--', linewidth=2, label='y=x')  

plt.xlabel('Actual', fontsize=14)  
plt.ylabel('Predicted', fontsize=14)  
plt.title('Comparison of Actual and Predicted Gully Depth', fontsize=16)  
plt.legend()  # 显示 y=x 的图例
plt.xlim(0, normalized_data['沟宽'].max())  # 设置 x 轴的范围从 0 到最大值
plt.ylim(0, normalized_data['沟宽'].max())  # 设置 y 轴的范围从 0 到最大值
plt.tight_layout()  
plt.savefig('Comparison_of_Actual_and_Predicted.png', dpi=300)  # 保存图像
plt.show()