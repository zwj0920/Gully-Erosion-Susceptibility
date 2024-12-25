import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np  
from statsmodels.formula.api import mixedlm  

# 读取数据  
file_path = r"C:\Users\huang\Desktop\训练.xlsx"  
data = pd.read_excel(file_path, sheet_name=3)  

# 定义模型公式并计算各个残差  
formulas = {
    '周长残差': '周长 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial',
    '面积残差': '面积 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial',
    '沟长残差': '沟长 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial',
    '沟宽残差': '沟宽 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial',
    '沟深残差': '沟深 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial',
    '体积残差': '体积 ~ ndvi + spi + twi + cm + aspect + slope + dem + artificial'
}

residuals = []
for name, formula in formulas.items():
    model = mixedlm(formula, data, groups=data["id"]).fit()
    data[name] = data[name.split('残差')[0]] - model.fittedvalues
    residuals.append(data[name])

# 合并所有残差数据到一个数组中
all_residuals = np.concatenate(residuals)

# 极坐标直方图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 设置每个数据集的角度偏移
num_datasets = len(residuals)
angles = np.linspace(0, 2 * np.pi, num_datasets, endpoint=False)

# 绘制每个残差的直方图
for i, res in enumerate(residuals):
    angle = angles[i]
    ax.hist(res, bins=30, alpha=0.6, color=plt.cm.viridis(i / num_datasets), orientation='horizontal')

# 设置theta方向的标签为空
ax.set_xticks([])

# 设置标题
plt.title('合并残差的极坐标直方图', fontsize=16)

plt.show()