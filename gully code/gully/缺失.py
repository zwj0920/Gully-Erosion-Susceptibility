import pandas as pd

# 读取 Excel 文件
file_path = r'C:\Users\huang\Desktop\贡献.xlsx'  # 请替换为你的文件路径
df = pd.read_excel(file_path, sheet_name='14-21')  # 请替换为需要检查的工作表名称

# 打印前几行数据，查看数据结构
print(df.head())

# 检查是否有 NaN 值
nan_check = df.isna().sum()  # 每列的 NaN 数量
print("每列 NaN 值的数量：")
print(nan_check)

# 如果你想检查整个数据框是否有 NaN
print("\n是否存在 NaN 值：", df.isna().any().any())  # 如果有 NaN 返回 True

# 如果你只关心某一列的 NaN
print("\n'影响因素'列的 NaN 数量：", df['影响因素'].isna().sum())  # 请替换为你需要检查的列名
