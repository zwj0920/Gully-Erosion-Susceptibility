import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置文件路径
file_path = r"C:\Users\huang\Desktop\贡献.xlsx"

# 读取所有工作笔（时间规模）
excel_data = pd.ExcelFile(file_path)
sheet_names = excel_data.sheet_names

# 创建一个字典，用于存储每个时间规模的数据
data_dict = {sheet: excel_data.parse(sheet) for sheet in sheet_names}

# 假设各个工作笔结构相同，获取因素和参数的列表
factors = data_dict[sheet_names[0]].columns[1:]  # 假设第一列是参数名称
parameters = data_dict[sheet_names[0]].iloc[:, 0].unique()  # 提取参数名称

# RGB格式颜色设置，每个时间规模用不同颜色
colors = [(155/255, 46/255, 43/255), (226/255, 83/255, 61/255), (249/255, 228/255, 169/255)]  # 三种RGB颜色
alpha = 1  # 透明度设置为100%

# 创建图表
for param in parameters:
    plt.figure(figsize=(10, 8))  # 创建一个新图表

    # 获取每个时间规模的贡献数据
    contributions = {
        sheet: data_dict[sheet].loc[data_dict[sheet].iloc[:, 0] == param, factors].values.flatten()
        for sheet in sheet_names
    }

    # 将数据转换为DataFrame
    contributions_df = pd.DataFrame(contributions, index=factors)

    # 按照每个因素的贡献值从小到大排序
    sorted_contributions = {}
    for factor in factors:
        sorted_contributions[factor] = contributions_df.loc[factor].sort_values(ascending=True).values

    # 绘制每个时间规模的堆叠柱状图
    bottom = np.zeros(len(factors))  # 初始化底部位置
    for i, sheet in enumerate(sheet_names):
        # 获取该时间规模下的各因素贡献数据，并按贡献值从小到大排序
        values = [sorted_contributions[factor][i] for factor in factors]
        
        # 绘制堆叠柱状图，每个时间规模独立从 0 开始
        bars = plt.barh(factors, values, left=bottom, label=sheet, color=colors[i], edgecolor="black", alpha=alpha)
        bottom += values  # 更新底部位置

        # 在柱状图中添加具体数值并加上百分号（仅显示非零数值）
        for bar, value in zip(bars, values):
            if value > 0:  # 仅当数值不为零时才显示
                text_x = bar.get_x() + bar.get_width() / 2  # 数值位置居中
                text_y = bar.get_y() + bar.get_height() / 2  # 数值在柱体内垂直居中

                plt.text(
                    text_x, 
                    text_y,
                    f"{value:.1f}%",  # 显示数值并添加百分号
                    ha="center", va="center", fontname="Times New Roman"  # 去掉字号放大的设置
                )

    # 设置y轴标签和网格
    plt.ylabel("Factors", fontname="Times New Roman", fontsize=16)  # 改变字号为16
    plt.xlabel("Contribution Value (%)", fontname="Times New Roman", fontsize=16)  # 改变字号为16
    plt.title(f"{param} - Contribution of Factors Across Time Scales", fontname="Times New Roman", fontsize=18)  # 改变字号为18
    plt.grid(axis='x', linestyle='--', color='gray', alpha=0.7)  # 虚线网格

    # 动态设置x轴的范围，以确保显示所有堆叠部分
    max_contribution = bottom.max()  # 计算所有因素堆叠后的最大值
    plt.xlim(0, max_contribution * 1.1)  # 设置x轴范围为0到最大贡献值的110%，以确保显示完全

    # 设置x轴刻度
    plt.xticks(np.arange(0, max_contribution * 1.1, max_contribution * 0.2), 
               [f"{int(i)}%" for i in np.arange(0, max_contribution * 1.1, max_contribution * 0.2)], fontsize=14)  # 改变字号为14

    # 将图例移到右上角
    plt.legend(loc="upper right", title="Time Scale", fontsize=14, title_fontsize=16)  # 改变图例字号

    # 保存图表为高分辨率图片，指定 DPI
    output_path = f"C:/Users/huang/Desktop/{param}_contribution_chart.png"  # 自定义文件名
    plt.savefig(output_path, dpi=600, bbox_inches='tight')  # 设置 DPI 为 600，确保图例不被截断

    # 显示图表
    plt.tight_layout()
    plt.show()
