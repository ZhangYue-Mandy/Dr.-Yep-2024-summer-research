import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_cluster(file_path, sheet_name, cluster):
    # 读取Excel文件中特定的工作表
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="D,F", skiprows=1, skipfooter=1)

    # 删除空白行
    df.dropna(how='all', inplace=True)

    # 将第一行非空白数据作为标题
    df.columns = df.iloc[1]
    df = df[2:]

    # 重置索引
    df.reset_index(drop=True, inplace=True)

    # 提取类别前缀并添加新列
    df['Category'] = df['Object'].str.extract(r'(CG\d+)', expand=False)

    # 将不符合特定前缀的行归为 'Others'
    df['Category'].fillna('Others', inplace=True)

    # 提取 CG 前缀对象名称中的数字部分，否则保留原来的字符
    df['Simplified Object'] = df.apply(
        lambda row: row['Object'].split('_')[-1] if row['Category'] != 'Others' else row['Object'], axis=1
    )

    # 筛选特定类别的数据
    cluster_data = df[df['Category'] == cluster]

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(cluster_data['Simplified Object'], cluster_data['V'], label=cluster)

    plt.xlabel('Object')
    plt.ylabel('V')
    plt.title(f'Object vs V for {cluster}')
    plt.legend(title='Category')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # 显示图表
    plt.show()

# 使用示例
source_folder=os.getcwd()
file_path = os.path.join(source_folder, 'Goodman_Observing_Log.xlsx')
sheet_name = 'summary'
cluster = 'CG14'
plot_cluster(file_path, sheet_name, cluster)
