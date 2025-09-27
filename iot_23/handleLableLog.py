import pandas as pd

def read_labeled_log(file_path, output_csv):
    with open(file_path, 'r') as file:
        # 读取文件的所有行
        lines = file.readlines()

    # 查找 #fields 行
    fields_line = [line for line in lines if line.startswith('#fields')][0]

    # 提取列名
    columns = fields_line.strip().split('\t')[1:]

    # 提取数据行，忽略以 # 开头的行
    data_lines = [line for line in lines if not line.startswith('#')]

    # 读取数据到DataFrame
    df = pd.DataFrame([line.strip().split('\t') for line in data_lines], columns=columns)

    # 保存为CSV文件，不写入表头
    df.to_csv(output_csv, index=False, header=True)

    return df

# 示例：读取 conn.log.labeled 文件并保存为 conn_log_labeled.csv
file_path = 'conn.log3.labeled'
output_csv = 'conn_log3_labeled.csv'
df = read_labeled_log(file_path, output_csv)

# 打印前几行数据
print(df.head())
