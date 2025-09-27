# python脚本，把test_data.csv文件的最后一列进行修改，如果是标签0，保持不变，如果是其他，改成1

import csv


def modify_last_column(input_file, output_file):
    """
    修改CSV文件的最后一列：
    - 如果最后一列的值是 '0'，保持不变。
    - 如果最后一列的值是其他值，改为 '1'。
    """
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)

        # 逐行读取并处理
        for row in csv_reader:
            # 检查最后一列的值
            if row[-1] != '0':
                row[-1] = '1'  # 如果不是 '0'，则改为 '1'
            # 写入新文件
            csv_writer.writerow(row)

    print(f"处理完成，结果已保存到 {output_file}")


# 输入文件和输出文件的路径
input_file = 'test_data.csv'
output_file = 'test_data_modified.csv'

# 调用函数处理文件
modify_last_column(input_file, output_file)