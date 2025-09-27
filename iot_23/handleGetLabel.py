import pandas as pd


def extract_second_column_elements(file_path):
    data = pd.read_csv(file_path)

    second_column_elements = data.iloc[:, 0].unique()

    print(list(second_column_elements))

file_path = '12.csv'
extract_second_column_elements(file_path)
