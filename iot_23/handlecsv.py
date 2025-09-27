import csv

from sklearn.model_selection import train_test_split


def get_col_types():
    protocol_type = ['icmp', 'tcp', 'udp']
    service_type = ['-','dhcp','dns','http','ssh']
    conn_state_type = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR','RSTRH','S0', 'S1', 'S2', 'SF', 'SH']
    history_type = ['F', 'Fa', 'Dd', 'ShAfFa', 'ShAdfFa', 'ShAdDafF', 'ShAdDaFf', 'ShAdDaTFf', 'ShAdDtafF', 'ShAdDaFRf', 'ShADadfF', 'ShADFadfRR', 'ShADafF', 'ShAdDatFf', 'ShAdDatfF', 'ShADdfFa', 'ShAdDfFr', 'ShAdDafFr', 'ShAFf', 'ShAdDafFrR', 'ShADadFf', 'ShAdDaTfF', 'ShAdDTafF', 'ShAdDaTFfR', 'ShAdDaftF', 'ShAdFaf', 'ShAFfR', 'ShAdDafrFr', 'ShADfFr', 'ShADfrFr', 'ShADafdtF', 'ShAdDaFfr', 'HaDdAfF', 'ShAdDaFfR', 'HaFfA', 'ShAdDafFR', 'ShAdDaFRfR', 'ShAdDaFTf', 'FfA', 'HaDdAFf', 'ShAFafR', 'HaDdAFTf', 'ShAdDaftFR', 'ShADfaF', '^hADafF', 'ShADfFa', 'ShAdDFfR', 'ShAfdtDFr', 'ShAdfDFr', 'ShADfFrr', 'ShAdDfFa', 'ShAadDfF', 'ShADfdtFaR', 'ShAFdRfR', 'ShAdDafFrr', 'Ffa', 'ShAdtDafF', 'ShAdDaFRRf', 'ShAdDtafFr', 'HaADdFf', 'HaDdTAFf', 'HafFr', 'ShAFdfRt', 'ShADfrF', 'ShADfF', 'ShAF', 'ShADF', 'ShAdDaTF', 'ShADaF', 'ShAFa', 'ShAdDaF', 'ShAdF', 'ShA', '^hA', 'S', 'D', 'D^', '^r', 'Ar', 'ShAdaFr', 'ShADr', 'ShADFr', 'ShAr', 'ShADafr', 'ShADarfF', 'ShAFr', 'ShAdDaFr', 'ShAdtDaFrR', 'ShAdDatFr', 'ShAdtDaFr', 'ShAdDatFrR', 'ShAdDaFrR', 'ShADrfR', 'ShAdDafrR', 'ShAdDarfR', 'SahAdDrfR', 'Fr', 'ShAdDatrfR', 'ShADar', 'ShAdDr', 'ShrA', 'ShAdr', 'ShAdDtaFr', 'ShAdDFar', 'ShAdDarr', 'FaAr', 'ShAdDar', 'ShAdDafr', '^hADFr', '^hADr', 'R', 'SaR', 'HaDdR', 'HaR', '^aR', 'FaR', 'ShADadfR', 'ShADadRf', 'ShADadR', 'ShADdafR', 'ShADfdtR', 'ShADdfR', 'ShADfR', '^hADadfR', 'Sr', '-1']
    label_type = ['(empty)   Benign   -1',
                  '(empty)   Malicious   PartOfAHorizontalPortScan']
    return protocol_type,service_type,conn_state_type,history_type,label_type
def handle_data():
    protocol_type,service_type,conn_state_type,history_type,label_type = get_col_types()
    with open('conn_log_labeled2.csv', 'r') as data_source:
        csv_reader = csv.reader(data_source)
        header = next(csv_reader)
        data = [row for row in csv_reader]
    train_data, test_data = train_test_split(data, test_size=0.3)
    with open('origin_train_data2.csv', 'w', newline='') as train_file:
        csv_writer = csv.writer(train_file)
        csv_writer.writerow(header)
        csv_writer.writerows(train_data)
    # 写入测试数据
    with open('origin_test_data2.csv', 'w', newline='') as test_file:
        csv_writer = csv.writer(test_file)
        csv_writer.writerow(header)
        csv_writer.writerows(test_data)
    # 从测试集开始
    source_file = 'origin_train_data2.csv'
    handled_file = 'train_data2.csv'  # write to csv file
    data_file = open(handled_file, 'w', newline='')
    csv_writer = csv.writer(data_file)
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        next(csv_reader)
        for row in csv_reader:
            row[0] = protocol_type.index(row[0])
            row[1] = service_type.index(row[1])
            row[4] = conn_state_type.index(row[4])
            row[5] = history_type.index(row[5])
            row[-1] = label_type.index(row[-1])
            csv_writer.writerow(row)

        data_file.close()
    test_source_file = 'origin_test_data2.csv'
    test_handled_file = 'test_data2.csv'  # write to csv file
    test_data_file = open(test_handled_file, 'w', newline='')
    test_csv_writer = csv.writer(test_data_file)
    with open(test_source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        next(csv_reader)
        for row in csv_reader:
            row[0] = protocol_type.index(row[0])
            row[1] = service_type.index(row[1])
            row[4] = conn_state_type.index(row[4])
            row[5] = history_type.index(row[5])
            row[-1] = label_type.index(row[-1])
            test_csv_writer.writerow(row)
        test_data_file.close()
    print('pre process completed!')
handle_data()