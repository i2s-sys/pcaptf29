import time

from N_BaIoT.Resnet.Loss.pcapResnetPureSeed import TRAIN_FILE
from dA import dA, dA_params
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
TRAIN_FILE = '../train_data2_S.csv'
TEST_FILE = '../test_data2_S.csv'

def train_a_TF():
    AD = dA(dA_params(n_visible=73, n_hidden=7, lr=0.001, corruption_level=0.0, gracePeriod=0, hiddenRatio=None))
    #### ==== train  =====
    train_csv_path = TRAIN_FILE
    train_data = pd.read_csv(train_csv_path).values
    print('train shape:' + str(train_data.shape))
    for n in range(50):
        for i in range(train_data.shape[0]):
            AD.train(train_data[i])
    print('trained.')
    AD.save_model()

def run():
    AD = dA(dA_params(n_visible=43, n_hidden=7, lr=0.001, corruption_level=0.0, gracePeriod=0, hiddenRatio=None))
    #### ==== train  =====
    train_csv_path = TRAIN_FILE
    train_data = pd.read_csv(train_csv_path).values
    print('train shape:' + str(train_data.shape))
    for n in range(50):
        for i in range(train_data.shape[0]):
            AD.train(train_data[i])
    print('trained.')
    #########################################
    test_csv_path = TEST_FILE
    test_data = pd.read_csv(test_csv_path).values
    RMSE_0 = []
    begin = time.time()
    for i in range(test_data.shape[0]):
        rmse = AD.execute(test_data[i])
        RMSE_0.append(rmse)
    print('use time: ' + str(time.time() - begin))
    print("size = " + str(len(RMSE_0)))
    RMSE_0 = sorted(RMSE_0)
    f = open('RMSE_0.txt', 'w')
    np.savetxt(f, RMSE_0)
    f.close()
    ########################################
    # file = 'datas\\TF\\test-1.txt'
    # mat = np.loadtxt(file)
    # RMSE_1 = []
    # for i in range(mat.shape[0]):
    #     rmse = AD.execute(mat[i])
    #     RMSE_1.append(rmse)
    # RMSE_1 = sorted(RMSE_1)
    # f = open('RMSE_1.txt', 'w')
    # np.savetxt(f, RMSE_1)
    # f.close()

if __name__ == '__main__':
    run()
