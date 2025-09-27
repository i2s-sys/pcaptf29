from Kitsune import Kitsune
import numpy as np
import time
import zipfile
# KitNET params:
# maxAE = 10 #maximum size for any autoencoder in the ensemble layer
# FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
# ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)

K = Kitsune()

print("Running Kitsune:")
# RMSEs = []
RMSEs = K.trainSAE()
# print(RMSEs)
K.train_classifier()

# from scipy.stats import norm
# benignSample = np.log(RMSEs[FMgrace+ADgrace+1:K.train_length])
# logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))
# print()
# plot the RMSE anomaly scores
# print("Plotting results")
# from matplotlib import pyplot as plt
# from matplotlib import cm
# plt.figure(figsize=(10,5))
# fig = plt.scatter(range(FMgrace+ADgrace+1,len(RMSEs)),RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
# plt.yscale("log")
# plt.title("Anomaly Scores from Kitsune's Execution Phase")
# plt.ylabel("RMSE (log scaled)")
# plt.xlabel("Time elapsed [min]")
# figbar=plt.colorbar()
# figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
# plt.show()
