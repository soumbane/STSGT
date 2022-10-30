import numpy as np
import matplotlib.pyplot as plt

#plt.style.use('fivethirtyeight')

plt.style.use('seaborn-white')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 15
plt.rcParams["font.weight"] = "bold"
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 30


y_real = np.load('y_real.npy')
#print(y_real)

#y_pred_ARIMA = np.load('y_pred_ARIMA.npy')

y_pred_STGCN = np.load('y_pred_STGCN.npy')

y_pred_ASTGCN = np.load('y_pred_ASTGCN.npy')

y_pred_GWNet = np.load('y_pred_GWNet.npy')

y_pred_STTN = np.load('y_pred_STTN.npy')

y_pred_STST = np.load('y_pred_STST.npy')
#print(y_pred_STST)

# plot all the predictions vs dates

fig = plt.figure(figsize=(15,9))
plt.plot(y_real, c='b', linewidth=3.0, marker='D', mfc='blue', mec='black', ms='10')
#plt.plot(y_pred_ARIMA, c='y', linewidth=3.0, marker='D', mfc='yellow', mec='black', ms='10')
plt.plot(y_pred_STGCN, c='r', linewidth=3.0, marker='D', mfc='red', mec='black', ms='10')
plt.plot(y_pred_ASTGCN, c='m', linewidth=3.0, marker='D', mfc='magenta', mec='black', ms='10')
plt.plot(y_pred_GWNet, c='g', linewidth=3.0, marker='D', mfc='green', mec='black', ms='10')
plt.plot(y_pred_STTN, c='y', linewidth=3.0, marker='D', mfc='yellow', mec='black', ms='10')
plt.plot(y_pred_STST, c='c', linewidth=3.0, marker='D', mfc='cyan', mec='black', ms='10')

#plt.title('Infected Cases for Wayne County, Michigan', fontname="Times New Roman", size=20, fontweight="bold")
#plt.xlabel('Date -->')
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 14, step=1))  # Set label locations.
plt.xticks(np.arange(15), ['10/4', '10/6', '10/8', '10/11', '10/13', '10/15', '10/18', '10/20', '10/22', '10/25', '10/27', '10/29', '11/1', '11/3', '11/5'])  # Set text labels.

#plt.ylabel('Number of Infected Cases -->')
plt.axis([-0.1,14.1,200,1600])
#plt.yscale('log')   
#plt.legend(['Ground Truth','ARIMA', 'STGCN','ASTGCN','GWNet','STTN', 'STSGT'], bbox_to_anchor=(1.02,1), loc='upper left', borderaxespad=0, ncol=2)

plt.legend(['Ground Truth', 'STGCN','ASTGCN(r)','Graph WaveNet','STTN', 'STSGT'], loc='upper left', ncol=2)
plt.show()


