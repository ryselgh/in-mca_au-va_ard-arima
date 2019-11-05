from scipy import stats
import numpy as np
import sklearn.preprocessing as skl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from os import listdir
from os.path import isfile, join

#initialization:
mpl.rcParams['figure.dpi'] = 300
scaler = skl.MinMaxScaler(feature_range=(-1, 1), copy=False)

valence_path = './emotional_behaviour/valence/'
arousal_path = './emotional_behaviour/arousal/'
au_path = './AU/'

va_cols = ['FM1','FM2','FM3','FF1','FF2','FF3']
au_cols = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

valence_csv = [f for f in listdir(valence_path) if isfile(join(valence_path, f))]
arousal_csv = [f for f in listdir(arousal_path) if isfile(join(arousal_path, f))]
au_csv = [f for f in listdir(au_path) if isfile(join(au_path, f))]

valence = []
for csv_file in valence_csv:
    new_valence = pd.read_csv(valence_path + csv_file, sep=';')
    new_valence.columns = new_valence.columns.str.replace(' ', '')
    valence.append(new_valence)

arousal = []
for csv_file in arousal_csv:
    new_arousal = pd.read_csv(arousal_path + csv_file, sep=';')
    new_arousal.columns = new_arousal.columns.str.replace(' ', '')
    arousal.append(new_arousal)
    
au = []
for csv_file in au_csv:
    new_au = pd.read_csv(au_path + csv_file, sep=',')
    new_au.columns = new_au.columns.str.replace(' ', '')
    au.append(new_au)
    
for au_data in au:
    for col in au_cols:
        scaler.fit_transform(au_data[col].values.reshape(-1, 1))

#functions:
def plot_annemo(data, xcol, index, label, ylabel):
    plt.plot(
        data[index][xcol], data[index][va_cols[0]],
        data[index][xcol], data[index][va_cols[1]],
        data[index][xcol], data[index][va_cols[2]],
        data[index][xcol], data[index][va_cols[3]],
        data[index][xcol], data[index][va_cols[4]],
        data[index][xcol], data[index][va_cols[5]])
    plt.title(label[index].replace('.csv',''))
    plt.ylabel(ylabel)
    plt.xlabel(xcol)
    plt.legend(va_cols, fontsize=6)
    plt.show()
    
def plot_au(data, xcol, index, label, ylabel):
    plt.plot(
        data[index][xcol], data[index][au_cols[0]],
        data[index][xcol], data[index][au_cols[1]],
        data[index][xcol], data[index][au_cols[2]],
        data[index][xcol], data[index][au_cols[3]],
        data[index][xcol], data[index][au_cols[4]],
        data[index][xcol], data[index][au_cols[5]],
        data[index][xcol], data[index][au_cols[6]],
        data[index][xcol], data[index][au_cols[7]],
        data[index][xcol], data[index][au_cols[8]],
        data[index][xcol], data[index][au_cols[9]],
        data[index][xcol], data[index][au_cols[10]],
        data[index][xcol], data[index][au_cols[11]],
        data[index][xcol], data[index][au_cols[12]],
        data[index][xcol], data[index][au_cols[13]],
        data[index][xcol], data[index][au_cols[14]],
        data[index][xcol], data[index][au_cols[15]],
        data[index][xcol], data[index][au_cols[16]])
    plt.title(label[index].replace('.txt',''))
    plt.ylabel(ylabel)
    plt.xlabel('time')
    plt.legend(au_cols, fontsize=3)
    plt.show()

#data analysis:
plt.show()
plot_annemo(valence, 'time', 0, valence_csv, 'valence')
plot_annemo(arousal, 'time', 0, arousal_csv, 'arousal')
plot_au(au, 'timestamp', 0, au_csv, 'AU')