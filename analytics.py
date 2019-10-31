from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from os import listdir
from os.path import isfile, join

#initialization:
mpl.rcParams['figure.dpi'] = 300

valence_path = './emotional_behaviour/valence/'
arousal_path = './emotional_behaviour/arousal/'
au_path = './AU/'

valence_csv = [f for f in listdir(valence_path) if isfile(join(valence_path, f))]
arousal_csv = [f for f in listdir(arousal_path) if isfile(join(arousal_path, f))]
au_csv = [f for f in listdir(au_path) if isfile(join(au_path, f))]

valence = []
for csv_file in valence_csv:
    valence.append(pd.read_csv(valence_path + csv_file, sep=';'))

arousal = []
for csv_file in arousal_csv:
    arousal.append(pd.read_csv(arousal_path + csv_file, sep=';'))
    
au = []
for csv_file in au_csv:
    au.append(pd.read_csv(au_path + csv_file, sep=','))

#functions:
def plot_annemo(data, xcol, index, label, ylabel):
    plt.plot(
        data[index][xcol], data[index]['FM1 '],
        data[index][xcol], data[index]['FM2 '],
        data[index][xcol], data[index]['FM3 '],
        data[index][xcol], data[index]['FF1 '],
        data[index][xcol], data[index]['FF2 '],
        data[index][xcol], data[index]['FF3'])
    plt.title(label[index].replace('.csv',''))
    plt.ylabel(ylabel)
    plt.xlabel(xcol)
    plt.legend(['FM1','FM2','FM3','FF1','FF2','FF3'], fontsize=6)
    plt.show()
    
def plot_au(data, xcol, index, label, ylabel):
    plt.plot(
        data[index][xcol], data[index][' AU01_r'],
        data[index][xcol], data[index][' AU02_r'],
        data[index][xcol], data[index][' AU04_r'],
        data[index][xcol], data[index][' AU05_r'],
        data[index][xcol], data[index][' AU06_r'],
        data[index][xcol], data[index][' AU07_r'],
        data[index][xcol], data[index][' AU09_r'],
        data[index][xcol], data[index][' AU10_r'],
        data[index][xcol], data[index][' AU12_r'],
        data[index][xcol], data[index][' AU14_r'],
        data[index][xcol], data[index][' AU15_r'],
        data[index][xcol], data[index][' AU17_r'],
        data[index][xcol], data[index][' AU20_r'],
        data[index][xcol], data[index][' AU23_r'],
        data[index][xcol], data[index][' AU25_r'],
        data[index][xcol], data[index][' AU26_r'],
        data[index][xcol], data[index][' AU45_r'])
    plt.title(label[index].replace('.txt',''))
    plt.ylabel(ylabel)
    plt.xlabel(xcol)
    plt.legend(['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'], fontsize=3)
    plt.show()

#data analysis:
plot_annemo(valence, 'time', 0, valence_csv, 'valence')
plot_annemo(arousal, 'time', 0, arousal_csv, 'arousal')
plot_au(au, ' timestamp', 0, au_csv, 'AU')