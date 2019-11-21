import pandas as pd
import numpy as np
import sklearn.preprocessing as skl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

#initialization:
scaler = skl.MinMaxScaler(feature_range=(-1, 1), copy=False)

au_path = './AU/'
au_ri_path = './AU_reindex/'
val_gs_path = './emotional_behaviour/gold_standard/valence/'

index_name = 'time'
au_cols = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

au_csv = [f for f in listdir(au_path) if isfile(join(au_path, f))]
valence_csv = [f for f in listdir(val_gs_path) if isfile(join(val_gs_path, f))]


val_ewe = []
for csv_file in valence_csv:
    new_val_ewe = pd.read_csv(val_gs_path + csv_file, sep=',')
    val_ewe.append(new_val_ewe)
for x in val_ewe:
    x.index = x[index_name]
    x.drop([index_name], 1, inplace=True)

au = []
for csv_file in au_csv:
    new_au = pd.read_csv(au_path + csv_file, sep=',')
    new_au.columns = new_au.columns.str.replace(' ', '')
    au.append(new_au)
for x in au:
    x.index = x['timestamp']
    x.index.names = [index_name]
    x.drop(x.columns.difference(au_cols), 1, inplace=True)
    scaler.fit(np.array((0, 5)).reshape(-1, 1))
    for col in au_cols:
        scaler.transform(x[col].values.reshape(-1, 1))

def log(string):
    with open("./log.txt", "a") as text_file:
        text_file.write("{0}".format(string))

#output
i = 0
for i in range(len(au)):
    print('Reindexing action units for ' + au_csv[i])
    df1 = au[i]
    df2 = val_ewe[i]
    df1.index = df1.index.to_series().apply(lambda x: float('{0:.2f}'.format(round(x,2))))
    df3 = pd.DataFrame(0.0, index=df2.index, columns=au_cols)
            
    for n in range(len(df1)):
        df3.iloc[n] = df1.iloc[n]
    
    for n in range(len(df1), len(df3)):
        df3.iloc[n] = df1.iloc[len(df1) - 1]
    
    print('...done!')
    print('Saving as: ' + au_ri_path + valence_csv[i])
    df3.to_csv(au_ri_path + valence_csv[i])

