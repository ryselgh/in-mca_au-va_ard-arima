import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from os import listdir
from os.path import isfile, join

#initialization:
valence_path = './emotional_behaviour/valence/'
arousal_path = './emotional_behaviour/arousal/'
val_gs_path = './emotional_behaviour/gold_standard/valence/'
aro_gs_path = './emotional_behaviour/gold_standard/arousal/'
index_name = 'time'

va_cols = ['FM1','FM2','FM3','FF1','FF2','FF3']

valence_csv = [f for f in listdir(valence_path) if isfile(join(valence_path, f))]
arousal_csv = [f for f in listdir(arousal_path) if isfile(join(arousal_path, f))]

valence = []
for csv_file in valence_csv:
    new_valence = pd.read_csv(valence_path + csv_file, sep=';')
    new_valence.columns = new_valence.columns.str.replace(' ', '')
    valence.append(new_valence)
for x in valence:
    x.index = x[index_name]
    x.drop(x.columns.difference(va_cols), 1, inplace=True)

arousal = []
for csv_file in arousal_csv:
    new_arousal = pd.read_csv(arousal_path + csv_file, sep=';')
    new_arousal.columns = new_arousal.columns.str.replace(' ', '')
    arousal.append(new_arousal)
for x in arousal:
    x.index = x[index_name]
    x.drop(x.columns.difference(va_cols), 1, inplace=True)
    
   
def get_gold_std(df):
    
    df_len = len(df)
    df_col_len = len(df.columns)
    
    #μₖ
    mu = []
    for col in df.columns:
        mu.append(df[col].sum())
    mu[:] = [k / df_len for k in mu]
    
    #x̄ₙᴹᴸᴱ
    ann_mean = []
    for index, row in df.iterrows():
        ann_sum = 0
        for col in df.columns:
            ann_sum += row[col]
        ann_mean.append(ann_sum / df_col_len)
    
    #μᴹᴸᴱ
    mu_mle = sum(ann_mean) / df_len
    
    #rₖ
    r = []
    for k in range(len(df.columns)):
        sum_up = 0
        sum_dnl = 0
        sum_dnr = 0
        n = 0
        for index, row in df.iterrows():
            x = row[va_cols[k]]
            sum_up += (x - mu[k]) * (ann_mean[n] - mu_mle)
            sum_dnl += ((x - mu[k])) ** 2
            sum_dnr += ((ann_mean[n] - mu_mle)) ** 2
            n += 1
        sum_dnl = sum_dnl ** 0.5
        sum_dnr = sum_dnr ** 0.5
        rk = (sum_up) / (sum_dnl * sum_dnr)
        r.append(rk)
    #gold standard
    key = 'gold standard'
    gs = pd.DataFrame(0.0, index=df.index, columns=[key])
    r_sum = sum(r)
    for n in range(df_len):
        for k in range(df_col_len):
            gs.iloc[n][key] += df.iloc[n][va_cols[k]] * r[k]
        gs.iloc[n][key] /= r_sum
    
    return gs

#output
for i in range(len(valence)):
    print('Computing valence Gold Standard for ' + valence_csv[i])
    val_gs = get_gold_std(valence[i])
    print(val_gs)
    print('...done!')
    print('Saving as: ' + val_gs_path + valence_csv[i])
    val_gs.to_csv(val_gs_path + valence_csv[i])
    print('...done!')
    
for i in range(len(arousal)):
    print('Computing valence Gold Standard for ' + arousal_csv[i])
    aro_gs = get_gold_std(arousal[i])
    print(aro_gs)
    print('...done!')
    print('Saving as: ' + aro_gs_path + arousal_csv[i])
    aro_gs.to_csv(aro_gs_path + arousal_csv[i])
    print('...done!')
