import re
import warnings
import time
from os import listdir
from os.path import isfile, join
from scipy import stats
import numpy as np
import sklearn.preprocessing as skl
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#initialization:
start = time.time()
register_matplotlib_converters()
warnings.filterwarnings("ignore")
mpl.rcParams['figure.dpi'] = 1200
scaler = skl.MinMaxScaler(feature_range=(-1, 1), copy=False)

#constant strings definition:
valence_path = './emotional_behaviour/valence/'
arousal_path = './emotional_behaviour/arousal/'
val_gs_path = './emotional_behaviour/gold_standard/valence/'
aro_gs_path = './emotional_behaviour/gold_standard/arousal/'
au_path = './AU_reindex/'
index_name = 'time'
gs_key = ['gold standard']
va_cols = ['FM1','FM2','FM3','FF1','FF2','FF3']
au_cols = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
valence_csv = [f for f in listdir(valence_path) if isfile(join(valence_path, f))]
arousal_csv = [f for f in listdir(arousal_path) if isfile(join(arousal_path, f))]
au_csv = [f for f in listdir(au_path) if isfile(join(au_path, f))]

#define data arrays:
valence = []
for csv_file in valence_csv:
    new_valence = pd.read_csv(valence_path + csv_file, sep=';')
    new_valence.columns = new_valence.columns.str.replace(' ', '')
    valence.append(new_valence)
for x in valence:
    x.index = x[index_name]
    x.drop(x.columns.difference(va_cols), 1, inplace=True)
    
val_ewe = []
for csv_file in valence_csv:
    new_val_ewe = pd.read_csv(val_gs_path + csv_file, sep=',')
    val_ewe.append(new_val_ewe)    
for x in val_ewe:
    x.index = x[index_name]
    x.drop([index_name], 1, inplace=True)
    
val_train = pd.DataFrame(columns=gs_key)
for i in range(12):
    temp_df = pd.DataFrame(val_ewe[i])
    temp_df.index = temp_df.index + 300*i
    val_train = val_train.append(temp_df)
val_valid = pd.DataFrame(columns=gs_key)
for i in range(12, 14):
    temp_df = pd.DataFrame(val_ewe[i])
    temp_df.index = temp_df.index + 300*i
    val_valid = val_valid.append(temp_df)

arousal = []
for csv_file in arousal_csv:
    new_arousal = pd.read_csv(arousal_path + csv_file, sep=';')
    new_arousal.columns = new_arousal.columns.str.replace(' ', '')
    arousal.append(new_arousal)
for x in arousal:
    x.index = x[index_name]
    x.drop(x.columns.difference(va_cols), 1, inplace=True)
    
aro_ewe = []
for csv_file in arousal_csv:
    new_aro_ewe = pd.read_csv(aro_gs_path + csv_file, sep=',')
    aro_ewe.append(new_aro_ewe)
for x in aro_ewe:
    x.index = x[index_name]
    x.drop([index_name], 1, inplace=True)
    
aro_train = pd.DataFrame(columns=gs_key)
for i in range(12):
    temp_df = pd.DataFrame(aro_ewe[i])
    temp_df.index = temp_df.index + 300*i
    aro_train = aro_train.append(temp_df)
aro_valid = pd.DataFrame(columns=gs_key)
for i in range(12, 14):
    temp_df = pd.DataFrame(aro_ewe[i])
    temp_df.index = temp_df.index + 300*i
    aro_valid = aro_valid.append(temp_df)
    
au = []
for csv_file in au_csv:
    new_au = pd.read_csv(au_path + csv_file, sep=',')
    new_au.columns = new_au.columns.str.replace(' ', '')
    au.append(new_au)
for x in au:
    x.index = x[index_name]
    x.index.names = [index_name]
    x.drop(x.columns.difference(au_cols), 1, inplace=True)

au_train = pd.DataFrame(columns=au_cols)
for i in range(12):
    temp_au_df = pd.DataFrame(au[i])
    temp_au_df.index = temp_au_df.index + 300*i
    au_train = au_train.append(temp_au_df)
    
au_valid = pd.DataFrame(columns=au_cols)
for i in range(12, 14):
    temp_au_df = pd.DataFrame(au[i])
    temp_au_df.index = temp_au_df.index + 300*i
    au_valid = au_valid.append(temp_au_df)


#functions:
def plot_data(data, ylabel, title, show_ewe=False, va_ewe=None, fs=8):
    for col in data.columns:
        plt.plot(data[col], label=col)
    title = re.sub(r'(.txt|.csv)', '', title)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(data.index.names[0])
    
    if show_ewe:
        plt.plot(va_ewe, color='black', linewidth=2, label="EWE")
        
    plt.legend(loc='best', fontsize=fs)
    plt.show()

def plot_ewe():
    for i in range(len(valence)):
        plot_data(valence[i], 'valence', valence_csv[i] + ': Evaluator Weighted Estimator', show_ewe=True, va_ewe=val_ewe[i])
    for i in range(len(arousal)):
        plot_data(arousal[i], 'arousal', arousal_csv[i] + ': Evaluator Weighted Estimator', show_ewe=True, va_ewe=aro_ewe[i])



# Dickey–Fuller test for stationarity of a time series
def get_stationarity(data, win, ylabel, can_plot=True):
    
    # rolling statistics
    rolling_mean = data.rolling(window = win).mean()
    rolling_std = data.rolling(window = win).std()
    
    # plot
    if can_plot:
        plt.plot(data, color = 'lightblue', label = 'Original')
        plt.plot(rolling_mean, color = 'orange', label = 'Rolling Mean')
        plt.plot(rolling_std, color = 'darkred', label = 'Rolling Std')
        plt.legend(loc = 'best')
        plt.title('Rolling Mean & Rolling Standard Deviation')
        plt.ylabel(ylabel + ' (gold std)')
        plt.xlabel(index_name)
        plt.show()

    # Dickey–Fuller test:
    result = adfuller(data.iloc[:,0].values)
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    print('')
    
def get_aug_stationarity(data, significance=0.05):
    
    p_value = None
    is_stationary = None

    #Dickey-Fuller test:
    adf_test = adfuller(data.iloc[:,0].values, autolag='AIC')
    print(adf_test)
    p_value = adf_test[1]
    
    if (p_value < significance):
        is_stationary = True
    else:
        is_stationary = False
    
    results = pd.Series(adf_test[0:4], index=['ADF Test Statistic','P-Value',
                                              '# Lags Used','# Observations Used'])

    #Add Critical Values
    for key,value in adf_test[4].items():
        results['Critical Value (%s)'%key] = value

    print('Augmented Dickey-Fuller Test Results:')
    print(results)
    print('Is the time series stationary? {0}'.format(is_stationary))
    
    #return adf_test[2]
    
def get_all_stationarity(win, can_plot=True):
    for i in range(len(val_ewe)):
        print('---- File: ' + valence_csv[i] + ' ----' )
        get_stationarity(val_ewe[i], win, 'valence', can_plot)
        get_stationarity(aro_ewe[i], win, 'arousal', can_plot)
        print('')

def plot_ACF(df, custom_title=''):
    minus_shift = df - df.shift()
    minus_shift.dropna(inplace=True)
    df = minus_shift
    custom_title = re.sub(r'(.txt|.csv)', '', custom_title)
    plot_acf(df, ax=plt.gca(), lags=range(50), title=custom_title+
             ' Autocorrelation')
    plt.show()
    
def plot_PACF(df, custom_title=''):
    minus_shift = df - df.shift()
    minus_shift.dropna(inplace=True)
    df = minus_shift
    custom_title = re.sub(r'(.txt|.csv)', '', custom_title)
    plot_pacf(df, ax=plt.gca(), lags=range(50), title=custom_title+
              'Partial Autocorrelation')
    plt.show()

def plot_ACF_PACF(df):
    plot_ACF(df)
    plot_PACF(df)


#####################################
#       ARIMA MODEL HYPERPARAMETERS #
#####################################   
"return p and estimated coefficents for the ARIMA MODEL in the train"
def auto_ARIMA_train(train_data, ylabel, moving_average=0):
    lag = get_aug_stationarity(train_data)
    plot_PACF(train_data)
    #reasonable values of p obtained trying PACF on training series
    p_array = [3,4,5,6,7,8,9]
    
    aics = []
    aic_min = float("inf")
    for x in range(len(p_array)):    
            
        print('Applying ARIMA with order=({0},1,{1})'.format(p_array[x], 
                                                             moving_average))
        aic = apply_ARIMA(train_data, ylabel, p=p_array[x], d=1, 
                          q=moving_average, exogenous=au_train.values, 
                          plot=False)
        aics.append(aic) 
        print('AIC value')
        print(aic)
        if aic < aic_min:
            aic_min = aic
            bestp = p_array[x]
    #Use Akaike Information Criterion (min AIC) to find the best model among the estimated
    print("AICS values")
    print(aics)
    print('Min AICs for p = {0}'.format(p_array))
    print('Minimum AIC value:')
    print(aic_min)
    print('best p:')
    print(bestp)
    return bestp


###########################
#       ARIMA MODEL TEST  #
###########################
ans = True
while ans == True:
    print('\nARIMA')
    print('\nChoose target:')
    print (" 1) Valence\n 2) Arousal")
    ans = input("Choice: ") 
    if ans == "1":
        train = val_train
        valid = val_valid
        label = 'valence'
    elif ans == "2":
        train = aro_train
        valid = aro_valid
        label = 'arousal'
    else:
        ans = True
        print("Invalid input.")

#To find order(p,q,d) with PACF, AIC and ADF test:       
#best_p = auto_ARIMA_train(train, label)

#run with found results  searching for best_p
alpha = 0.05 # 95% confidence
best_p = 9 #estimated with AIC and PACF, d=1 for approx stationarity
order = (best_p, 1, 0)

# Build Model
model = ARIMA(train, order=order, exog=au_train.values)
fitted = model.fit(disp=-1)
print(fitted.summary())




#SHOW model parameter estimation----------------------------------------------
coeff = fitted.params
au_coeff = coeff[1:18]
au_coeff.index = au_cols
ar_coeff = coeff[18:]

#show sorted action units weights
au_sort= au_coeff.abs().sort_values(ascending=False)
au_sort.index
au_coeff_sort = au_coeff[au_sort.index]


print('Lags coefficent values:\n{0}'.format(ar_coeff))
print('Action units coefficent values:\n{0}'.format(au_coeff_sort))
#plot weights bar in log scale
cols_plot = ['1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', 
             '20', '23', '25', '26', '45']
for i in range(1,best_p+1):
    cols_plot.append('L{0}'.format(i))
    
#exclude const coeff[0]
coeff_plot = coeff[1:]
coeff_plot.index = cols_plot
plt.figure(figsize=(7,5))
plt.title("ARIMA model absolute value of weights - Log scale")
plt.bar(coeff_plot.index,np.absolute(coeff_plot.values),log=True)
plt.xlabel("Action units + Lags")
plt.ylabel("Values of the weights")
cols_plot = ['1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', 
             '20', '23', '25', '26', '45']
au_coeff.index = cols_plot
plt.figure(figsize=(7,5))
plt.title("Action unit weights")
plt.bar(au_coeff.index,au_coeff.values)
plt.xlabel("Action units")
plt.ylabel("Values of the weights")




#forecast the validation set---------------------------------------------------

fc, se, conf = fitted.forecast(au_valid.shape[0], exog=au_valid.values, 
                               alpha=alpha)
    
# Make as pandas series
fc_series = pd.Series(fc, index=val_valid.index)
lower_series = pd.Series(conf[:, 0], index=val_valid.index)
upper_series = pd.Series(conf[:, 1], index=val_valid.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(valid, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

#RMSE and MAE metrics
print("MAE")
print(mean_absolute_error(fc_series, valid))
print("RMSE")
print(np.sqrt(mean_squared_error(fc_series, valid)))
timeexec = time.time()-start
print('{0}'.format(timeexec))
