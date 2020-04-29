import re
import warnings
from os import listdir
from os.path import isfile, join
from scipy import stats
import numpy as np
import sklearn.preprocessing as skl
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import ARDRegression, LinearRegression

#initialization:
register_matplotlib_converters()
warnings.filterwarnings("ignore")
mpl.rcParams['figure.dpi'] = 1200


valence_path = './emotional_behaviour/valence/'
arousal_path = './emotional_behaviour/arousal/'
val_gs_path = './emotional_behaviour/gold_standard/valence/'
aro_gs_path = './emotional_behaviour/gold_standard/arousal/'
au_path = './AU_reindex_new/'
index_name = 'time'
gs_key = ['gold standard']

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



#Validation
    
val_valid = pd.DataFrame(columns=gs_key)
for i in range(12, 14):
    temp_df = pd.DataFrame(val_ewe[i])
    temp_df.index = temp_df.index + 300*i
    val_valid = val_valid.append(temp_df)
    
aro_valid = pd.DataFrame(columns=gs_key)
for i in range(12, 14):
    temp_df = pd.DataFrame(aro_ewe[i])
    temp_df.index = temp_df.index + 300*i
    aro_valid = aro_valid.append(temp_df)
    
au_valid = pd.DataFrame(columns=au_cols)
for i in range(12, 14):
    temp_au_df = pd.DataFrame(au[i])
    temp_au_df.index = temp_au_df.index + 300*i
    au_valid = au_valid.append(temp_au_df)
    
    


# #############################################################################
# Generating simulated data with Gaussian weights

"""
# Parameters of the example
np.random.seed(0)
n_samples, n_features = 100, 45
# Create Gaussian data
X = np.random.randn(n_samples, n_features)
"""
n_features = 17
#one video 7501

#n_samples = au_train.shape[0]/4
#n_samples = 7501

#validation set
n_samples_valid = 7501 
#try with 1000/3000/5000/7501 
#n_samples = au_train.shape[0]/4
n_samples = 7501

#n_samples_valid = 7501 #np.floor(n_samples/10)

# Create weights with a precision lambda_ of 4.
lambda_ = 4.




X = au_train.values[0:n_samples,:]
x_valid = au_valid.values[0:n_samples_valid,:]

#target: swtich between Valence and Arousal
#target
y = val_train.values[0:n_samples]
#y= aro_train.values[0:n_samples]
y_valid = val_valid.values[0:n_samples_valid]
#y_valid = aro_valid.values[0:n_samples_valid]

plt.figure(figsize=(6, 5))
plt.plot(np.arange(0,n_samples_valid),y_valid, color='gold', linewidth=2,
         label="Ground Truth")
# Fit the ARD Regression
clf = ARDRegression(compute_score=True)
print('ARD regression')
clf.fit(X, y)
ARD_coef = clf.coef_
print('fitted')


#show sorted action units weights

au_coeff = pd.Series(ARD_coef)
au_coeff.index = au_cols
au_sort= au_coeff.abs().sort_values(ascending=False)
au_sort.index
au_sort = au_coeff[au_sort.index]

au_cols_plot = ['1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', '20', '23', '25', '26', '45']
au_coeff.index = au_cols_plot
print(au_sort)



plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(clf.coef_, color='darkblue', linestyle='-', linewidth=2,
         label="ARD estimate")
#plt.plot(ols.coef_, color='yellowgreen', linestyle=':', linewidth=2,)
#plt.plot(w, color='orange', linestyle='-', linewidth=2, label="Ground truth")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc=1)

plt.figure(figsize=(6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins=n_features, color='navy', log=True)
#plt.scatter(clf.coef_[relevant_features], np.full(10, 5.),
#            color='gold', marker='o', label="Relevant features")
plt.ylabel("#Features")
plt.xlabel("Values of the weights")
plt.legend(loc=1)

plt.figure(figsize=(6, 5))
plt.title("Marginal log-likelihood")
plt.plot(np.arange(0,len(clf.scores_),1),clf.scores_, color='navy', linewidth=2)
plt.ylabel("Score")
plt.xlabel("Iterations")

au_coeff.index = au_cols_plot
plt.figure(figsize=(6,5))
plt.title("Action unit weights")
plt.bar(au_coeff.index,au_coeff.values)
plt.xlabel("Action units")
plt.ylabel("Values of the weights")


#fare lo smooth qui
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.figure(figsize=(6, 5))
plt.title("Predictions")
y_predict, y_std = clf.predict(x_valid, return_std=True)
axis =np.arange(0,n_samples_valid)
plt.plot(axis,y_predict, color='lightsteelblue', linewidth=0.5, linestyle='dotted', markersize=0.8, label="ARD", marker='.')
plt.plot(axis,smooth(y_predict,100), color='navy',label="ARD smoothed")
plt.plot(axis,y_valid, color='gold', linewidth=2, label="Ground Truth")
plt.xlabel("Samples")
plt.ylabel("Valence")
plt.legend(loc='upper left', fontsize=8)

"""
# Plotting some predictions for polynomial regression
def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise


degree = 10
for i in range(n_features):
    featureX = X[:,i]
    #y = f(X, noise_amount=1)
    clf_poly = ARDRegression(threshold_lambda=1e5)
    clf_poly.fit(np.vander(featureX, degree), y)

    X_plot = np.linspace(0, n_samples, n_samples)
    y_plot = y
    y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
    plt.figure(figsize=(6, 5))
    plt.errorbar(X_plot, y_mean, y_std, color='navy',
                 label="Polynomial ARD", linewidth=2)
    plt.plot(X_plot, y_plot, color='gold', linewidth=2,
             label="Ground Truth")
    plt.ylabel("Output y")
    plt.xlabel("Feature X")
    plt.legend(loc="lower left")
    plt.show()
"""
