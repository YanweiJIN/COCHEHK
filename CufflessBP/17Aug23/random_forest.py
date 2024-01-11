#### Random Forest Model
#### Please take a look on the Features Data before use it


import numpy as np
import pandas as pd
import h5py
import os
import re
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.signal as signal
import scipy.io as sio
import scipy.stats
import neurokit2 as nk
import heartpy as hp
import nolds
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings



################################################################################################################################################################################

def run_random_forest(df):

    X_scaled = df.iloc[:, 1:-2].values
    y_bp = df.iloc[:, -2:]

    #### Split into training and test sets randomly
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bp, test_size=0.2, random_state=42)

    #### Initialize the random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    #### Train the model on the training data
    model.fit(X_train, y_train)

    #### Predict on the test set
    y_pred = model.predict(X_test)

    #### Get the result of BP
    result_bp = pd.DataFrame(columns=['SBP_actu','SBP_pred','DBP_actu','DBP_pred'])
    result_bp['SBP_actu'] = y_test['SBP']
    result_bp['DBP_actu'] = y_test['DBP']
    result_bp['SBP_pred'] = y_pred[:,0]
    result_bp['DBP_pred'] = y_pred[:,1]
    result_bp = result_bp.sort_index()

    #### Calculate root mean squared error and mean absolute error for both SBP and DBP
    rmse_sbp = metrics.mean_squared_error(result_bp['SBP_actu'], result_bp['SBP_pred'])**0.5
    rmse_dbp = metrics.mean_squared_error(result_bp['DBP_actu'], result_bp['DBP_pred'])**0.5
    mae_sbp = metrics.mean_absolute_error(result_bp['SBP_actu'], result_bp['SBP_pred'])
    mae_dbp = metrics.mean_absolute_error(result_bp['DBP_actu'], result_bp['DBP_pred'])

    print(f'{len(y_train)} beats to train and {len(y_test)} beats to test.', 
      '\n'f"Root mean squared error for SBP: {rmse_sbp:.3f}", 
      '\n'f"Root mean squared error for DBP: {rmse_dbp:.3f}", 
      '\n'f"Mean absolute error for SBP: {mae_sbp:.3f}", 
      '\n'f"Mean absolute error for DBP: {mae_dbp:.3f}"
      )
    
    def write_record_totxt():
      file_handle = open('/Users/jinyanwei/Desktop/BP_Model/Model_record/random_forest_result.txt', mode='a')
      file_handle.write(f'{len(y_train)} beats to train and {len(y_test)} beats to test.')
      file_handle.write('\n')
      file_handle.write(f"Root mean squared error for SBP: {rmse_sbp:.3f}")
      file_handle.write('\n')
      file_handle.write(f"Root mean squared error for DBP: {rmse_dbp:.3f}")
      file_handle.write('\n')
      file_handle.write(f"Mean absolute error for SBP: {mae_sbp:.3f}")
      file_handle.write('\n')
      file_handle.write(f"Mean absolute error for DBP: {mae_dbp:.3f}")
      file_handle.write('\n')
      file_handle.write('\n')
      return

    write_record_totxt()
    #### Draw pictures

    plt.figure(figsize=(30, 15))

    plt.subplot(2, 1, 1)
    x1 = np.array((range(len(result_bp))))
    sbp1 = np.array(result_bp['SBP_pred'])
    sbp2 = np.array(result_bp['SBP_actu'])
    plt.plot(x1, sbp1, label='SBP_pred')
    plt.plot(x1, sbp2, label='SBP_actu')
    plt.title('SBP')

    plt.subplot(2, 1, 2)
    x2 = x1
    dbp1 = np.array(result_bp['DBP_pred'])
    dbp2 = np.array(result_bp['DBP_actu'])
    plt.plot(x2, dbp1, label='DBP_pred')
    plt.plot(x2, dbp2, label='DBP_actu')
    plt.title('DBP')

    plt.legend()

    return plt.show()





################################################################################################################################################################################


## Try to save the model:


## Another way to spilt

'''    
#### Split into training and test sets
X_train = df.iloc[:, 1:-2].values
X_test = df.iloc[:, 1:-2].values
y_train = df.iloc[:, -2:]
y_test = df.iloc[:, -2:]

'''








################################################################################################################################################################################

