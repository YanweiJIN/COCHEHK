'''
Created data: 27 Aug 2023
Last modified: 27 Aug 2023
Editor: Yanwei JIN (HKCOCHE)
Supervisors: Prof. Beeluan KHOO, Prof. Rosa CHAN and Prof. Raymond CHAN
Introduction: Methods of reading signals data.
'''

####--------------------------------------------- Part1: Read UCI Data (1000 Patients per file)----------------------------------------------------####

## 1. Open a 1000 patients mat file of UCI dataset
import scipy.io
data = scipy.io.loadmat('/Users/jinyanwei/Desktop/BP_Model/Data/Cuffless_BP_Estimation/part_1.mat')
#### 'data' is a dictionary, { 'p': array([[array([[ 1.75953079e+00,  1.71847507e+00,  1.68426197e+00, ...}
#### data['p'].shape: (1, 1000). Which contains 1000 patients.
#### Single patients data['p'][0, patient].shape is (3, 61000), patient1 for example, contains PPG, BP, ECG in order.

## 2. Choose a patient from UCI dataset
import numpy as np
fs = 125
patient = 0
n_seconds_to_load = 30 # seconds
signal_length = n_seconds_to_load * fs
patient_data = data['p'][0,patient][:,:signal_length]

## 3. Choose signals of this patient
ppg_ori = patient_data[0]
bp_ori = patient_data[1]
ecg_ori = patient_data[2]
#### type(signal_ori) is numpy.ndarray, signal_ori.shape = (1, 3000)

## 4. Show 3 signals
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 6), dpi=96)

axs[0].plot(ppg_ori)
axs[0].set_title('PPG ori')

axs[1].plot(bp_ori)
axs[1].set_title('BP ori')

axs[2].plot(ecg_ori)
axs[2].set_title('ECG ori')
xtick_positions = np.arange(0, patient_data.shape[1], fs)
xtick_labels = [f'{int(xtick_position/fs)}s' for xtick_position in xtick_positions]
plt.xticks(xtick_positions, xtick_labels)
plt.xlabel('Time (seconds)', fontsize=12)

plt.tight_layout()
plt.show()

####--------------------------------------------------- Part2: Read MIMIC Ⅳ Waveform Data -------------------------------------------------------####

## 1. Creat segement '.csv' file: /Users/jinyanwei/Desktop/BP_Model/Jinyw_code/wfdb_read/wfdb_mimic4wdb.ipynb

## 2. Read MIMICⅣ segment list
import sys
from pathlib import Path
import wfdb
import pandas as pd
database_name = 'mimic4wdb/0.1.0'
segment_table = pd.read_csv('/Users/jinyanwei/Desktop/BP_Model/Data/mimic4wdf/matching_records_10mins.csv')
segment_names = segment_table['seg_name']
segment_dirs = segment_table['dir']
print(f'Total {len(segment_names)} segments.')
#### Total 52 segments.

## 3. Choose a patient from MIMIC Ⅳ Waveform
rel_segment_n = 2
rel_segment_name = segment_names[rel_segment_n]
rel_segment_dir = segment_dirs[rel_segment_n]
print(f"Specified segment '{rel_segment_name}' in directory: '{rel_segment_dir}'")
#### Specified segment '84248019_0005' in directory: 'mimic4wdb/0.1.0/waves/p102/p10209410/84248019'
start_seconds = 0 # time since the start of the segment at which to begin extracting data
n_seconds_to_load = 300 # seconds
segment_metadata = wfdb.rdheader(record_name=rel_segment_name,
                                 pn_dir=rel_segment_dir)
print(f"Metadata loaded from segment: {rel_segment_name}")
#### Metadata loaded from segment: 84248019_0005
fs = round(segment_metadata.fs)
sampfrom = fs * start_seconds
sampto = fs * (start_seconds + n_seconds_to_load)
segment_data = wfdb.rdrecord(record_name=rel_segment_name,
                             sampfrom=sampfrom,
                             sampto=sampto,
                             pn_dir=rel_segment_dir)
print(f"{n_seconds_to_load} seconds of data extracted from segment {rel_segment_name}")
#### 300 seconds of data extracted from segment 84248019_0005
print(f'Signal names: {segment_data.sig_name}') 
#### ['II', 'V', 'aVR', 'ABP', 'Pleth', 'Resp']
patient_data = pd.DataFrame(segment_data.p_signal, columns=segment_data.sig_name) 

## 4. Show Signals
title_text = f"Segment {rel_segment_name}"
wfdb.plot_wfdb(record=segment_data,
               title=title_text,
               time_units='seconds', figsize=(30,6)) 


####-------------------------------------------- Part3: Read UCI Data (3000 Patients per file)  ------------------------------------------------------####

#### dataset: UCI Machine Learning Repository_Cuff-Less Blood Pressure Estimation Data Set
#### data address: https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation
#### Citation Request: M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time, IEEE International Symposium on Circuits and Systems (ISCAS'15), 2015.
####    The data set is in matlab's v7.3 mat file, accordingly it should be opened using new versions of matlab or HDF libraries in other environments.(Please refer to the Web for more information about this format) 
####    This database consist of a cell array of matrices, each cell is one record part. 
####    In each matrix each row corresponds to one signal channel: 
####        1: PPG signal, FS=125Hz; photoplethysmograph from fingertip 
####        2: ABP signal, FS=125Hz; invasive arterial blood pressure (mmHg) 
####        3: ECG signal, FS=125Hz; electrocardiogram from channel II

import numpy as np
import h5py

##1. Open big '.mat' data and check the shape:
def open_data(file_path = '/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_1.mat'):
    with h5py.File(file_path, 'r') as file:
        references = np.array(file[list(file.keys())[1]]) # Access a specific variable and convert it to a NumPy array
        # Dereference the objects and store them in a list
        data = []
        for ref in references.flat:
            dereferenced_object = np.array(file[ref])
            data.append(dereferenced_object)
    '''    
    ## Show the shape of the data
    for i, array in enumerate(data):
        print(f"Shape of data[{i}]: {array.shape}")
    '''

    return data    
        
##2. Merge many patients into one df
'''
Include_Data = []
df = pd.DataFrame(columns=(('PPG','BP', 'ECG')))

for patient in range(len(data)):
    if len(data[patient]) > 7499:
        data_df = pd.DataFrame(data[patient][:1000], columns=(('PPG','BP', 'ECG')))
        df = pd.concat([df, data_df])
        Include_Data.append(patient)

display(df)

#### Open data:

RMSE_SBP, RMSE_DBP, MAE_SBP, MAE_DBP, Abnormal_location= [], [], [], [], []

for patient in range(10):
    if len(data[patient]) > 999:
        df = pd.DataFrame(data[patient], columns=(('PPG','BP', 'ECG'))) #选取前10000条数据
        ecg_data = df['ECG']
        ppg_data = df['PPG']
        bp_data = df['BP']

        #### Filter signals and ...

        .
        .
        .
        .
        .
        .

        #### Calculate root mean squared error and mean absolute error for both SBP and DBP
        rmse_sbp = metrics.mean_squared_error(y_test["SBP"], y_pred[:, 0])**0.5
        rmse_dbp = metrics.mean_squared_error(y_test["DBP"], y_pred[:, 1])**0.5
        mae_sbp = metrics.mean_absolute_error(y_test["SBP"], y_pred[:, 0])
        mae_dbp = metrics.mean_absolute_error(y_test["DBP"], y_pred[:, 1])
        
        if rmse_sbp > 5 or rmse_dbp >5:
            Abnormal_location.append(patient)

        RMSE_SBP.append(rmse_sbp)
        RMSE_DBP.append(rmse_dbp)
        MAE_SBP.append(mae_sbp)
        MAE_DBP.append(mae_dbp)

print('RMSE_SBP: ', RMSE_SBP)
print('RMSE_DBP: ', RMSE_DBP)
print('MAE_SBP: ', MAE_SBP)
print('MAE_DBP: ', MAE_DBP)
print('Abnormal location: ', Abnormal_location)
print('RMSE_SBP: ', mean(RMSE_SBP), 'RMSE_DBP', mean(RMSE_DBP), 'MAE_SBP: ', mean(MAE_SBP), 'MAE_DBP', mean(MAE_DBP))

'''
