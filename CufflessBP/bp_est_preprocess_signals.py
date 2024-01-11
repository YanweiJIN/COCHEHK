'''
Created data: 27 Aug 2023
Last modified: 27 Aug 2023
Editor: Yanwei JIN (HKCOCHE)
Supervisors: Prof. Beeluan KHOO, Prof. Rosa CHAN and Prof. Raymond CHAN
Introduction: Input original signals
'''

import numpy as np
ppg_ori = np.array([])
bp_ori = np.array([])
ecg_ori = np.array([])
patient_data = np.array([])
fs = 125
patient = 0

####------------------------------------------------------- Part1: Normalize PPG-BP-ECG signals ---------------------------------------------------------####

## 1. Normalize signals to [0,1]
ppg_norm = (ppg_ori-min(ppg_ori)) / (max(ppg_ori)-min(ppg_ori))
bp_norm = (bp_ori-min(bp_ori)) / (max(bp_ori)-min(bp_ori))
ecg_norm = (ecg_ori-min(ecg_ori)) / (max(ecg_ori)-min(ecg_ori))
#### type(signal_norm) is numpy.ndarray, signal_norm.shape = (1, 3000)

## 2. Show 3 signals in 3 axs
#import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 6), dpi=96)

axs[0].plot(ppg_norm)
axs[0].set_title('PPG norm')

axs[1].plot(bp_norm)
axs[1].set_title('BP norm')

axs[2].plot(ecg_norm)
axs[2].set_title('ECG norm')
xtick_positions = np.arange(0, patient_data.shape[1], fs)
xtick_labels = [f'{int(xtick_position/fs)}s' for xtick_position in xtick_positions]
plt.xticks(xtick_positions, xtick_labels)
plt.xlabel('Time (seconds)', fontsize=12)

fig.suptitle(f'Patient {patient}')
plt.tight_layout()
plt.show()

## Show 3 signal_norm in 1 fig
#import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,3),dpi=96)
plt.plot(ppg_norm, label='PPG_norm')
plt.plot(bp_norm, label='BP_norm')
plt.plot(ecg_norm, label='ECG_norm')
xtick_positions = np.arange(0, patient_data.shape[1], fs)
xtick_labels = [f'{int(xtick_position/fs)}s' for xtick_position in xtick_positions]
plt.xticks(xtick_positions, xtick_labels)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Normalized Signal', fontsize=12)
plt.title(f'Signal of Patient {patient}\n', fontsize=12)
plt.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.83, 1.155))
plt.show()

####------------------------------------------------------- Part2: Filter and Clean PPG-BP-ECG signals ---------------------------------------------------------####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.signal as signal
import scipy.io as sio
import scipy.stats
import neurokit2 as nk

## 1. Clean data: make every beat only have one PPG/BP peak

def clean_data(data, number):
    df = pd.DataFrame(data, columns=(('PPG','BP','ECG')))
    ppg_data = df['PPG']
    bp_data = df['BP']
    ecg_data = df['ECG']

    ## Filter signals
    sampling_rate = 125
    ppg_filtered = nk.signal_filter(ppg_data, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=sampling_rate)
    ecg_filtered = nk.signal_filter(ecg_data, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=sampling_rate)
    
    filtered_df = df
    filtered_df['PPG'], filtered_df['ECG'] = ppg_filtered, ecg_filtered
    
    #### Find ECG R-peaks
    ecgpeaks, _ = signal.find_peaks(ecg_filtered, distance=sampling_rate//2.5)

    #### Make sure onle one PPG peak and one BP peak between two adjacent ECG R-peaks
    cleaned_df = pd.DataFrame()
    sum_beats = 0
    times_recorder = []

    for R_peak_number in range(len(ecgpeaks)-1):
        sum_beats += 1
        if ecg_filtered[ecgpeaks[R_peak_number]] > 0.2 and ecg_filtered[ecgpeaks[R_peak_number + 1]] > 0.2: # make sure the peaks are R-peak
            onebeat_ppgpeak, _ = signal.find_peaks(ppg_filtered[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], distance = sampling_rate//2.5)
            onebeat_bppeak, _ = signal.find_peaks(bp_data[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], distance = sampling_rate//2.5)
            if len(onebeat_ppgpeak) == 1 and ppg_filtered[ecgpeaks[R_peak_number] + onebeat_ppgpeak] > 0 and len(onebeat_bppeak) == 1 : # make sure only one BP signal for one beat 
                cleaned_df = pd.concat([cleaned_df, filtered_df[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]]])
                times_recorder.append(sum_beats)                
    
    #### Reset indexs:
    cleaned_df = cleaned_df.reset_index()

    if len(cleaned_df) > 18000:
        return cleaned_df.to_csv(f'/Users/jinyanwei/Desktop/BP_Model/Model_record/cleaned_data/Part1_cleaned{number}.csv')
    else:
        return

## 2. Show the results

'''  

#### show the filtered PPG and ECG signals
ppg_df = pd.DataFrame(columns=(('PPG_original', 'PPG_filtered')))
ppg_df['PPG_original'] = ppg_data
ppg_df['PPG_filtered'] = ppg_filtered
display(ppg_df)
figPPG = px.line(ppg_df)
figPPG.show()

ecg_df = pd.DataFrame(columns=(('ECG_original', 'ECG_filtered')))
ecg_df['ECG_original'] = ecg_data
ecg_df['ECG_filtered'] = ecg_filtered
display(ecg_df)
figECG = px.line(ecg_df)
figECG.show()


#### show peaks in PPG and ECG signals

ppgpeaks, _ = signal.find_peaks(ppg_filtered, distance=sampling_rate//2.5)
print(ppgpeaks, len(ppgpeaks))
plt.figure(figsize=(50, 10))
for index in ppgpeaks:
    plt.scatter(index, ppg_filtered[index], marker="*")
plt.plot(ppg_filtered)
plt.show()

print(ecgpeaks, len(ecgpeaks))
plt.figure(figsize=(50, 10))
for index in ecgpeaks:
    plt.scatter(index, ecg_filtered[index], marker="*")
plt.plot(ecg_filtered)
plt.show()


#### Show  original cleaned data:

print('All beats: ', sum_beats, ', Reserved beats: ', len(times_recorder))
display(cleaned_df)
figcleaned = px.line(cleaned_df)
figcleaned.show()


#### Show final cleaned data:

display(cleaned_df)
figcleaned = px.line(cleaned_df)
figcleaned.show()

'''

#### Save acceptable cleaned data to Part_Cleaned_.cvs
#### Warning!!!! NEVER FORGET to change the name of the csv!!!!!!!!!

