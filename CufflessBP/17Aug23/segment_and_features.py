#### Please use cleaned data

'''
#### if warning:

warnings.filterwarnings("ignore", category=RuntimeWarning, module='nolds')

'''


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

def features_data(file_path="/Users/jinyanwei/Desktop/BP_Model/Model_record/cleaned_data", number=9999999):
    if not os.path.exists(f"{file_path}/Part1_cleaned{number}.csv"):
        return
           
    df = pd.read_csv(f"{file_path}/Part1_cleaned{number}.csv")
    ppg_data = df['PPG']
    bp_data = df['BP']
    ecg_data = df['ECG']

    ## Segment signals from R-peak to get data of each beat.
    sampling_rate = 125
    ecgpeaks, _ = signal.find_peaks(ecg_data, distance=sampling_rate//2.5)
    
    ppg_segments, ecg_segments, SBPlist, DBPlist, bplist = [], [], [], [], []

    for peak_number in range(1, len(ecgpeaks)-2):    # data will be more stable from the second R-peak
        ppg_segments.append(ppg_data[ecgpeaks[peak_number]:ecgpeaks[peak_number + 1]].values)
        ecg_segments.append(ecg_data[ecgpeaks[peak_number]:ecgpeaks[peak_number + 1]].values)
        bplist = bp_data[ecgpeaks[peak_number]:ecgpeaks[peak_number+1]]
        SBPlist.append(max(bplist))
        DBPlist.append(min(bplist))
    ## Extract features of PPG and ECG.

    #### Def features:

    from numpy.fft import fft

    def extract_ppg_features(signal):

        '''
        #### Chaotic features

        lyap_r = nolds.lyap_r(signal)
        hurst_exp = nolds.hurst_rs(signal)
        corr_dim = nolds.corr_dim(signal, 1)

        '''

        #### Time domain features
        mean = np.mean(signal)
        std_dev = np.std(signal)
        skewness = scipy.stats.skew(signal)
        kurtosis = scipy.stats.kurtosis(signal)

        #### Frequency domain features
        fft_values = fft(signal)
        power_spectrum = np.abs(fft_values)**2
        total_power = np.sum(power_spectrum)
        low_freq_power = np.sum(power_spectrum[:len(power_spectrum)//2]) / total_power
        high_freq_power = np.sum(power_spectrum[len(power_spectrum)//2:]) / total_power

        ppg_features = {
            #'lyap_r': lyap_r,
            #'hurst_exp': hurst_exp,
            #'corr_dim': corr_dim,
            'ppg_mean': mean,
            'ppg_std_dev': std_dev,
            'ppg_skewness': skewness,
            'ppg_kurtosis': kurtosis,
            'ppg_low_freq_power': low_freq_power,
            'ppg_high_freq_power': high_freq_power
        }

        return ppg_features


    def extract_ecg_features(signal):

        '''
        #### Chaotic features

        lyap_r = nolds.lyap_r(signal)
        hurst_exp = nolds.hurst_rs(signal)
        corr_dim = nolds.corr_dim(signal, 1)

        '''

        # Time domain features
        mean = np.mean(signal)
        std_dev = np.std(signal)
        skewness = scipy.stats.skew(signal)
        kurtosis = scipy.stats.kurtosis(signal)

        # Frequency domain features
        fft_values = fft(signal)
        power_spectrum = np.abs(fft_values)**2
        total_power = np.sum(power_spectrum)
        low_freq_power = np.sum(power_spectrum[:len(power_spectrum)//2]) / total_power
        high_freq_power = np.sum(power_spectrum[len(power_spectrum)//2:]) / total_power

        ecg_features = {
            #'lyap_r': lyap_r,
            #'hurst_exp': hurst_exp,
            #'corr_dim': corr_dim,
            'ecg_mean': mean,
            'ecg_std_dev': std_dev,
            'ecg_skewness': skewness,
            'ecg_kurtosis': kurtosis,
            'ecg_low_freq_power': low_freq_power,
            'ecg_high_freq_power': high_freq_power
        }

        return ecg_features


    #### Get features and save to csv:
    ppg_feature_list = [extract_ppg_features(ppg_segment) for ppg_segment in ppg_segments]
    ecg_feature_list = [extract_ecg_features(ecg_segment) for ecg_segment in ecg_segments]
    features_df = pd.concat([pd.DataFrame(ppg_feature_list), pd.DataFrame(ecg_feature_list), pd.DataFrame({'SBP': SBPlist, 'DBP':DBPlist})], axis=1)

    return features_df.to_csv(f'/Users/jinyanwei/Desktop/BP_Model/Model_record/features_data/Part1_feature{number}.csv')




################################################################################################################################################################################




'''
#### Use moving average methods to segment signals

def segment_signal(signal, rpeaks, window=15):
    segments = []
    for rpeak in rpeaks:
        start = max(0, rpeak - window)
        end = min(len(signal), rpeak + window)
        segment = signal[start:end]
        segments.append(segment)
    return segments

ppgpeaks, _ = signal.find_peaks(ppg_filtered, distance=sampling_rate//2.5)
ecgpeaks, _ = signal.find_peaks(ecg_filtered, distance=sampling_rate//2.5)

ppg_segments = segment_signal(ppg_data, ppgpeaks[:len(ppgpeaks)-1])
ecg_segments = segment_signal(ecg_data, ecgpeaks[:len(ecgpeaks)-1])


'''


## To be better: Try to save the Fist and the Last fragment of BP


################################################################################################################################################################################



'''
#### Other features:

fs = sampling_rate

#### Calculate pulse rate (in beats per minute)

ppg_pulse_rate = 60 * fs / np.mean(np.diff(ppgpeaks))
ecg_pulse_rate = 60 * fs / np.mean(np.diff(ecgpeaks))
print('ppg_pulse_rate:' , ppg_pulse_rate)
print('ecg_pulse_rate:' , ecg_pulse_rate)

#### Calculate peak-to-peak time intervals

peak_to_peak_intervals = np.diff(peaks) / fs

#### Calculate pulse width at half-maximum amplitude
pulse_widths = signal.peak_widths(ppg_filtered, peaks, rel_height=0.5)[0] / fs

#### Calculate area under the curve
auc = np.trapz(ppg_filtered, dx=1/fs)

#### Calculate power spectral density (PSD) and peak frequency
freqs, psd = signal.welch(ppg_filtered, fs)
peak_frequency = freqs[np.argmax(psd)]

#### Calculate spectral entropy
spectral_entropy = -np.sum(psd * np.log2(psd))

#### Compile features into a dictionary
features = {
    'pulse_rate': pulse_rate,
    'mean_peak_to_peak_interval': np.mean(peak_to_peak_intervals),
    'std_peak_to_peak_interval': np.std(peak_to_peak_intervals),
    'mean_pulse_width': np.mean(pulse_widths),
    'std_pulse_width': np.std(pulse_widths),
    'area_under_curve': auc,
    'peak_frequency': peak_frequency,
    'spectral_entropy': spectral_entropy
}

print(features)

'''



################################################################################################################################################################################

