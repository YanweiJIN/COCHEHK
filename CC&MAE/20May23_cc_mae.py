## Editor: YanweiJIN (COCHE HK)
## Supervisor: Prof. ZHANG Yuanting
## Note: Calculate CC and MAE.
## 17Aug23 Modify: Optimized key points extraction in signals.
## 04Sep23 Modify: Optimized the site retention process.
## 12Sep23 Modify: Adjust the LSTM model to one patient loss close to 0.

def fourth_version():
    import pandas as pd
    import numpy as np  

    lstmdata = pd.read_csv('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/100bpppgecgcc20.csv')
    randomdata = pd.read_csv('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/fourth_July_23_ccmae/wave_ppg_ecg_random_20test_30s.csv')
    display(lstmdata)
    display(randomdata)
    np.std(list(randomdata['refbp_estbp_cc']))

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from keras.losses import mean_squared_error
    def custom_loss(y_true, y_pred):
        loss = mean_squared_error(y_true*250, y_pred*250)
        return loss
    # lstmmodel = tf.keras.models.load_model('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/bpwave_lstm_model100.h5')
    tf.keras.utils.get_custom_objects().update({'custom_loss': custom_loss})
    lstmmodel = tf.keras.models.load_model('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/bpwave_lstm_model100.h5', custom_objects={'custom_loss': custom_loss})

    patients_list = [1,3,5,16,23,24,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,44,45,46,63,67,70,71,73,76,78,79,80,81,82,103,104,120,128,129,151,156,158,161,170,172,173,183,186,187,192,193,194,196,197,198,199,200,201,203,206,207,213,214,218,220,221,227,229,236,243,248,253,254,284,286,295,298,299,303,304,305,317,336,338,343,344,346,348,350,351,355,356,357,368,369,371,372,403,412]
    testlist = patients_list[-20:]

    ### CC
    #from keras.models import load_model
    #bpwave_lstm_model = load_model('lstm_model.h5')
    import pandas as pd
    pred_bps = []
    cc_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ecg_cc', 'refbp_estbp_md', 'refbp_estbp_sd')))

    for i in range(len(test_bps)):
        test_ppg = test_ppgs[i]
        test_ecg = test_ecgs[i]
        test_feature = np.column_stack((test_ppg, test_ecg))
        X_test, y_test = reshape_data(test_feature, test_bps[i], time_steps)
        y_pred = bpwave_lstm_model.predict(X_test)
        y_pred = y_pred.flatten()
        test_ppg = test_ppg[time_steps-1:-1]
        test_ecg = test_ecg[time_steps-1:-1]
        #print(y_test.shape, y_pred.shape, test_ppg.shape, test_ecg.shape)
        pred_bps.append(y_pred)
        md = np.mean(y_pred*250-y_test*250)
        std = np.std(y_pred*250-y_test*250)
        cc_refbp_estbp = np.abs(np.corrcoef(y_test, y_pred)[0, 1])
        #print(cc_refbp_estbp)
        cc_refbp_ppg = np.abs(np.corrcoef(y_test, test_ppg)[0, 1])
        cc_refbp_ecg = np.abs(np.corrcoef(y_test, test_ecg)[0, 1])
        cc_df.loc[len(cc_df)] = [cc_refbp_estbp, cc_refbp_ppg, cc_refbp_ecg, md, std]

    cc_df.loc[len(cc_df)] = [np.mean(cc_df['refbp_estbp_cc']), np.mean(cc_df['refbp_ppg_cc']), np.mean(cc_df['refbp_ecg_cc']), np.mean(cc_df['refbp_estbp_md']), np.mean(cc_df['refbp_estbp_sd'])]
    cc_df = cc_df.rename(index={len(cc_df)-1: 'ave'})
    cc_df.to_csv('100bpppgecgcc20.csv')
    return






def third_version():

    import numpy as np
    from numpy import *
    import pandas as pd
    import h5py
    import os
    import re
    import matplotlib.pyplot as plt
    import plotly.express as px
    import scipy.signal as signal
    import scipy.io as sio
    import scipy.stats
    from scipy.fft import fft
    import neurokit2 as nk
    import heartpy as hp
    import nolds
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
    import warnings
    import nolds
    from sklearn.preprocessing import MinMaxScaler

    pd.read_csv("/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/wave_ppg_ecg_1.csv")
    import sys
    sys.path.append('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/')
    from read_data import open_data
    from filter_and_clean_data import clean_data
    from segment_and_features import features_data
    from random_forest import run_random_forest

    all_data = open_data('/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_1.mat')

    class SelectData:
        def __init__(self, data_array, data_begin=0, data_end=-1):
            self.array = data_array[data_begin:data_end]

        def signal_data(self): # PPG, BP, ECG
            return self.array[:,0], self.array[:,1], self.array[:,2]

        def show_data(self, show_begin=0, show_end=-1):
            fig, ax = plt.subplots(figsize=(30,6), dpi=300)
            ax.plot(self.array[show_begin:show_end])
            plt.show(fig)
            
    def find_first_peak(signal):
        # Calculate the first order difference
        diff_signal = np.diff(signal)
        middle_signal = (np.max(signal) + np.min(signal))/2
        # Find the index of the first positive-to-negative transition
        for i in range(len(diff_signal) - 1):
            if (diff_signal[i] > 0 and diff_signal[i + 1] < 0) and (signal[i] > middle_signal):
                return i + 1
        return -1

    def align_first_peaks(signal1, signal2):
        # Find the index of the first peak in each signal
        first_peak_index1 = find_first_peak(signal1)
        first_peak_index2 = find_first_peak(signal2)
        index_diff = first_peak_index1 - first_peak_index2
        if index_diff < 0:
            aligned_signal1 = signal1
            aligned_signal2 = signal2[abs(index_diff):]
        elif index_diff > 0:
            aligned_signal1 = signal1[index_diff:]
            aligned_signal2 = signal2
        else:
            aligned_signal1 = signal1
            aligned_signal2 = signal2

        return aligned_signal1, aligned_signal2

    def cleaned_data(ppg_aligned, bp_aligned, ecg_standardized):
        sampling_rate = 125
        ecgpeaks, _ = signal.find_peaks(ecg_standardized, distance=sampling_rate//2.5)
        sum_beats = 0
        times_recorder = []
        cleaned_df = pd.DataFrame()
        for R_peak_number in range(len(ecgpeaks)-1):
            sum_beats += 1
            if ecg_standardized[ecgpeaks[R_peak_number]] > 0.2 and ecg_standardized[ecgpeaks[R_peak_number + 1]]>0.2:
                onebeat_bppeak, _ = signal.find_peaks(bp_aligned[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], distance = sampling_rate//2.5)
                if len(onebeat_bppeak) == 1 : # make sure only one BP signal for one beat 
                    saved_df = pd.DataFrame([ppg_aligned[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], bp_aligned[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], ecg_standardized[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]]]).T
                    cleaned_df = pd.concat([cleaned_df, saved_df])
                    times_recorder.append(sum_beats)
        cleaned_df.columns=('PPG', 'BP', 'ECG')
        return cleaned_df

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

    def get_bp_features(df):
        sampling_rate = 125
        ecgpeaks, _ = signal.find_peaks(df['ECG'], distance=sampling_rate//2.5)
        ppg_segments, ecg_segments, SBPlist, DBPlist, bplist = [], [], [], [], []
        for peak_number in range(1, len(ecgpeaks)-2):    # data will be more stable from the second R-peak
            ppg_segments.append(df['PPG'][ecgpeaks[peak_number]:ecgpeaks[peak_number + 1]].values)
            ecg_segments.append(df['ECG'][ecgpeaks[peak_number]:ecgpeaks[peak_number + 1]].values)
            bplist = df['BP'][ecgpeaks[peak_number]:ecgpeaks[peak_number+1]]
            SBPlist.append(max(bplist))
            DBPlist.append(min(bplist))

        ppg_feature_list = [extract_ppg_features(ppg_segment) for ppg_segment in ppg_segments]
        ecg_feature_list = [extract_ecg_features(ecg_segment) for ecg_segment in ecg_segments]
        features_df = pd.concat([pd.DataFrame(ppg_feature_list), pd.DataFrame(ecg_feature_list)], axis=1)

        return np.array(SBPlist), np.array(DBPlist), features_df

    def for_one_patient(patient_number):  
        patient_array = SelectData(all_data[patient_number])
        ppg, bp, ecg = patient_array.signal_data()
        ppg_filtered = nk.signal_filter(ppg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
        ppg_standardized = np.around(((ppg_filtered - ppg_filtered.min()) / (ppg_filtered.max() - ppg_filtered.min())), decimals=4)
        ecg_filtered = nk.signal_filter(ecg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
        ecg_standardized = np.around(((ecg_filtered - ecg_filtered.min()) / (ecg_filtered.max() - ecg_filtered.min())), decimals=4)
        ppg_aligned, bp_aligned = align_first_peaks(ppg_standardized, bp)
        cleaned_df = cleaned_data(ppg_aligned, bp_aligned, ecg_standardized)
        ori_cleaned_df = cleaned_data(ppg_standardized, bp, ecg_standardized)
        train_sbp, train_dbp, train_features_df = get_bp_features(cleaned_df[:22500])
        sbp_lstm_model = bp_lstm_model(train_features_df, train_sbp)
        dbp_lstm_model = bp_lstm_model(train_features_df, train_dbp)
        bp_wave_lstm_model = bp_wave_lstm_modle(cleaned_df[:22500])

        cc_wave_list = []
        mae_lstm_list = []
        sampling_rate = 125
        time_steps = 10
        test_sbp, test_dbp, test_features_df = get_bp_features(cleaned_df[22500:26500])
        ref_sbp_lstm, est_sbp_lstm = get_bp_point(test_features_df, test_sbp, sbp_lstm_model)
        ref_dbp_lstm, est_dbp_lstm = get_bp_point(test_features_df, test_dbp, dbp_lstm_model)
        ref_sbp_lstm = np.squeeze(ref_sbp_lstm)
        est_sbp_lstm = np.squeeze(est_sbp_lstm)
        ref_dbp_lstm = np.squeeze(ref_dbp_lstm)
        est_dbp_lstm = np.squeeze(est_dbp_lstm)
        SelectData(np.array([ref_sbp_lstm, est_sbp_lstm]).T).show_data()
        SelectData(np.array([ref_dbp_lstm, est_dbp_lstm]).T).show_data()
        ref_bp_wave, est_bp_wave = get_bp_wave(cleaned_df[22500:26500], bp_wave_lstm_model)
        est_bp_wave = np.squeeze(est_bp_wave)
        ref_bp_aligned, est_bp_aligned = align_first_peaks(ref_bp_wave, np.squeeze(est_bp_wave))
        ppg_wave = np.squeeze(np.array(cleaned_df[22500+time_steps:22500+time_steps+len(ref_bp_wave)]['PPG']))
        ori_ppg_wave = np.squeeze(np.array(ori_cleaned_df[22500+time_steps:(22500+time_steps+len(est_bp_wave))]['PPG']))
        min_len_est = min(len(ref_bp_aligned), len(est_bp_aligned))
        ref_bp_wave_a, ppg_wave_a = align_first_peaks(ref_bp_wave, ppg_wave)
        min_len_ppg = min(len(ref_bp_wave_a), len(ppg_wave_a))

        correlation_matrix_1 = np.corrcoef(ref_bp_wave, est_bp_wave)
        correlation_coefficient_1 = correlation_matrix_1[0, 1]
        correlation_matrix_2 = np.corrcoef(ref_bp_wave_a[:min_len_ppg], ppg_wave_a[:min_len_ppg])
        correlation_coefficient_2 = correlation_matrix_2[0, 1]
        correlation_matrix_3 = np.corrcoef(ref_bp_wave[:min_len_est], est_bp_aligned[:min_len_est])
        correlation_coefficient_3 = correlation_matrix_3[0, 1]
        correlation_matrix_4 = np.corrcoef(ref_bp_wave, ori_ppg_wave)
        correlation_coefficient_4 = correlation_matrix_4[0, 1]
        cc_wave_list += [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]

        mae_bp1 = np.mean(np.abs(ref_sbp_lstm - est_sbp_lstm))
        mae_bp2 = np.mean(np.abs(ref_dbp_lstm - est_dbp_lstm))
        mae_bp3 = np.mean(np.abs(ref_bp_wave - est_bp_wave))
        mae_lstm_list += [mae_bp1, mae_bp2, mae_bp3]

        SelectData(np.array([ref_sbp_lstm, est_sbp_lstm]).T).show_data()
        SelectData(np.array([ref_dbp_lstm, est_dbp_lstm]).T).show_data()
        SelectData(np.array([ref_bp_wave, est_bp_wave]).T).show_data()
        SelectData(np.array([ref_bp_wave, est_bp_aligned]).T).show_data()
        SelectData(ppg_wave).show_data()
        SelectData(ref_bp_wave).show_data()
        SelectData(ori_ppg_wave).show_data()
        SelectData(ppg_wave_a[:min_len_ppg]).show_data()
        SelectData(ref_bp_wave_a[:min_len_ppg]).show_data()
        print(cc_wave_list)
        print(mae_lstm_list)
        return
    ## 3*（10， 30， 60）ppg 
    def for_bp_ppg_cc_mae(patient_list):
        cc_wave_df10 = pd.DataFrame(columns=(('refbp_estbp1', 'refbp_ppg1', 'refbp_aligened_estbp1', 'refbp_orippg1','refbp_estbp2', 'refbp_ppg2', 'refbp_aligened_estbp2', 'refbp_orippg2', 'refbp_estbp3', 'refbp_ppg3', 'refbp_aligened_estbp3', 'refbp_orippg3')))
        mae_lstm_df10 = pd.DataFrame(columns=(('refsbp_estsbp1', 'refdbp_estdbp1', 'refbpwave_estbpwave1', 'refsbp_estsbp2', 'refdbp_estdbp2', 'refbpwave_estbpwave2', 'refsbp_estsbp3', 'refdbp_estdbp3', 'refbpwave_estbpwave3')))
        cc_wave_df30 = pd.DataFrame(columns=(('refbp_estbp1', 'refbp_ppg1', 'refbp_aligened_estbp1', 'refbp_orippg1','refbp_estbp2', 'refbp_ppg2', 'refbp_aligened_estbp2', 'refbp_orippg2', 'refbp_estbp3', 'refbp_ppg3', 'refbp_aligened_estbp3', 'refbp_orippg3')))
        mae_lstm_df30 = pd.DataFrame(columns=(('refsbp_estsbp1', 'refdbp_estdbp1', 'refbpwave_estbpwave1', 'refsbp_estsbp2', 'refdbp_estdbp2', 'refbpwave_estbpwave2', 'refsbp_estsbp3', 'refdbp_estdbp3', 'refbpwave_estbpwave3')))
        cc_wave_df60 = pd.DataFrame(columns=(('refbp_estbp1', 'refbp_ppg1', 'refbp_aligened_estbp1', 'refbp_orippg1','refbp_estbp2', 'refbp_ppg2', 'refbp_aligened_estbp2', 'refbp_orippg2', 'refbp_estbp3', 'refbp_ppg3', 'refbp_aligened_estbp3', 'refbp_orippg3')))
        mae_lstm_df60 = pd.DataFrame(columns=(('refsbp_estsbp1', 'refdbp_estdbp1', 'refbpwave_estbpwave1', 'refsbp_estsbp2', 'refdbp_estdbp2', 'refbpwave_estbpwave2', 'refsbp_estsbp3', 'refdbp_estdbp3', 'refbpwave_estbpwave3')))
        for patient in patient_list:
            patient_array = SelectData(all_data[patient])
            ppg, bp, ecg = patient_array.signal_data()
            ppg_filtered = nk.signal_filter(ppg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
            ppg_standardized = np.around(((ppg_filtered - ppg_filtered.min()) / (ppg_filtered.max() - ppg_filtered.min())), decimals=4)
            ecg_filtered = nk.signal_filter(ecg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
            ecg_standardized = np.around(((ecg_filtered - ecg_filtered.min()) / (ecg_filtered.max() - ecg_filtered.min())), decimals=4)
            ppg_aligned, bp_aligned = align_first_peaks(ppg_standardized, bp)
            cleaned_df = cleaned_data(ppg_aligned, bp_aligned, ecg_standardized)
            ori_cleaned_df = cleaned_data(ppg_standardized, bp, ecg_standardized)
            train_sbp, train_dbp, train_features_df = get_bp_features(cleaned_df[:22500])
            sbp_lstm_model = bp_lstm_model(train_features_df, train_sbp)
            dbp_lstm_model = bp_lstm_model(train_features_df, train_dbp)
            cc_wave_list10, mae_lstm_list10 = [], []
            cc_wave_list30, mae_lstm_list30 = [], []
            cc_wave_list60, mae_lstm_list60 = [], []
            bp_wave_lstm_model = bp_wave_lstm_modle(cleaned_df[:22500])
            sampling_rate = 125
            time_steps = 10

            for i in range(3):
                test_sbp, test_dbp, test_features_df = get_bp_features(cleaned_df[(22500+(60*sampling_rate)*i):(22500+(60*sampling_rate)*(i+1))])
                ref_sbp_lstm, est_sbp_lstm = get_bp_point(test_features_df, test_sbp, sbp_lstm_model)
                ref_dbp_lstm, est_dbp_lstm = get_bp_point(test_features_df, test_dbp, dbp_lstm_model)
                ref_sbp_lstm = np.squeeze(ref_sbp_lstm)
                est_sbp_lstm = np.squeeze(est_sbp_lstm)
                ref_dbp_lstm = np.squeeze(ref_dbp_lstm)
                est_dbp_lstm = np.squeeze(est_dbp_lstm)
                ref_bp_wave, est_bp_wave = get_bp_wave(cleaned_df[(22500+(60*sampling_rate)*i):(22500+(60*sampling_rate)*(i+1))], bp_wave_lstm_model)
                est_bp_wave = np.squeeze(est_bp_wave)
                ref_bp_aligned, est_bp_aligned = align_first_peaks(ref_bp_wave, np.squeeze(est_bp_wave))
                ref_bp_aligned, est_bp_aligned = align_first_peaks(ref_bp_aligned, np.squeeze(est_bp_aligned))
                min_len_est = min(len(ref_bp_aligned), len(est_bp_aligned))
                ref_bp_aligned = ref_bp_aligned[:min_len_est]
                est_bp_aligned = est_bp_aligned[:min_len_est]
                ppg_wave = np.squeeze(np.array(cleaned_df[22500+time_steps:22500+time_steps+len(ref_bp_wave)]['PPG']))
                ori_ppg_wave = np.squeeze(np.array(ori_cleaned_df[22500+time_steps:(22500+time_steps+len(ref_bp_wave))]['PPG']))
                ref_bp_wave_a, ppg_wave_a = align_first_peaks(ref_bp_wave, ppg_wave)
                ref_bp_wave_a = ref_bp_wave_a[:min_len_ppg]
                ppg_wave_a = ppg_wave_a[:min_len_ppg]
                min_len_ppg = min(len(ref_bp_wave_a), len(ppg_wave_a))

                once_time = 10
                ref_bp_wave10 = ref_bp_wave[:int((once_time/60)*len(ref_bp_wave))]
                est_bp_wave10 = est_bp_wave[:int((once_time/60)*len(est_bp_wave))]
                ref_bp_wave_a10 = ref_bp_wave_a[:int((once_time/60)*len(ref_bp_wave_a))]
                ppg_wave_a10 = ppg_wave_a[:int((once_time/60)*len(ppg_wave_a))]
                ref_bp_aligned10 = ref_bp_aligned[:int((once_time/60)*len(ref_bp_aligned))]
                est_bp_aligned10 = est_bp_aligned[:int((once_time/60)*len(est_bp_aligned))]
                ori_ppg_wave10 = ori_ppg_wave[:int((once_time/60*len(ori_ppg_wave)))]

                correlation_matrix_1 = np.corrcoef(ref_bp_wave10, est_bp_wave10)
                correlation_coefficient_1 = correlation_matrix_1[0, 1]
                correlation_matrix_2 = np.corrcoef(ref_bp_wave_a10, ppg_wave_a10)
                correlation_coefficient_2 = correlation_matrix_2[0, 1]
                correlation_matrix_3 = np.corrcoef(ref_bp_aligned10, est_bp_aligned10)
                correlation_coefficient_3 = correlation_matrix_3[0, 1]
                correlation_matrix_4 = np.corrcoef(ref_bp_wave10, ori_ppg_wave10)
                correlation_coefficient_4 = correlation_matrix_4[0, 1]           

                ref_sbp_lstm10 = ref_sbp_lstm[:int((once_time/60)*len(ref_sbp_lstm))]
                est_sbp_lstm10 = est_sbp_lstm[:int((once_time/60)*len(est_sbp_lstm))]
                ref_dbp_lstm10 = ref_dbp_lstm[:int((once_time/60)*len(ref_sbp_lstm))]
                est_dbp_lstm10 = est_dbp_lstm[:int((once_time/60)*len(est_dbp_lstm))]
                mae_bp1 = np.mean(np.abs(ref_sbp_lstm10 - est_sbp_lstm10))
                mae_bp2 = np.mean(np.abs(ref_dbp_lstm10 - est_dbp_lstm10))
                mae_bp3 = np.mean(np.abs(ref_bp_wave10 - est_bp_wave10))

                cc_wave_list10 += [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
                mae_lstm_list10 += [mae_bp1, mae_bp2, mae_bp3] 

                once_time = 30
                ref_bp_wave30 = ref_bp_wave[:int((once_time/60)*len(ref_bp_wave))]
                est_bp_wave30 = est_bp_wave[:int((once_time/60)*len(est_bp_wave))]
                ref_bp_wave_a30 = ref_bp_wave_a[:int((once_time/60)*len(ref_bp_wave_a))]
                ppg_wave_a30 = ppg_wave_a[:int((once_time/60)*len(ppg_wave_a))]
                ref_bp_aligned30 = ref_bp_aligned[:int((once_time/60)*len(ref_bp_aligned))]
                est_bp_aligned30 = est_bp_aligned[:int((once_time/60)*len(est_bp_aligned))]
                ori_ppg_wave30 = ori_ppg_wave[:int((once_time/60*len(ori_ppg_wave)))]

                correlation_matrix_1 = np.corrcoef(ref_bp_wave30, est_bp_wave30)
                correlation_coefficient_1 = correlation_matrix_1[0, 1]
                correlation_matrix_2 = np.corrcoef(ref_bp_wave_a30, ppg_wave_a30)
                correlation_coefficient_2 = correlation_matrix_2[0, 1]
                correlation_matrix_3 = np.corrcoef(ref_bp_aligned30, est_bp_aligned30)
                correlation_coefficient_3 = correlation_matrix_3[0, 1]
                correlation_matrix_4 = np.corrcoef(ref_bp_wave30, ori_ppg_wave30)
                correlation_coefficient_4 = correlation_matrix_4[0, 1]           

                ref_sbp_lstm30 = ref_sbp_lstm[:int((once_time/60)*len(ref_sbp_lstm))]
                est_sbp_lstm30 = est_sbp_lstm[:int((once_time/60)*len(est_sbp_lstm))]
                ref_dbp_lstm30 = ref_dbp_lstm[:int((once_time/60)*len(ref_sbp_lstm))]
                est_dbp_lstm30 = est_dbp_lstm[:int((once_time/60)*len(est_dbp_lstm))]
                mae_bp1 = np.mean(np.abs(ref_sbp_lstm30 - est_sbp_lstm30))
                mae_bp2 = np.mean(np.abs(ref_dbp_lstm30 - est_dbp_lstm30))
                mae_bp3 = np.mean(np.abs(ref_bp_wave30 - est_bp_wave30))
                cc_wave_list30 += [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
                mae_lstm_list30 += [mae_bp1, mae_bp2, mae_bp3] 

                once_time = 60
                ref_bp_wave60 = ref_bp_wave[:int((once_time/60)*len(ref_bp_wave))]
                est_bp_wave60 = est_bp_wave[:int((once_time/60)*len(est_bp_wave))]
                ref_bp_wave_a60 = ref_bp_wave_a[:int((once_time/60)*len(ref_bp_wave_a))]
                ppg_wave_a60 = ppg_wave_a[:int((once_time/60)*len(ppg_wave_a))]
                ref_bp_aligned60 = ref_bp_aligned[:int((once_time/60)*len(ref_bp_aligned))]
                est_bp_aligned60 = est_bp_aligned[:int((once_time/60)*len(est_bp_aligned))]
                ori_ppg_wave60 = ori_ppg_wave[:int((once_time/60*len(ori_ppg_wave)))]

                correlation_matrix_1 = np.corrcoef(ref_bp_wave60, est_bp_wave60)
                correlation_coefficient_1 = correlation_matrix_1[0, 1]
                correlation_matrix_2 = np.corrcoef(ref_bp_wave_a60, ppg_wave_a60)
                correlation_coefficient_2 = correlation_matrix_2[0, 1]
                correlation_matrix_3 = np.corrcoef(ref_bp_aligned60, est_bp_aligned60)
                correlation_coefficient_3 = correlation_matrix_3[0, 1]
                correlation_matrix_4 = np.corrcoef(ref_bp_wave60, ori_ppg_wave60)
                correlation_coefficient_4 = correlation_matrix_4[0, 1]           

                ref_sbp_lstm60 = ref_sbp_lstm[:int((once_time/60)*len(ref_sbp_lstm))]
                est_sbp_lstm60 = est_sbp_lstm[:int((once_time/60)*len(est_sbp_lstm))]
                ref_dbp_lstm60 = ref_dbp_lstm[:int((once_time/60)*len(ref_sbp_lstm))]
                est_dbp_lstm60 = est_dbp_lstm[:int((once_time/60)*len(est_dbp_lstm))]
                mae_bp1 = np.mean(np.abs(ref_sbp_lstm60 - est_sbp_lstm60))
                mae_bp2 = np.mean(np.abs(ref_dbp_lstm60 - est_dbp_lstm60))
                mae_bp3 = np.mean(np.abs(ref_bp_wave60 - est_bp_wave60))
                cc_wave_list60 += [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
                mae_lstm_list60 += [mae_bp1, mae_bp2, mae_bp3] 

            cc_wave_df10.loc[len(cc_wave_df10)] = cc_wave_list10
            mae_lstm_df10.loc[len(mae_lstm_df10)] =mae_lstm_list10
            cc_wave_df30.loc[len(cc_wave_df30)] = cc_wave_list30
            mae_lstm_df30.loc[len(mae_lstm_df30)] =mae_lstm_list30
            cc_wave_df60.loc[len(cc_wave_df60)] = cc_wave_list60
            mae_lstm_df60.loc[len(mae_lstm_df60)] =mae_lstm_list60
            display(cc_wave_df10)
            display(mae_lstm_df10)
            display(cc_wave_df30)
            display(mae_lstm_df30)
            display(cc_wave_df60)
            display(mae_lstm_df60)
            SelectData(np.array([ref_sbp_lstm, est_sbp_lstm]).T).show_data()
            SelectData(np.array([ref_dbp_lstm, est_dbp_lstm]).T).show_data()
            SelectData(np.array([ref_bp_wave, est_bp_wave]).T).show_data()
            SelectData(np.array([ref_bp_aligned, est_bp_aligned]).T).show_data()
            SelectData(ppg_wave_a[:min_len_ppg]).show_data()
            SelectData(ref_bp_wave_a[:min_len_ppg]).show_data()
        cc_wave_df10.to_csv(f'cc_wave10_{len(patient_list)}.csv')
        mae_lstm_df10.to_csv(f'mae_bp10_{len(patient_list)}.csv')
        cc_wave_df30.to_csv(f'cc_wave30_{len(patient_list)}.csv')
        mae_lstm_df30.to_csv(f'mae_bp30_{len(patient_list)}.csv')
        cc_wave_df60.to_csv(f'cc_wave60_{len(patient_list)}.csv')
        mae_lstm_df60.to_csv(f'mae_bp60_{len(patient_list)}.csv')
        return 
    patient_list = get_patient_list(120)


    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import TensorBoard

    def create_sequences(X, y, time_steps=10):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X.iloc[i:(i + time_steps)])
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)  

    def bp_lstm_model(train_features_df, train_bp):
        time_steps = 10
        X_train, y_train_ori = create_sequences(train_features_df, pd.DataFrame(train_bp), time_steps)
        y_train = np.around(((y_train_ori - y_train_ori.min()) / (y_train_ori.max() - y_train_ori.min())), decimals=4)
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(time_steps, 12)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=200, verbose=1)
        return model

    def get_bp_point(test_features_df, test_bp, model):
        time_steps = 10
        X_test, y_test = create_sequences(test_features_df, pd.DataFrame(test_bp), time_steps)
        y_pred = model.predict(X_test)
        y_pred_recover = (y_pred * (y_test.max() - y_test.min())) + y_test.min()
        return y_test, y_pred_recover

    def bp_wave_lstm_modle(df_train):
        time_steps = 10
        X_train, y_train_ori = create_sequences(df_train[['PPG', 'ECG']], df_train['BP'], time_steps)
        y_train = np.around(((y_train_ori - y_train_ori.min()) / (y_train_ori.max() - y_train_ori.min())), decimals=4)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 2)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, verbose=1)
        return model

    def get_bp_wave(df_test, model):
        time_steps = 10
        X_test, y_test = create_sequences(df_test[['PPG', 'ECG']], df_test['BP'], time_steps)
        y_pred = model.predict(X_test)
        y_pred_recover = (y_pred * (y_test.max() - y_test.min())) + y_test.min()
        return y_test, y_pred_recover


    def get_patient_list(patient_num):
        pat = 0
        pat_list = []

        while len(pat_list) < (patient_num*1.5):
            datapat = all_data[pat]
            if len(datapat) > 45000:
                pat_list.append(pat)
            pat += 1
        print(pat_list)

        patient_list = []
        data_ready = pd.DataFrame()
        for i in pat_list:
            patient_array = SelectData(all_data[i])
            ppg, bp, ecg = patient_array.signal_data()
            ppg_filtered = nk.signal_filter(ppg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
            ppg_standardized = np.around(((ppg_filtered - ppg_filtered.min()) / (ppg_filtered.max() - ppg_filtered.min())), decimals=4)
            ecg_filtered = nk.signal_filter(ecg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
            ecg_standardized = np.around(((ecg_filtered - ecg_filtered.min()) / (ecg_filtered.max() - ecg_filtered.min())), decimals=4)
            ppg_aligned, bp_aligned = align_first_peaks(ppg_standardized, bp)
            cleaned_df = cleaned_data(ppg_aligned, bp_aligned, ecg_standardized)
            if len(cleaned_df) > 45000:
                patient_list.append(i)
                data_ready = pd.concat([data_ready, cleaned_df[:22500]])
        data_ready.to_csv(f'wave_train_data{patient_list[0]}_{patient_list[-1]}.csv')
        train_sbp, train_dbp, train_features_df = get_bp_features(data_ready)
        train_features_df.to_csv(f'fearutes_train_data{patient_list[0]}_{patient_list[-1]}.csv')
        print(len(patient_list))
        return patient_list

    ## 1* 30 (wave, ppg, ecg) cc
    ## 10min/patient
    def cc_waveppgecg(patient_list):
        wave_ppg_ecg_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ecg_cc', 'refbp_estbp_md', 'refbp_ppg_md', 'refbp_ecg_md', 'refbp_estbp_sd', 'refbp_ppg_sd', 'refbp_ecg_sd')))
        for patient in patient_list:
            patient_array = SelectData(all_data[patient])
            ppg, bp, ecg = patient_array.signal_data()
            ppg_filtered = nk.signal_filter(ppg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
            ppg_standardized = np.around(((ppg_filtered - ppg_filtered.min()) / (ppg_filtered.max() - ppg_filtered.min())), decimals=4)
            ecg_filtered = nk.signal_filter(ecg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
            ecg_standardized = np.around(((ecg_filtered - ecg_filtered.min()) / (ecg_filtered.max() - ecg_filtered.min())), decimals=4)
            ppg_aligned, bp_aligned = align_first_peaks(ppg_standardized, bp)
            cleaned_df = cleaned_data(ppg_aligned, bp_aligned, ecg_standardized)
            bp_wave_lstm_model = bp_wave_lstm_modle(cleaned_df[:22500])
            sampling_rate = 125
            once_time = 30
            time_steps = 10
            ref_bp_wave, est_bp_wave = get_bp_wave(cleaned_df[22500:(22500+30*sampling_rate)], bp_wave_lstm_model)
            est_bp_wave = np.squeeze(est_bp_wave)
            ref_bp_aligned, est_bp_aligned = align_first_peaks(ref_bp_wave, np.squeeze(est_bp_wave))
            ref_bp_aligned, est_bp_aligned = align_first_peaks(ref_bp_aligned, np.squeeze(est_bp_aligned))
            min_len_est = min(len(ref_bp_aligned), len(est_bp_aligned))
            ref_bp_aligned = ref_bp_aligned[:min_len_est]
            est_bp_aligned = est_bp_aligned[:min_len_est]
            ppg_wave = np.squeeze(np.array(cleaned_df[22500+time_steps:(22500+time_steps+len(ref_bp_wave))]['PPG']))
            ref_bp_wave_a, ppg_wave_a = align_first_peaks(ref_bp_wave, ppg_wave)
            min_len_ppg = min(len(ref_bp_wave_a), len(ppg_wave_a))
            ref_bp_wave_a = ref_bp_wave_a[:min_len_ppg]
            ppg_wave_a = ppg_wave_a[:min_len_ppg]
            ecg_wave = np.squeeze(np.array(cleaned_df[22500+time_steps:(22500+time_steps+len(ref_bp_wave))]['ECG']))

            correlation_matrix_1 = np.corrcoef(ref_bp_wave, est_bp_wave)
            correlation_coefficient_1 = correlation_matrix_1[0, 1]
            correlation_matrix_2 = np.corrcoef(ref_bp_wave_a, ppg_wave_a)
            correlation_coefficient_2 = correlation_matrix_2[0, 1]
            correlation_matrix_3 = np.corrcoef(ref_bp_wave, ecg_wave)
            correlation_coefficient_3 = correlation_matrix_3[0, 1]     
            cc_list = [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3]
            md1 = np.mean(ref_bp_wave - est_bp_wave)
            md2 = np.mean(ref_bp_wave_a - ppg_wave_a)
            md3 = np.mean(ref_bp_wave - ecg_wave)
            md_list = [md1, md2, md3]
            sd1 = np.std(ref_bp_wave - est_bp_wave)
            sd2 = np.std(ref_bp_wave_a - ppg_wave_a)
            sd3 = np.std(ref_bp_wave - ecg_wave)
            sd_list = [sd1, sd2, sd3]
            wave_ppg_ecg_list = cc_list + md_list + sd_list
            wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = wave_ppg_ecg_list

            display(wave_ppg_ecg_df)
            SelectData(np.array([ref_bp_wave, est_bp_wave]).T).show_data()
            SelectData(ref_bp_wave[1200:1800]).show_data()
            SelectData(est_bp_wave[1200:1800]).show_data()
            SelectData(ppg_wave_a[1200:1800]).show_data()
            SelectData(ecg_wave[1200:1800]).show_data()

        wave_ppg_ecg_df.to_csv(f'wave_ppg_ecg_{len(patient_list)}.csv')
        return

    patient_array = SelectData(all_data[1])
    ppg, bp, ecg = patient_array.signal_data()
    ppg_filtered = nk.signal_filter(ppg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
    ppg_standardized = np.around(((ppg_filtered - ppg_filtered.min()) / (ppg_filtered.max() - ppg_filtered.min())), decimals=4)
    ecg_filtered = nk.signal_filter(ecg, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=125)
    ecg_standardized = np.around(((ecg_filtered - ecg_filtered.min()) / (ecg_filtered.max() - ecg_filtered.min())), decimals=4)
    ppg_aligned, bp_aligned = align_first_peaks(ppg_standardized, bp)
    cleaned_df = cleaned_data(ppg_aligned, bp_aligned, ecg_standardized)
    bp_wave_lstm_model = bp_wave_lstm_modle(cleaned_df[:22500])
    sampling_rate = 125
    once_time = 30
    time_steps = 10
    ref_bp_wave, est_bp_wave = get_bp_wave(cleaned_df[22500:(22500+30*sampling_rate)], bp_wave_lstm_model)
    est_bp_wave = np.squeeze(est_bp_wave)
    ref_bp_aligned, est_bp_aligned = align_first_peaks(ref_bp_wave, np.squeeze(est_bp_wave))
    ref_bp_aligned, est_bp_aligned = align_first_peaks(ref_bp_aligned, np.squeeze(est_bp_aligned))
    min_len_est = min(len(ref_bp_aligned), len(est_bp_aligned))
    ref_bp_aligned = ref_bp_aligned[:min_len_est]
    est_bp_aligned = est_bp_aligned[:min_len_est]
    ppg_wave = np.squeeze(np.array(cleaned_df[22500+time_steps:(22500+time_steps+len(ref_bp_wave))]['PPG']))
    ref_bp_wave_a, ppg_wave_a = align_first_peaks(ref_bp_wave, ppg_wave)
    min_len_ppg = min(len(ref_bp_wave_a), len(ppg_wave_a))
    ref_bp_wave_a = ref_bp_wave_a[:min_len_ppg]
    ppg_wave_a = ppg_wave_a[:min_len_ppg]
    ecg_wave = np.squeeze(np.array(cleaned_df[22500+time_steps:(22500+time_steps+len(ref_bp_wave))]['ECG']))

    correlation_matrix_1 = np.corrcoef(ref_bp_wave, est_bp_wave)
    correlation_coefficient_1 = correlation_matrix_1[0, 1]
    correlation_matrix_2 = np.corrcoef(ref_bp_wave_a, ppg_wave_a)
    correlation_coefficient_2 = correlation_matrix_2[0, 1]
    correlation_matrix_3 = np.corrcoef(ref_bp_wave, ecg_wave)
    correlation_coefficient_3 = correlation_matrix_3[0, 1]     
    cc_list = [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3]
    md1 = np.mean(ref_bp_wave - est_bp_wave)
    md2 = np.mean(ref_bp_wave_a - ppg_wave_a)
    md3 = np.mean(ref_bp_wave - ecg_wave)
    md_list = [md1, md2, md3]
    sd1 = np.std(ref_bp_wave - est_bp_wave)
    sd2 = np.std(ref_bp_wave_a - ppg_wave_a)
    sd3 = np.std(ref_bp_wave - ecg_wave)
    sd_list = [sd1, sd2, sd3]
    wave_ppg_ecg_list = cc_list + md_list + sd_list

    fig = plt.figure(figsize=(30,6), dpi=300)
    plt.plot(ref_bp_wave[1200:1800], label='ref BP wave')
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(30,6), dpi=300)
    plt.plot(est_bp_wave[1200:1800], label='est BP wave')
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(30,6), dpi=300)
    plt.plot(ppg_wave_a[1200:1800], label='PPG aligned')
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(30,6), dpi=300)
    plt.plot(ecg_wave[1200:1800], label='ECG')
    plt.legend()
    plt.show()

    wave_ppg_ecg_df80 = pd.concat([pd.read_csv('wave_ppg_ecg10.csv'), pd.read_csv('wave_ppg_ecg_70.csv')])
    # wave_ppg_ecg_df = wave_ppg_ecg_df.reset_index(drop=True)

    wave_ppg_ecg_df80.to_csv('wave_ppg_ecg80.csv')
    def cc_histogram(wave_ppg_ecg_df):
        plt.figure(figsize=(15, 6), dpi=300)
        # Set your color palette
        colors = ['#B0C4DE', '#5F9EA0', '#C6868E']

        # Specify data
        data1 = np.array(wave_ppg_ecg_df['refbp_estbp_cc'])
        data2 = np.array(wave_ppg_ecg_df['refbp_ppg_cc'])
        data3 = np.array(wave_ppg_ecg_df['refbp_ecg_cc'])

        # Create bins and histogram
        bins = np.linspace(0, 1, 11)
        counts1, _ = np.histogram(abs(data1), bins=bins)
        counts2, _ = np.histogram(abs(data2), bins=bins)
        counts3, _ = np.histogram(abs(data3), bins=bins)

        # Calculate frequencies
        freq1 = counts1 / len(data1)
        freq2 = counts2 / len(data2)
        freq3 = counts3 / len(data3)

        barWidth = 0.25
        r1 = np.arange(len(freq1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]

        plt.bar(r1, freq1, color=colors[0], width=barWidth, edgecolor='grey', label='refBP_estBP')
        plt.bar(r2, freq2, color=colors[1], width=barWidth, edgecolor='grey', label='refBP_PPG')
        plt.bar(r3, freq3, color=colors[2], width=barWidth, edgecolor='grey', label='refBP_ECG')

        # Adding labels
        for i in range(len(r1)):
            plt.text(x = r1[i] + barWidth/2 - 0.1 , y = freq1[i] + 0.02, s = f"{counts1[i]}", size = 10, ha='center')
            plt.text(x = r2[i] + barWidth/2 - 0.1 , y = freq2[i] + 0.02, s = f"{counts2[i]}", size = 10, ha='center')
            plt.text(x = r3[i] + barWidth/2 - 0.1 , y = freq3[i] + 0.02, s = f"{counts3[i]}", size = 10, ha='center')

        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.xticks([r + barWidth for r in range(len(freq1))], ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
        plt.legend()
        plt.show()
        return
    return









def second_version():
    fs =125 # Sample rate in Hz
    bp_standard_rate = 200

    def remove_worse_patient(patient_list, number_remove_list):
        for number in number_remove_list:
            if number in patient_list:
                patient_list.remove(number)
        return patient_list

    import numpy as np
    def straighten_ecg(ecg_signal):
        detrended_ecg = np.subtract(ecg_signal, np.mean(ecg_signal))
        return detrended_ecg
        
    import numpy as np
    def normalize_sinal(ppg):
    # Assuming ppg_signal and ecg_signal are your original PPG and ECG signals
        ppg_min = np.min(ppg)
        ppg_max = np.max(ppg)
        normalized_ppg = (ppg - ppg_min) / (ppg_max - ppg_min)
        return normalized_ppg

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    def align_ppgbp(ppg_signal, bp_signal1, bp_signal2, ecg_signal, show=0): ## ppg_signal = ppg_normalized, bp_signal = bp_normalized, ecg_signal = ecg_normalized; get ppg_aligned, bp_aligned
        try:
            ppg_peaks, _ = find_peaks(ppg_signal, height=0.5)  # Adjust the height threshold as needed
            bp_peaks, _ = find_peaks(bp_signal1, height=0.1)
            ecg_peaks, _ = find_peaks(ecg_signal, height=0.65)
            #print(f'ppg peaks: {len(ppg_peaks)} {ppg_peaks}')
            #print(f'ecg peaks: {len(ecg_peaks)} {ecg_peaks}')

            first_ecg_peak = ecg_peaks[0]
            #print(f'first ecg peak: {first_ecg_peak}')
            indices_ppg = np.argwhere(ppg_peaks[:10] > first_ecg_peak)
            first_ppg_peak = ppg_peaks[int(indices_ppg[0])]
            #print(f'first ppg peak: {first_ppg_peak}')
            indices_bp = np.argwhere(bp_peaks[:10] > first_ecg_peak)
            if len(indices_bp) > 0:
                first_bp_peak = bp_peaks[int(indices_bp[0])]
                #print(f'first bp peak: {first_bp_peak}')
                ppg_bp_peaks_subtraction = abs(bp_peaks[int(indices_bp[0]):int(indices_bp[0])+5] - ppg_peaks[int(indices_ppg[0]):int(indices_ppg[0])+5])
                #print(ppg_bp_peaks_subtraction)
                distance_ppgbp = np.bincount(ppg_bp_peaks_subtraction).argmax()
                #print(distance_ppgbp)
                #print(bp_peaks[int(indices_bp[0]):int(indices_bp[0])+20] - ppg_peaks[int(indices_ppg[0]):int(indices_ppg[0])+20])
                #print(f'distance:{distance_ppgbp}')
                if first_bp_peak > first_ppg_peak:
                    bp_aligned = bp_signal1[distance_ppgbp:]
                    bp_ori_aligned = bp_signal2[distance_ppgbp:]
                    ppg_aligned = ppg_signal
                elif first_bp_peak < first_ppg_peak:
                    bp_aligned = bp_signal1
                    bp_ori_aligned = bp_signal2
                    ppg_aligned = ppg_signal[distance_ppgbp:]
                else:
                    bp_aligned = bp_signal1
                    bp_ori_aligned = bp_signal2
                    ppg_aligned = ppg_signal
                #print(f'ppg len: {len(ppg_aligned)}')
                #print(f'bp len: {len(bp_aligned)}')
                min_len = min(len(bp_aligned), len(ppg_aligned))
                ppg_aligned = ppg_aligned[:min_len]
                bp_aligned = bp_aligned[:min_len]
                bp_ori_aligned = bp_ori_aligned[:min_len]
                ecg_aligned = ecg_signal[:min_len]

                if show == 1:
                    plt.figure(figsize=(30, 6))
                    plt.plot(ppg_signal, label='PPG')
                    plt.plot(bp_signal1, label='BP')
                    plt.plot(ecg_signal, label='ECG')
                    plt.scatter(ppg_peaks, ppg_signal[ppg_peaks], color='c', marker='o', label='Aligned PPG Peaks')
                    plt.scatter(bp_peaks, bp_signal1[bp_peaks], color='orange', marker='o', label='Aligned BP Peaks')
                    plt.scatter(ecg_peaks, ecg_signal[ecg_peaks], color='green', marker='o', label='Aligned BP Peaks')
                    plt.xlabel('Time')
                    plt.ylabel('Amplitude')
                    plt.legend()
                    plt.show()

                    plt.figure(figsize=(30, 6))
                    plt.plot(ppg_aligned, label='PPG')
                    plt.plot(bp_aligned, label='BP')
                    plt.plot(ecg_aligned, label='ECG')
                    plt.xlabel('Time')
                    plt.ylabel('Amplitude')
                    plt.legend()
                    plt.show()
                return ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned
            else:
                raise IndexError
        except IndexError:
            raise IndexError


    # Reshape
    def reshape_data(X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps), :])
            ys.append(y[i+time_steps-1])
        return np.array(Xs), np.array(ys) 

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from keras.losses import mean_squared_error

    def custom_loss(y_true, y_pred):
        loss = mean_squared_error(y_true*250, y_pred*250)
        return loss

    def bpwave_lstm_model(X_train, y_train, time_steps):
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(time_steps, X_train.shape[-1])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss=custom_loss)
        model.fit(X_train, y_train, epochs=200, verbose=1)
        return model

    import matplotlib.pyplot as plt
    def show_one(signal1):
        fig = plt.figure(figsize=(30,6))
        plt.plot(signal1)
        return plt.show()
    def show_two(signal1, signal2):
        fig = plt.figure(figsize=(30,6))
        plt.plot(signal1, label='1')
        plt.plot(signal2, label='2')
        plt.legend()
        return plt.show()
    def show_three(signal1, signal2, signal3):
        fig = plt.figure(figsize=(30,6))
        plt.plot(signal1, label='1')
        plt.plot(signal2, label='2')
        plt.plot(signal3, label='3')
        plt.legend()
        return plt.show()

    ## read the 1000 patients data
    import scipy.io
    data1 = scipy.io.loadmat('/Users/jinyanwei/Desktop/BP_Model/Data/Cuffless_BP_Estimation/part_1.mat')
    for patient in range(1):
        patient_data = data1['p'][0, patient][0, :]
        if len(patient_data) > 4000:
            #print(patient)
            patient_data = data1['p'][0, patient][:, :4000]
            ppg_ori = patient_data[0]
            bp_ori = patient_data[1]
            ecg_ori = patient_data[2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_segmented, bp_segmented, bp_ori_segmented, ecg_segmented = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
    patients_list = [1,3,5,16,23,24,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,44,45,46,63,67,70,71,73,76,78,79,80,81,82,103,104,120,128,129,151,156,158,161,170,172,173,183,186,187,192,193,194,196,197,198,199,200,201,203,206,207,213,214,218,220,221,227,229,236,243,248,253,254,284,286,295,298,299,303,304,305,317,336,338,343,344,346,348,350,351,355,356,357,368,369,371,372,403,412]
    #print(len(patients_list))   
    ### 100patients
    train_ppgs = []
    train_ecgs = []
    train_bps = []
    train_bp_oris = []
    test_ppgs = []
    test_ori_ppgs = []
    test_ecgs = []
    test_bps = []
    test_bp_oris = []
    for patient in patients_list[:-20]:
        patient_data = data1['p'][0,patient][:,:4000]
        ppg_ori = patient_data[0]
        bp_ori = patient_data[1]
        ecg_ori = patient_data[2]
        ecg_detrened = straighten_ecg(ecg_ori)
        ppg_normalized = normalize_sinal(ppg_ori)
        bp_standarded = bp_ori / bp_standard_rate
        ecg_normalized = normalize_sinal(ecg_detrened)
        ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
        train_ppgs.append(ppg_aligned)
        train_ecgs.append(ecg_aligned)
        train_bps.append(bp_aligned)
        train_bp_oris.append(bp_ori_aligned)
    for patient in patients_list[-20:]:
        patient_data = data1['p'][0,patient][:,:4000]
        ppg_ori = patient_data[0]
        bp_ori = patient_data[1]
        ecg_ori = patient_data[2]
        ecg_detrened = straighten_ecg(ecg_ori)
        ppg_normalized = normalize_sinal(ppg_ori)
        bp_standarded = bp_ori / bp_standard_rate
        ecg_normalized = normalize_sinal(ecg_detrened)
        ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
        train_ppgs.append(ppg_aligned[:int(0.2*len(ppg_aligned))])
        train_ecgs.append(ecg_aligned[:int(0.2*len(ecg_aligned))])
        train_bps.append(bp_aligned[:int(0.2*len(bp_aligned))]) 
        train_bp_oris.append(bp_ori_aligned[:int(0.2*len(bp_ori_aligned))]) 
        test_ppgs.append(ppg_aligned[int(0.2*len(ppg_aligned)):])
        ppg_ori = ppg_ori[:len(ppg_aligned)]
        test_ori_ppgs.append(ppg_ori[int(0.2*len(ppg_ori)):])
        test_ecgs.append(ecg_aligned[int(0.2*len(ecg_aligned)):])
        test_bps.append(bp_aligned[int(0.2*len(bp_aligned)):])  
        test_bp_oris.append(bp_ori_aligned[int(0.2*len(bp_ori_aligned)):])  
    train_ppg = np.concatenate(train_ppgs, axis=0)
    train_ecg = np.concatenate(train_ecgs, axis=0)
    train_bp = np.concatenate(train_bps, axis=0)
    train_bp_ori = np.concatenate(train_bp_oris, axis=0)
    train_feature = np.column_stack((train_ppg, train_ecg)) 

    '''### Model
    time_steps = 15
    X_train, y_train = reshape_data(train_feature,train_bp,time_steps)
    bpwave_lstm_model = bpwave_lstm_model(X_train, y_train, time_steps)
    bpwave_lstm_model.save('bpwave_lstm_model100.h5')

    ### CC
    #from keras.models import load_model
    #bpwave_lstm_model = load_model('lstm_model.h5')
    import pandas as pd
    pred_bps = []
    cc_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ecg_cc', 'refbp_estbp_md', 'refbp_estbp_sd')))

    for i in range(len(test_bps)):
        test_ppg = test_ppgs[i]
        test_ecg = test_ecgs[i]
        test_feature = np.column_stack((test_ppg, test_ecg))
        X_test, y_test = reshape_data(test_feature, test_bps[i], time_steps)
        y_pred = bpwave_lstm_model.predict(X_test)
        y_pred = y_pred.flatten()
        test_ppg = test_ppg[time_steps-1:-1]
        test_ecg = test_ecg[time_steps-1:-1]
        #print(y_test.shape, y_pred.shape, test_ppg.shape, test_ecg.shape)
        pred_bps.append(y_pred)
        md = np.mean(y_pred*250-y_test*250)
        std = np.std(y_pred*250-y_test*250)
        cc_refbp_estbp = np.abs(np.corrcoef(y_test, y_pred)[0, 1])
        #print(cc_refbp_estbp)
        cc_refbp_ppg = np.abs(np.corrcoef(y_test, test_ppg)[0, 1])
        cc_refbp_ecg = np.abs(np.corrcoef(y_test, test_ecg)[0, 1])
        cc_df.loc[len(cc_df)] = [cc_refbp_estbp, cc_refbp_ppg, cc_refbp_ecg, md, std]

    cc_df.loc[len(cc_df)] = [np.mean(cc_df['refbp_estbp_cc']), np.mean(cc_df['refbp_ppg_cc']), np.mean(cc_df['refbp_ecg_cc']), np.mean(cc_df['refbp_estbp_md']), np.mean(cc_df['refbp_estbp_sd'])]
    cc_df = cc_df.rename(index={len(cc_df)-1: 'ave'})
    cc_df.to_csv('100bpppgecgcc20.csv')'''

    # wave 20 test
    import pandas as pd
    wave_ppg_ecg_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ori_ppg_cc', 'refbp_ecg_cc')))
    listselect = [1, 12, 9, 11]
    for i in listselect:#len(test_bp)
        X_test = test_features_wave[i].T
        bp_test = test_bp[i]
        bp_test_ori = test_bp_ori[i]
        X_test = X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test
        bp_pred_randomforest = randomforest_model_bp80.predict(X_test)
        # print(len(bp_pred_randomforest), bp_pred_randomforest)
        bp_pred_ori_randomforest = (bp_pred_randomforest * (max(bp_test_ori) - min(bp_test_ori))) + min(bp_test_ori)
        '''print(len(bp_pred_ori_randomforest), bp_pred_ori_randomforest)
        plt.figure(figsize=(30,6))
        plt.plot(bp_pred_randomforest)
        plt.plot(test_features_wave[i][0])
        plt.show()
        print(len(bp_test_ori), bp_test_ori)
        print(len(test_features_wave[i][0]), test_features_wave[i][0])
        print(len(ppg_signal[:len(bp_pred_ori_randomforest)]), ppg_signal[:len(bp_pred_ori_randomforest)])
        print(len(test_features_wave[i][1]), test_features_wave[i][1])'''
        print(i)
        
        correlation_matrix_1 = np.corrcoef(bp_test_ori, bp_pred_ori_randomforest)
        correlation_coefficient_1 = correlation_matrix_1[0, 1]
        correlation_matrix_2 = np.corrcoef(test_features_wave[i][0], bp_pred_ori_randomforest)
        correlation_coefficient_2 = correlation_matrix_2[0, 1]
        ppg_signal = data['p'][0][patient60s_saved1[400+i]][:, :7500][0]
        correlation_matrix_3 = np.corrcoef(ppg_signal[:len(bp_pred_ori_randomforest)], bp_pred_ori_randomforest)
        correlation_coefficient_3 = correlation_matrix_3[0, 1]
        correlation_matrix_4 = np.corrcoef(test_features_wave[i][1], bp_pred_ori_randomforest)
        correlation_coefficient_4 = correlation_matrix_4[0, 1]
        cc_list = [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
        wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = cc_list
        print(cc_list)
        plt.figure(figsize=(15,6))
        plt.plot(bp_test_ori, label = 'ref BP')
        plt.plot(bp_pred_ori_randomforest, label = 'est BP')
        plt.legend(loc='upper center', prop={'size': 14})
        plt.title(f'CC between ref BP and est BP is {round(cc_list[0], 4)}', fontsize = 18)
        plt.show()
    #wave_ppg_ecg_df.to_csv('wave_ppg_ecg_random_20test_30s_21Jul.csv')

    ### CC add ori ppg
    #from keras.models import load_model
    #bpwave_lstm_model = load_model('lstm_model.h5')
    time_steps = 15
    def custom_loss(y_true, y_pred):
        loss = mean_squared_error(y_true*250, y_pred*250)
        return loss

    import tensorflow as tf
    bpwave_lstm_model = tf.keras.models.load_model('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/bpwave_lstm_model100.h5', custom_objects={'custom_loss': custom_loss})
    import pandas as pd
    pred_bps = []
    #cc_df = pd.DataFrame(columns=['refbp_estbp_cc', 'estbp_ppg_cc', 'estbp_ori_ppg_cc' 'estbp_ecg_cc', 'refbp_estbp_md', 'refbp_estbp_sd'])
    cc_df = pd.DataFrame(columns=['cc_refbp_estbp', 'cc_estbp_ppg', 'cc_estbp_ori_ppg', 'cc_estbp_ecg', 'md', 'std'])
    listselect = [9,5,19,17]
    for i in listselect:
        test_ppg = test_ppgs[i]
        test_ori_ppg = test_ori_ppgs[i]
        test_ecg = test_ecgs[i]
        test_feature = np.column_stack((test_ppg, test_ecg))
        X_test, y_test = reshape_data(test_feature, test_bps[i], time_steps)
        y_pred = bpwave_lstm_model.predict(X_test)
        y_pred = y_pred.flatten()
        test_ppg = test_ppg[time_steps-1:-1]
        test_ori_ppg = test_ori_ppg[time_steps-1:-1]
        test_ecg = test_ecg[time_steps-1:-1]
        #print(y_test.shape, y_pred.shape, test_ppg.shape, test_ecg.shape)
        pred_bps.append(y_pred)
        md = np.mean(y_pred*250-y_test*250)
        std = np.std(y_pred*250-y_test*250)
        cc_refbp_estbp = np.abs(np.corrcoef(y_test, y_pred)[0, 1])
        #print(cc_refbp_estbp)
        cc_estbp_ppg = np.abs(np.corrcoef(y_test, test_ppg)[0, 1])
        cc_estbp_ori_ppg = np.abs(np.corrcoef(y_test, test_ori_ppg)[0, 1])
        cc_estbp_ecg = np.abs(np.corrcoef(y_test, test_ecg)[0, 1])
        cc_df.loc[len(cc_df)] = [cc_refbp_estbp, cc_estbp_ppg, cc_estbp_ori_ppg, cc_estbp_ecg, md, std]
        print(f'cc_refbp_estbp: {cc_refbp_estbp}')
        plt.figure(figsize=(15,6), dpi=300)
        plt.plot(list(y_test*250)[:500], label = 'ref BP')
        plt.plot(list(y_pred*250+20)[:500], label = 'est BP')
        plt.legend(loc='upper center', prop={'size': 14})
        plt.title(f'CC between ref BP and est BP is {round(cc_refbp_estbp, 4)}', fontsize = 18)
        plt.show()
    #cc_df.loc[len(cc_df)] = [np.mean(cc_df['cc_refbp_estbp']), np.mean(cc_df['cc_estbp_ppg']), np.mean(cc_df['cc_estbp_ori_ppg']), np.mean(cc_df['cc_estbp_ecg']), np.mean(cc_df['md']), np.mean(cc_df['std'])]
    #cc_df = cc_df.rename(index={len(cc_df)-1: 'ave'})
    #cc_df.to_csv('100bpppgecgcc20_21July.csv')
    ### CC add ori ppg
    #from keras.models import load_model
    #bpwave_lstm_model = load_model('lstm_model.h5')
    time_steps = 15
    def custom_loss(y_true, y_pred):
        loss = mean_squared_error(y_true*250, y_pred*250)
        return loss

    import tensorflow as tf
    bpwave_lstm_model = tf.keras.models.load_model('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/bpwave_lstm_model100.h5', custom_objects={'custom_loss': custom_loss})
    import pandas as pd
    pred_bps = []
    #cc_df = pd.DataFrame(columns=['refbp_estbp_cc', 'estbp_ppg_cc', 'estbp_ori_ppg_cc' 'estbp_ecg_cc', 'refbp_estbp_md', 'refbp_estbp_sd'])
    cc_df = pd.DataFrame(columns=['cc_refbp_estbp', 'cc_estbp_ppg', 'cc_estbp_ori_ppg', 'cc_estbp_ecg', 'md', 'std'])

    for i in range(len(test_bps)):
        test_ppg = test_ppgs[i]
        test_ori_ppg = test_ori_ppgs[i]
        test_ecg = test_ecgs[i]
        test_feature = np.column_stack((test_ppg, test_ecg))
        X_test, y_test = reshape_data(test_feature, test_bps[i], time_steps)
        y_pred = bpwave_lstm_model.predict(X_test)
        y_pred = y_pred.flatten()
        test_ppg = test_ppg[time_steps-1:-1]
        test_ori_ppg = test_ori_ppg[time_steps-1:-1]
        test_ecg = test_ecg[time_steps-1:-1]
        #print(y_test.shape, y_pred.shape, test_ppg.shape, test_ecg.shape)
        pred_bps.append(y_pred)
        md = np.mean(y_pred*250-y_test*250)
        std = np.std(y_pred*250-y_test*250)
        cc_refbp_estbp = np.abs(np.corrcoef(y_test, y_pred)[0, 1])
        #print(cc_refbp_estbp)
        cc_estbp_ppg = np.abs(np.corrcoef(y_test, test_ppg)[0, 1])
        cc_estbp_ori_ppg = np.abs(np.corrcoef(y_test, test_ori_ppg)[0, 1])
        cc_estbp_ecg = np.abs(np.corrcoef(y_test, test_ecg)[0, 1])
        cc_df.loc[len(cc_df)] = [cc_refbp_estbp, cc_estbp_ppg, cc_estbp_ori_ppg, cc_estbp_ecg, md, std]
    cc_df.loc[len(cc_df)] = [np.mean(cc_df['cc_refbp_estbp']), np.mean(cc_df['cc_estbp_ppg']), np.mean(cc_df['cc_estbp_ori_ppg']), np.mean(cc_df['cc_estbp_ecg']), np.mean(cc_df['md']), np.mean(cc_df['std'])]
    cc_df = cc_df.rename(index={len(cc_df)-1: 'ave'})
    cc_df.to_csv('100bpppgecgcc20_21July.csv')

    ### adjust time_steps
    time_steps = 15
    feature0 = np.column_stack((train_ppgs[0], train_ecgs[0])) 
    bp0 = train_bps[0]
    X_train0, y_train0 = reshape_data(feature0[:int(0.8*len(feature0))],bp0[:int(0.8*len(bp0))],time_steps)
    model0 = bpwave_lstm_model(X_train0,y_train0,time_steps)

    X_test0, y_test0 = reshape_data(feature0[int(0.8*len(feature0)):],bp0[int(0.8*len(bp0)):],time_steps)
    y_pred0 = model0.predict(X_test0)
    mae = np.mean(np.abs(y_pred0*bp_standard_rate - y_test0*bp_standard_rate))
    rmse = np.sqrt(np.mean((y_pred0*bp_standard_rate-y_test0*bp_standard_rate)**2))
    print(f'MAE: {mae}, RMSE: {rmse}')
    show_two(y_test0*bp_standard_rate, y_pred0*bp_standard_rate)

    ## read 3000 patients data
    import sys
    sys.path.append('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/')
    from read_data import open_data
    datab1 = open_data('/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_1.mat')
    datab2 = open_data('/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_2.mat')
    datab3 = open_data('/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_3.mat')
    datab4 = open_data('/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_4.mat')

    import pandas as pd
    all4patients_text = pd.read_csv('/Users/jinyanwei/Desktop/BP_Model/Data/UCI/lly_features.csv')
    part1patients = [int(string.split('_')[-1]) for string in all4patients_text['part1'].dropna()]
    part2patients = [int(string.split('_')[-1]) for string in all4patients_text['part2'].dropna()]
    part3patients = [int(string.split('_')[-1]) for string in all4patients_text['part3'].dropna()]
    part4patients = [int(string.split('_')[-1]) for string in all4patients_text['part4'].dropna()]

    part1remove = [2335,41,82,1749,744,2040,5,1159,42,1776,553,219,1174,1]
    part1patients = remove_worse_patient(part1patients, part1remove)
    part2remove = []
    part2patients = remove_worse_patient(part2patients,part2remove)
    part3remove = []
    part3patients = remove_worse_patient(part3patients,part3remove)
    part4remove = []
    part4patients = remove_worse_patient(part4patients,part4remove)

    ### 660patients
    train_ppgs = []
    train_ecgs = []
    train_bps = []
    train_bp_oris = []
    test_ppgs = []
    test_ecgs = []
    test_bps = []
    test_bp_oris = []
    for patient in part1patients[:-20]:
        if len(datab1[patient])>4000:
            print(f'part1{patient}')
            patient_data = datab1[patient][:,:4000]
            ppg_ori = patient_data[:,0]
            bp_ori = patient_data[:,1]
            ecg_ori = patient_data[:,2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
            train_ppgs.append(ppg_aligned)
            train_ecgs.append(ecg_aligned)
            train_bps.append(bp_aligned)
            train_bp_oris.append(bp_ori_aligned)  
    for patient in part2patients[:-20]:
        if len(datab2[patient])>4000:
            patient_data = datab2[patient][:,:4000]
            ppg_ori = patient_data[:,0]
            bp_ori = patient_data[:,1]
            ecg_ori = patient_data[:,2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
            train_ppgs.append(ppg_aligned)
            train_ecgs.append(ecg_aligned)
            train_bps.append(bp_aligned)
            train_bp_oris.append(bp_ori_aligned)
    for patient in part3patients[:-20]:
        if len(datab3[patient])>4000:
            patient_data = datab3[patient][:,:4000]
            ppg_ori = patient_data[:,0]
            bp_ori = patient_data[:,1]
            ecg_ori = patient_data[:,2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
            train_ppgs.append(ppg_aligned)
            train_ecgs.append(ecg_aligned)
            train_bps.append(bp_aligned)
            train_bp_oris.append(bp_ori_aligned)
    for patient in part4patients[:-20]:
        if len(datab4[patient])>4000:
            patient_data = datab4[patient][:,:4000]
            ppg_ori = patient_data[:,0]
            bp_ori = patient_data[:,1]
            ecg_ori = patient_data[:,2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
            train_ppgs.append(ppg_aligned)
            train_ecgs.append(ecg_aligned)
            train_bps.append(bp_aligned)
            train_bp_oris.append(bp_ori_aligned) 
    for patient in part1patients[-20:]:
        if len(datab1[patient])>4000:
            patient_data = datab1[patient][:,:4000]
            ppg_ori = patient_data[:,0]
            bp_ori = patient_data[:,1]
            ecg_ori = patient_data[:,2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
            train_ppgs.append(ppg_aligned[:int(0.2*len(ppg_aligned))])
            train_ecgs.append(ecg_aligned[:int(0.2*len(ecg_aligned))])
            train_bps.append(bp_aligned[:int(0.2*len(bp_aligned))]) 
            train_bp_oris.append(bp_ori_aligned[:int(0.2*len(bp_ori_aligned))]) 
            test_ppgs.append(ppg_aligned[int(0.2*len(ppg_aligned)):])
            test_ecgs.append(ecg_aligned[int(0.2*len(ecg_aligned)):])
            test_bps.append(bp_aligned[int(0.2*len(bp_aligned)):])  
            test_bp_oris.append(bp_ori_aligned[int(0.2*len(bp_ori_aligned)):])  
    for patient in part2patients[-20:]:
        if len(datab2[patient])>4000:
            patient_data = datab2[patient][:,:4000]
            ppg_ori = patient_data[:,0]
            bp_ori = patient_data[:,1]
            ecg_ori = patient_data[:,2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
            train_ppgs.append(ppg_aligned[:int(0.2*len(ppg_aligned))])
            train_ecgs.append(ecg_aligned[:int(0.2*len(ecg_aligned))])
            train_bps.append(bp_aligned[:int(0.2*len(bp_aligned))]) 
            train_bp_oris.append(bp_ori_aligned[:int(0.2*len(bp_ori_aligned))]) 
            test_ppgs.append(ppg_aligned[int(0.2*len(ppg_aligned)):])
            test_ecgs.append(ecg_aligned[int(0.2*len(ecg_aligned)):])
            test_bps.append(bp_aligned[int(0.2*len(bp_aligned)):])  
            test_bp_oris.append(bp_ori_aligned[int(0.2*len(bp_ori_aligned)):])  
    for patient in part3patients[-20:]:
        if len(datab3[patient])>4000:
            patient_data = datab3[patient][:,:4000]
            ppg_ori = patient_data[:,0]
            bp_ori = patient_data[:,1]
            ecg_ori = patient_data[:,2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
            train_ppgs.append(ppg_aligned[:int(0.2*len(ppg_aligned))])
            train_ecgs.append(ecg_aligned[:int(0.2*len(ecg_aligned))])
            train_bps.append(bp_aligned[:int(0.2*len(bp_aligned))]) 
            train_bp_oris.append(bp_ori_aligned[:int(0.2*len(bp_ori_aligned))]) 
            test_ppgs.append(ppg_aligned[int(0.2*len(ppg_aligned)):])
            test_ecgs.append(ecg_aligned[int(0.2*len(ecg_aligned)):])
            test_bps.append(bp_aligned[int(0.2*len(bp_aligned)):])  
            test_bp_oris.append(bp_ori_aligned[int(0.2*len(bp_ori_aligned)):])  
    for patient in part4patients[-20:]:
        if len(datab4[patient])>4000:
            patient_data = datab4[patient][:,:4000]
            ppg_ori = patient_data[:,0]
            bp_ori = patient_data[:,1]
            ecg_ori = patient_data[:,2]
            ecg_detrened = straighten_ecg(ecg_ori)
            ppg_normalized = normalize_sinal(ppg_ori)
            bp_standarded = bp_ori / bp_standard_rate
            ecg_normalized = normalize_sinal(ecg_detrened)
            ppg_aligned, bp_aligned, bp_ori_aligned, ecg_aligned = align_ppgbp(ppg_signal = ppg_normalized, bp_signal1 = bp_standarded, bp_signal2 = bp_ori, ecg_signal = ecg_normalized, show=0)
            train_ppgs.append(ppg_aligned[:int(0.2*len(ppg_aligned))])
            train_ecgs.append(ecg_aligned[:int(0.2*len(ecg_aligned))])
            train_bps.append(bp_aligned[:int(0.2*len(bp_aligned))]) 
            train_bp_oris.append(bp_ori_aligned[:int(0.2*len(bp_ori_aligned))]) 
            test_ppgs.append(ppg_aligned[int(0.2*len(ppg_aligned)):])
            test_ecgs.append(ecg_aligned[int(0.2*len(ecg_aligned)):])
            test_bps.append(bp_aligned[int(0.2*len(bp_aligned)):])  
            test_bp_oris.append(bp_ori_aligned[int(0.2*len(bp_ori_aligned)):])  
    train_ppg = np.concatenate(train_ppgs, axis=0)
    train_ecg = np.concatenate(train_ecgs, axis=0)
    train_bp = np.concatenate(train_bps, axis=0)
    train_bp_ori = np.concatenate(train_bp_oris, axis=0)
    train_feature = np.column_stack((train_ppg, train_ecg)) 

    ### Model
    time_steps = 15
    X_train, y_train = reshape_data(train_feature,train_bp,time_steps)
    bpwave_lstm_model = bpwave_lstm_model(X_train, y_train, time_steps)
    bpwave_lstm_model.save('bpwave_lstm_model660.h5')

    ### CC
    #from keras.models import load_model
    #bpwave_lstm_model = load_model('lstm_model.h5')
    import pandas as pd
    pred_bps = []
    cc_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ecg_cc', 'refbp_estbp_md', 'refbp_estbp_sd')))

    for i in range(len(test_bps)):
        test_ppg = test_ppgs[i]
        test_ecg = test_ecgs[i]
        test_feature = np.column_stack((test_ppg, test_ecg))
        X_test, y_test = reshape_data(test_feature, test_bps[i], time_steps)
        y_pred = bpwave_lstm_model.predict(X_test)
        y_pred = y_pred.flatten()
        test_ppg = test_ppg[time_steps-1:-1]
        test_ecg = test_ecg[time_steps-1:-1]
        #print(y_test.shape, y_pred.shape, test_ppg.shape, test_ecg.shape)
        pred_bps.append(y_pred)
        md = np.mean(y_pred*bp_standard_rate-y_test*bp_standard_rate)
        std = np.std(y_pred*bp_standard_rate-y_test*bp_standard_rate)
        cc_refbp_estbp = np.abs(np.corrcoef(y_test, y_pred)[0, 1])
        #print(cc_refbp_estbp)
        cc_refbp_ppg = np.abs(np.corrcoef(y_test, test_ppg)[0, 1])
        cc_refbp_ecg = np.abs(np.corrcoef(y_test, test_ecg)[0, 1])
        cc_df.loc[len(cc_df)] = [cc_refbp_estbp, cc_refbp_ppg, cc_refbp_ecg, md, std]

    cc_df.loc[len(cc_df)] = [np.mean(cc_df['refbp_estbp_cc']), np.mean(cc_df['refbp_ppg_cc']), np.mean(cc_df['refbp_ecg_cc']), np.mean(cc_df['refbp_estbp_md']), np.mean(cc_df['refbp_estbp_sd'])]
    cc_df = cc_df.rename(index={len(cc_df)-1: 'ave'})
    cc_df.to_csv('660bpppgecgcc80.csv')
    return





def first_version():
    '''
    # BP wave 
    ## 100 patients (80 train + 20 test)
    ### 10s
    #### Random Forest Model

    #### LSTM Model
    ##### CC
    * ref bp -- est bp
    * ppg -- est bp
    * unaligned ppg -- est bp
    * ecg -- est bp

    ##### MAE
    * ref bp -- est bp



    ### 30s
    #### Random Forest Model

    #### LSTM Model

    ### 60s
    #### Random Forest Model

    #### LSTM Model


    ## 600 patients(500 train + 100 test)
    ### 30s
    #### Random Forest Model

    #### LSTM Model



    # BP point to point 

    ##### MAE
    * ref sbp -- est sbp
    * ref dbp -- ref dbp
    '''

    ## read the data
    import scipy.io
    data = scipy.io.loadmat('/Users/jinyanwei/Desktop/BP_Model/Data/Cuffless_BP_Estimation/part_1.mat')

    patient60s_list1 = [] 
    for i in range(data['p'].shape[1]):
        patient_data = data['p'][0][i] ## shapes like (3, 61000)
        if patient_data.shape[1] > 8000:
            patient60s_list1.append(i)
    print(len(patient60s_list1), patient60s_list1)

    import numpy as np
    import matplotlib.pyplot as plt

    def check_signal(patient):
        patient_signal = data['p'][0][patient][:, :8000]
        ppg_signal = patient_signal[0]
        bp_signal = patient_signal[1]
        ecg_signal = patient_signal[2]
        ppg_normalized = (ppg_signal - min(ppg_signal)) / (max(ppg_signal) - min(ppg_signal))
        bp_normalized = (bp_signal - min(bp_signal)) / (max(bp_signal) - min(bp_signal))
        ecg_normalized = (ecg_signal - min(ecg_signal)) / (max(ecg_signal) - min(ecg_signal))
        plt.figure(figsize=(30,6))
        plt.plot(ppg_normalized)
        plt.plot(bp_normalized)
        plt.plot(ecg_normalized)
        plt.show()
        #plt.show(block=False)
        #plt.pause(3)
        #plt.clf()


    #plt.ion()
    import time
    from IPython.display import display, clear_output

    for patient in patient60s_list1[450:]:
        check_signal(patient)
        print(patient)
        #plt.show(block=False)
        #plt.pause(3)
        time.sleep(3.5)
        #plt.clf()
        #plt.close(fig)
        clear_output(wait=True)
    #plt.ioff()

    patient60s_deleted1 = [2, 4, 62, 71, 83, 85, 86, 87, 88, 89, 99, 100, 101, 102, 104, 105, 106,108, 111, 114, 115, 116, 117, 121, 125, 126, 130, 131,132, 134, 139, 141, 146, 146, 149, 150, 152, 153, 157, 160, 165, 169, 171, 178, 179, 182, 184, 195, 224, 237, 240, 241, 242, 245, 246, 247, 249,250, 261, 262, 267, 272, 279, 283, 296, 308, 310, 311, 312, 313, 319, 320, 325, 326, 327, 329, 330, 331, 332, 333, 337, 339, 342, 345, 349, 356, 361, 363, 371, 376, 380, 382, 385, 388, 389, 407, 410, 416, 425, 440, 444, 449, 452, 453, 454, 457, 460, 462, 463, 464, 465, 466, 469, 471, 472, 474, 491, 492, 493, 496, 498, 500, 501, 502, 503, 516, 517, 520, 522, 524, 535, 537, 538, 539, 549, 550, 551, 555, 561, 564, 576, 577, 578, 586, 599, 601, 602, 603, 604, 605, 606, 607, 609, 612, 613, 614, 623, 629, 633, 634, 636, 639, 640, 642, 646, 653, 654, 657, 670,673, 676, 678, 679, 680, 681, 682, 683, 692, 694, 701, 711, 712, 717, 724, 727, 729, 731, 738, 740, 743, 756, 758, 759, 786, 787, 796, 807, 808, 809, 810, 812, 813, 815, 816, 835, 845, 846, 847, 848, 849, 852, 881, 903, 914, 917, 920, 948, 950, 977, 984, 986, 994, 997]
    patient60s_saved1 = []
    for i in patient60s_list1:
        if i not in patient60s_deleted1:
            patient60s_saved1.append(i)
    len(patient60s_saved1)

    # functions:
    fs =125 # Sample rate in Hz

    import scipy.signal as signal
    def chebyshev_filter(signal):
        # Define the filter order and cutoff frequency
        order = 4
        cutoff_freq = 20  # Cutoff frequency in Hz
        # Create the Chebyshev low-pass filter
        b, a = signal.cheby1(order, 0.5, cutoff_freq / (fs / 2), 'low', analog=False)
        return signal.lfilter(b, a, signal)

    from scipy.signal import butter, filtfilt
    def butter_filter(signal):
        fs = 125  # Sample rate
        cutoff = 5  # Cutoff frequency in Hz
        # Design the Butterworth low-pass filter
        nyquist = 0.5 * fs
        cutoff_norm = cutoff / nyquist
        b, a = butter(4, cutoff_norm, btype='low')
        return filtfilt(b, a, signal)

    import numpy as np
    def straighten_ecg(ecg_signal):
        detrended_ecg = np.subtract(ecg_signal, np.mean(ecg_signal))
        return detrended_ecg
        
    import numpy as np
    def normalize_sinal(ppg):
    # Assuming ppg_signal and ecg_signal are your original PPG and ECG signals
        ppg_min = np.min(ppg)
        ppg_max = np.max(ppg)
        normalized_ppg = (ppg - ppg_min) / (ppg_max - ppg_min)
        return normalized_ppg
    def standard_signal(bp): 
        return (bp - np.mean(bp)) / np.std(bp)
    def inverse_standard_signal(bp_ori, bp_est):
        mean = np.mean(bp_ori)
        std = np.std(bp_ori)
        return (bp_est * std) + mean

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    def align_ppgbp_segment(ppg_signal, bp_signal1, bp_signal2, ecg_signal, show=0): ## ppg_signal = ppg_normalized, bp_signal = bp_normalized, ecg_signal = ecg_normalized; get ppg_aligned, bp_aligned
        ppg_peaks, _ = find_peaks(ppg_signal, height=0.5)  # Adjust the height threshold as needed
        bp_peaks, _ = find_peaks(bp_signal1, height=0.4)
        ecg_peaks, _ = find_peaks(ecg_signal, height=0.65)
        #print(f'ppg peaks: {len(ppg_peaks)} {ppg_peaks}')
        #print(f'ecg peaks: {len(ecg_peaks)} {ecg_peaks}')

        first_ecg_peak = ecg_peaks[0]
        #print(f'first ecg peak: {first_ecg_peak}')
        indices_ppg = np.argwhere(ppg_peaks[:10] > first_ecg_peak)
        first_ppg_peak = ppg_peaks[int(indices_ppg[0])]
        #print(f'first ppg peak: {first_ppg_peak}')
        indices_bp = np.argwhere(bp_peaks[:10] > first_ecg_peak)
        first_bp_peak = bp_peaks[int(indices_bp[0])]
        #print(f'first bp peak: {first_bp_peak}')
        ppg_bp_peaks_subtraction = abs(bp_peaks[int(indices_bp[0]):int(indices_bp[0])+20] - ppg_peaks[int(indices_ppg[0]):int(indices_ppg[0])+20])
        #print(ppg_bp_peaks_subtraction)
        distance_ppgbp = np.bincount(ppg_bp_peaks_subtraction).argmax()
        #print(move_distance)
        #print(bp_peaks[int(indices_bp[0]):int(indices_bp[0])+20] - ppg_peaks[int(indices_ppg[0]):int(indices_ppg[0])+20])
        #print(f'distance:{distance_ppgbp}')
        if first_bp_peak > first_ppg_peak:
            bp_aligned = bp_signal1[distance_ppgbp:]
            bp_ori_aligned = bp_signal2[distance_ppgbp:]
            ppg_aligned = ppg_signal
        elif first_bp_peak < first_ppg_peak:
            bp_aligned = bp_signal1
            bp_ori_aligned = bp_signal2
            ppg_aligned = ppg_signal[distance_ppgbp:]
        else:
            bp_aligned = bp_signal1
            bp_ori_aligned = bp_signal2
            ppg_aligned = ppg_signal
        #print(f'ppg len: {len(ppg_aligned)}')
        #print(f'bp len: {len(bp_aligned)}')
        min_len = min(len(bp_aligned), len(ppg_aligned))
        ppg_aligned = ppg_aligned[:min_len]
        bp_aligned = bp_aligned[:min_len]
        bp_ori_aligned = bp_ori_aligned[:min_len]
        ecg_aligned = ecg_signal[:min_len]
        #print(ecg_aligned)
        ppg_segmented = ppg_aligned[:first_ecg_peak-5]
        bp_segmented = bp_aligned[:first_ecg_peak-5]
        bp_ori_segmented = bp_ori_aligned[:first_ecg_peak-5]
        ecg_segmented = ecg_aligned[:first_ecg_peak-5]

        for ecgi in range(len(ecg_peaks)-1):
            one_ppg_peak, _ = find_peaks(ppg_aligned[ecg_peaks[ecgi]-5:ecg_peaks[ecgi + 1]-5], height=0.5)
            #print(ecg_peaks[ecgi], one_ppg_peak)
            if len(one_ppg_peak) == 1:
                ppg_segmented = np.concatenate((ppg_segmented, ppg_aligned[ecg_peaks[ecgi]-5:ecg_peaks[ecgi + 1]-5]))
                bp_segmented = np.concatenate((bp_segmented, bp_aligned[ecg_peaks[ecgi]-5:ecg_peaks[ecgi + 1]-5]))
                bp_ori_segmented = np.concatenate((bp_ori_segmented, bp_ori_aligned[ecg_peaks[ecgi]-5:ecg_peaks[ecgi + 1]-5]))
                ecg_segmented = np.concatenate((ecg_segmented, ecg_aligned[ecg_peaks[ecgi]-5:ecg_peaks[ecgi + 1]-5]))

        if show == 1:
            plt.figure(figsize=(30, 6))
            plt.plot(ppg_signal, label='PPG')
            plt.plot(bp_signal1, label='BP')
            plt.plot(ecg_signal, label='ECG')
            plt.scatter(ppg_peaks, ppg_signal[ppg_peaks], color='c', marker='o', label='Aligned PPG Peaks')
            plt.scatter(bp_peaks, bp_signal1[bp_peaks], color='orange', marker='o', label='Aligned BP Peaks')
            plt.scatter(ecg_peaks, ecg_signal[ecg_peaks], color='green', marker='o', label='Aligned BP Peaks')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()

            plt.figure(figsize=(30, 6))
            plt.plot(ppg_segmented, label='PPG')
            plt.plot(bp_segmented, label='BP')
            plt.plot(ecg_segmented, label='ECG')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()

        return ppg_segmented, bp_segmented, bp_ori_segmented, ecg_segmented

    import numpy as np
    def get_feautres(ppg_signal, bp_signal1, bp_signal2, ecg_signal):
        ori_ecg_peaks, _ = find_peaks(ecg_signal)
        r_peaks = np.array([ecg_peak for ecg_peak in ori_ecg_peaks if ecg_signal[ecg_peak] > 0.65])
        #print(f'r_peaks: {len(r_peaks)} {r_peaks}')
        r_peak_amplitudes = (ecg_signal[r_peaks]).tolist()
        r_peak_amplitudes = r_peak_amplitudes[:-1]
        #print(f'r_peak_amplitudes: {len(r_peak_amplitudes)} {r_peak_amplitudes}')
        r_peak_intervals = (np.diff(r_peaks) / fs).tolist()
        #print(f'r_peak_intervals: {len(r_peak_intervals)} {r_peak_intervals}')
        # calculate low peak, s-peak
        low_peak_amplitudes, r_to_low_peak_amplitudes, s_peak_amplitudes= [], [], []
        low_peaks, s_peaks = [], []
        for i in range(len(r_peaks) - 1):
            r_peak = r_peaks[i]
            next_r_peak = r_peaks[i + 1]
            low_peak_amplitude = np.min(ecg_signal[r_peak:next_r_peak])
            r_to_low_peak_amplitude = ecg_signal[r_peak]-low_peak_amplitude
            low_peak_amplitudes.append(low_peak_amplitude)
            r_to_low_peak_amplitudes.append(r_to_low_peak_amplitude)
            low_peak = r_peak + np.argmin(ecg_signal[r_peak:next_r_peak])
            low_peaks.append(low_peak)
            s_peak_amplitude = np.min(ecg_signal[r_peak:low_peak])
            s_peak_amplitudes.append(s_peak_amplitude)
            s_peak = r_peak + np.argmin(ecg_signal[r_peak:low_peak])
            s_peaks.append(s_peak)   
        #print(f'low peaks: {len(low_peaks)} {low_peaks}')  
        #print(f's peaks: {len(s_peaks)} {s_peaks}')  
        # T-Wave Amplitude Calculation
        r_peaks = np.insert(r_peaks, 0, 0) #the first t-peak is before the R-peak
        t_wave_amplitudes, q_wave_amplitudes = [], []
        t_peaks, q_peaks = [], []
        for i in range(len(r_peaks) - 1):
            r_peak = r_peaks[i]
            next_r_peak = r_peaks[i + 1]
            t_wave_amplitude = np.max(ecg_signal[r_peak:next_r_peak])
            t_wave_amplitudes.append(t_wave_amplitude)
            t_peak = r_peak + np.argmax(ecg_signal[r_peak:next_r_peak])
            t_peaks.append(t_peak)
            q_wave_amplitude = np.min(ecg_signal[t_peak:next_r_peak])
            q_wave_amplitudes.append(q_wave_amplitude)
            q_peak = r_peak + np.argmin(ecg_signal[t_peak:next_r_peak])
            q_peaks.append(q_peak)

        t_wave_amplitudes = t_wave_amplitudes[:-1]
        q_wave_amplitudes = q_wave_amplitudes[:-1]
        #print(f't peaks: {len(t_peaks)} {t_peaks}')  
        #print(f'q peaks: {len(q_peaks)} {q_peaks}')  
        
        # QRS interval
        r_peaks = np.array([ecg_peak for ecg_peak in ori_ecg_peaks if ecg_signal[ecg_peak] > 0.65])
        qrs_intervals = []
        for i in range(len(r_peaks)-1):
            qrs_interval = (ecg_signal[s_peaks[i]] - ecg_signal[q_peaks[i]]) / fs
            qrs_intervals.append(abs(qrs_interval))
        #print(f'qrs_intervals: {len(qrs_intervals)} {qrs_intervals}')  

        # get ppg features:
        ppg_pulses, bp_pulses, bp_ori_pulses = [], [], []
        for i in range(len(r_peaks)-1):
            ppg_pulse = ppg_signal[r_peaks[i]:r_peaks[i+1]]
            bp_pulse = bp_signal1[r_peaks[i]:r_peaks[i+1]]
            bp_ori_pulse = bp_signal2[r_peaks[i]:r_peaks[i+1]]
            ppg_pulses.append(ppg_pulse)
            bp_pulses.append(bp_pulse)
            bp_ori_pulses.append(bp_ori_pulse)
        #print(f'bp pulses: {bp_pulses}')
        ppg_pulse_amplitude = [np.max(pulse) - np.min(pulse) for pulse in ppg_pulses]
        ppg_pulse_width = [pulse.shape[0]/fs for pulse in ppg_pulses]
        ppg_high_to_low_interval = [np.argmin(pulse)/fs for pulse in ppg_pulses]
        ppg_slope_change_std = [np.std(np.diff(pulse)) for pulse in ppg_pulses]

        sbp = [np.max(pulse) for pulse in bp_pulses]
        dbp = [np.min(pulse) for pulse in bp_pulses]
        sbp_ori = [np.max(pulse) for pulse in bp_ori_pulses]
        dbp_ori = [np.min(pulse) for pulse in bp_ori_pulses]
        
        return sbp, dbp, sbp_ori, dbp_ori, ppg_pulse_amplitude, ppg_pulse_width, ppg_high_to_low_interval, ppg_slope_change_std, t_wave_amplitudes, q_wave_amplitudes, r_peak_amplitudes, s_peak_amplitudes, low_peak_amplitudes, qrs_intervals, r_to_low_peak_amplitudes, r_peak_intervals

    import matplotlib.pyplot as plt
    def show_one(signal1):
        fig = plt.figure(figsize=(30,6))
        plt.plot(signal1)
        return plt.show()
    def show_two(signal1, signal2):
        fig = plt.figure(figsize=(30,6))
        plt.plot(signal1, label='1')
        plt.plot(signal2, label='2')
        plt.legend()
        return plt.show()
    def show_three(signal1, signal2, signal3):
        fig = plt.figure(figsize=(30,6))
        plt.plot(signal1, label='1')
        plt.plot(signal2, label='2')
        plt.plot(signal3, label='3')
        plt.legend()
        return plt.show()

    sbp500 = []
    dbp500 = []
    sbp_ori500 = []
    dbp_ori500 = []
    features500 = []
    for patient in patient60s_saved1:
        patient_signal = data['p'][0][patient][:, :3000]
        ppg_signal = patient_signal[0]
        bp_signal = patient_signal[1]
        ecg_signal = patient_signal[2]
        ppg_normalized = (ppg_signal - min(ppg_signal)) / (max(ppg_signal) - min(ppg_signal))
        bp_normalized = (bp_signal - min(bp_signal)) / (max(bp_signal) - min(bp_signal))
        ecg_normalized = (ecg_signal - min(ecg_signal)) / (max(ecg_signal) - min(ecg_signal))
        ppg_segmented, bp_segmented, bp_ori_segmented, ecg_segmented = align_ppgbp_segment(ppg_signal = ppg_normalized, bp_signal1 = bp_normalized, bp_signal2 = bp_signal, ecg_signal = ecg_normalized, show=0)
        bps_features = get_feautres(ppg_segmented, bp_segmented, bp_ori_segmented, ecg_segmented)
        sbp = np.array(bps_features[0])
        dbp = np.array(bps_features[1])
        sbp_ori = np.array(bps_features[2])
        dbp_ori = np.array(bps_features[3])
        features = np.array(bps_features[4:])
        sbp500.append(sbp)
        dbp500.append(dbp)
        sbp_ori500.append(sbp_ori)
        dbp_ori500.append(dbp_ori)
        features500.append(features)

    bp500 = []
    bp_ori500 = []
    features_wave500 = []
    for patient in patient60s_saved1[:500]:
        patient_signal = data['p'][0][patient][:, :3750]
        ppg_signal = patient_signal[0]
        bp_signal = patient_signal[1]
        ecg_signal = patient_signal[2]
        ppg_normalized = (ppg_signal - min(ppg_signal)) / (max(ppg_signal) - min(ppg_signal))
        bp_normalized = (bp_signal - min(bp_signal)) / (max(bp_signal) - min(bp_signal))
        ecg_normalized = (ecg_signal - min(ecg_signal)) / (max(ecg_signal) - min(ecg_signal))
        ppg_segmented, bp_segmented, bp_ori_segmented, ecg_segmented = align_ppgbp_segment(ppg_signal = ppg_normalized, bp_signal1 = bp_normalized, bp_signal2 = bp_signal, ecg_signal = ecg_normalized, show=0)
        features_wave = np.array([ppg_segmented, ecg_segmented])
        bp500.append(bp_segmented)
        bp_ori500.append(bp_ori_segmented)
        features_wave500.append(features_wave)

    ## point
    train_sbp = sbp500[:400]
    train_dbp = dbp500[:400]
    train_sbp_ori = sbp_ori500[:400]
    train_dbp_ori = dbp_ori500[:400]
    train_features = features500[:400]
    test_sbp = []
    test_dbp = []
    test_sbp_ori = []
    test_dbp_ori = []
    test_features = []

    for i in range(len(sbp500)-400):
        train_sbp.append(sbp500[i+400][:int(0.2*len(sbp500[i+400]))])
        train_dbp.append(dbp500[i+400][:int(0.2*len(dbp500[i+400]))])
        train_sbp_ori.append(sbp_ori500[i+400][:int(0.2*len(sbp_ori500[i+400]))])
        train_dbp_ori.append(dbp_ori500[i+400][:int(0.2*len(dbp_ori500[i+400]))])
        train_features.append(features500[i+400][:,:int(0.2*(features500[i+400].shape[1]))])
        test_sbp.append(sbp500[i+400][int(0.2*len(sbp500[i+400])):])
        test_dbp.append(dbp500[i+400][int(0.2*len(dbp500[i+400])):])
        test_sbp_ori.append(sbp_ori500[i+400][int(0.2*len(sbp_ori500[i+400])):])
        test_dbp_ori.append(dbp_ori500[i+400][int(0.2*len(dbp_ori500[i+400])):])
        test_features.append(features500[i+400][:,int(0.2*(features500[i+400].shape[1])):])

    merged_train_sbp = np.concatenate(train_sbp, axis=0)
    merged_train_dbp = np.concatenate(train_dbp, axis=0)
    merged_train_sbp_ori = np.concatenate(train_sbp_ori, axis=0)
    merged_train_dbp_ori = np.concatenate(train_dbp_ori, axis=0)
    merged_train_features = np.concatenate(train_features, axis=1)

    # wave
    train_bp = bp500[:400]
    train_bp_ori = bp_ori500[:400]
    train_features_wave = features_wave500[:400]
    test_bp = []
    test_bp_ori = []
    test_features_wave = []

    for i in range(len(bp500)-400):
        train_bp.append(bp500[i+400][:int(0.2*len(bp500[i+400]))])
        train_bp_ori.append(bp_ori500[i+400][:int(0.2*len(bp_ori500[i+400]))])
        train_features_wave.append(features_wave500[i+400][:,:int(0.2*(features_wave500[i+400].shape[1]))])
        test_bp.append(bp500[i+400][int(0.2*len(bp500[i+400])):])
        test_bp_ori.append(bp_ori500[i+400][int(0.2*len(bp_ori500[i+400])):])
        test_features_wave.append(features_wave500[i+400][:,int(0.2*(features_wave500[i+400].shape[1])):])

    merged_train_bp = np.concatenate(train_bp, axis=0)
    merged_train_bp_ori = np.concatenate(train_bp_ori, axis=0)
    merged_train_features_wave = np.concatenate(train_features_wave, axis=1)

    # wave 100 patients features
    bp500 = []
    bp_ori500 = []
    features_wave500 = []
    for patient in patient60s_saved1[:100]:
        patient_signal = data['p'][0][patient][:, :2000]
        ppg_signal = patient_signal[0]
        bp_signal = patient_signal[1]
        ecg_signal = patient_signal[2]
        ppg_normalized = (ppg_signal - min(ppg_signal)) / (max(ppg_signal) - min(ppg_signal))
        bp_normalized = (bp_signal - min(bp_signal)) / (max(bp_signal) - min(bp_signal))
        ecg_normalized = (ecg_signal - min(ecg_signal)) / (max(ecg_signal) - min(ecg_signal))
        ppg_segmented, bp_segmented, bp_ori_segmented, ecg_segmented = align_ppgbp_segment(ppg_signal = ppg_normalized, bp_signal1 = bp_normalized, bp_signal2 = bp_signal, ecg_signal = ecg_normalized, show=0)
        features_wave = np.array([ppg_segmented, ecg_segmented])
        bp500.append(bp_segmented)
        bp_ori500.append(bp_ori_segmented)
        features_wave500.append(features_wave)
    train_bp = bp500[:80]
    train_bp_ori = bp_ori500[:80]
    train_features_wave = features_wave500[:80]
    test_bp = []
    test_bp_ori = []
    test_features_wave = []

    for i in range(len(bp500)-80):
        train_bp.append(bp500[i+80][:int(0.2*len(bp500[i+80]))])
        train_bp_ori.append(bp_ori500[i+80][:int(0.2*len(bp_ori500[i+80]))])
        train_features_wave.append(features_wave500[i+80][:,:int(0.2*(features_wave500[i+80].shape[1]))])
        test_bp.append(bp500[i+80][int(0.2*len(bp500[i+80])):])
        test_bp_ori.append(bp_ori500[i+80][int(0.2*len(bp_ori500[i+80])):])
        test_features_wave.append(features_wave500[i+80][:,int(0.2*(features_wave500[i+80].shape[1])):])

    merged_train_bp = np.concatenate(train_bp, axis=0)
    merged_train_bp_ori = np.concatenate(train_bp_ori, axis=0)
    merged_train_features_wave = np.concatenate(train_features_wave, axis=1)

    # points
    ## array X_train, linear and forest model
    X_train = merged_train_features.T
    sbp_train = merged_train_sbp
    dbp_train = merged_train_dbp

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    linear_model_sbp = LinearRegression()
    linear_model_sbp.fit(X_train, sbp_train)
    linear_model_dbp = LinearRegression()
    linear_model_dbp.fit(X_train, dbp_train)
    from sklearn.ensemble import RandomForestRegressor
    X_train = X_train.reshape(-1, 1) if len(X_train.shape) == 1 else X_train
    randomforest_model_sbp = RandomForestRegressor(n_estimators=100, random_state=42)
    randomforest_model_sbp.fit(X_train, sbp_train)
    randomforest_model_dbp = RandomForestRegressor(n_estimators=100, random_state=42)
    randomforest_model_dbp.fit(X_train, dbp_train)

    # wave 400 train
    X_train = merged_train_features_wave.T
    bp_train = merged_train_bp

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    linear_model_bp = LinearRegression()
    linear_model_bp.fit(X_train, bp_train)
    from sklearn.ensemble import RandomForestRegressor
    X_train = X_train.reshape(-1, 1) if len(X_train.shape) == 1 else X_train
    randomforest_model_bp = RandomForestRegressor(n_estimators=100, random_state=42)
    randomforest_model_bp.fit(X_train, bp_train)

    # wave 80 train
    X_train = merged_train_features_wave.T
    bp_train = merged_train_bp

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.ensemble import RandomForestRegressor
    X_train = X_train.reshape(-1, 1) if len(X_train.shape) == 1 else X_train
    randomforest_model_bp80 = RandomForestRegressor(n_estimators=100, random_state=42)
    randomforest_model_bp80.fit(X_train, bp_train)

    # wave 20 test
    import pandas as pd
    wave_ppg_ecg_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ori_ppg_cc', 'refbp_ecg_cc')))
    for i in range(1):#len(test_bp)
        X_test = test_features_wave[i].T
        bp_test = test_bp[i]
        bp_test_ori = test_bp_ori[i]
        X_test = X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test
        bp_pred_randomforest = randomforest_model_bp80.predict(X_test)
        # print(len(bp_pred_randomforest), bp_pred_randomforest)
        bp_pred_ori_randomforest = (bp_pred_randomforest * (max(bp_test_ori) - min(bp_test_ori))) + min(bp_test_ori)
        '''print(len(bp_pred_ori_randomforest), bp_pred_ori_randomforest)
        plt.figure(figsize=(30,6))
        plt.plot(bp_pred_randomforest)
        plt.plot(test_features_wave[i][0])
        plt.show()
        print(len(bp_test_ori), bp_test_ori)
        print(len(test_features_wave[i][0]), test_features_wave[i][0])
        print(len(ppg_signal[:len(bp_pred_ori_randomforest)]), ppg_signal[:len(bp_pred_ori_randomforest)])
        print(len(test_features_wave[i][1]), test_features_wave[i][1])'''
        correlation_matrix_1 = np.corrcoef(bp_test_ori, bp_pred_ori_randomforest)
        correlation_coefficient_1 = correlation_matrix_1[0, 1]
        correlation_matrix_2 = np.corrcoef(test_features_wave[i][0], bp_pred_ori_randomforest)
        correlation_coefficient_2 = correlation_matrix_2[0, 1]
        ppg_signal = data['p'][0][patient60s_saved1[400+i]][:, :7500][0]
        correlation_matrix_3 = np.corrcoef(ppg_signal[:len(bp_pred_ori_randomforest)], bp_pred_ori_randomforest)
        correlation_coefficient_3 = correlation_matrix_3[0, 1]
        correlation_matrix_4 = np.corrcoef(test_features_wave[i][1], bp_pred_ori_randomforest)
        correlation_coefficient_4 = correlation_matrix_4[0, 1]
        cc_list = [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
        wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = cc_list
        display(wave_ppg_ecg_df)
    wave_ppg_ecg_df.to_csv('wave_ppg_ecg_random_20test_60s.csv')

    # points
    # X_test, linear and forest result
    mae_sbp_linears = []
    rmse_sbp_linears = []
    mae_sbp_randomforests = []
    rmse_sbp_randomforests = []
    mae_dbp_linears = []
    rmse_dbp_linears = []
    mae_dbp_randomforests = []
    rmse_dbp_randomforests = []
    for i in range(len(test_sbp)):#len(test_sbp)
        X_test = test_features[i].T
        sbp_test = test_sbp[i]
        sbp_test_ori = test_sbp_ori[i]
        dbp_test = test_dbp[i]
        dbp_test_ori = test_dbp_ori[i]
        sbp_pred_linear = linear_model_sbp.predict(X_test)
        sbp_pred_ori_linear = (sbp_pred_linear * (max(sbp_test_ori) - min(sbp_test_ori))) + min(sbp_test_ori)
        dbp_pred_linear = linear_model_dbp.predict(X_test)
        dbp_pred_ori_linear = (dbp_pred_linear * (max(dbp_test_ori) - min(dbp_test_ori))) + min(dbp_test_ori)
        # Evaluate the model
        mae_sbp_linear = mean_absolute_error(sbp_test_ori, sbp_pred_ori_linear)
        rmse_sbp_linear = mean_squared_error(sbp_test_ori, sbp_pred_ori_linear, squared=False)
        mae_dbp_linear = mean_absolute_error(dbp_test_ori, dbp_pred_ori_linear)
        rmse_dbp_linear = mean_squared_error(dbp_test_ori, dbp_pred_ori_linear, squared=False)
        # Print the evaluation metrics
        '''print("Root Mean Squared Error (RMSE):", rmse_sbp_linear)
        print("Root Mean Squared Error (RMSE):", rmse_dbp_linear)
        print("Mean Absolute Error (MAE):", mae_sbp_linear)
        print("Mean Absolute Error (MAE):", mae_dbp_linear)
        show_two(sbp_test, sbp_pred_linear)
        show_two(dbp_test_ori, dbp_pred_ori_linear)'''

        X_test = X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test
        sbp_pred_randomforest = randomforest_model_sbp.predict(X_test)
        sbp_pred_ori_randomforest = (sbp_pred_randomforest * (max(sbp_test_ori) - min(sbp_test_ori))) + min(sbp_test_ori)
        dbp_pred_randomforest = randomforest_model_dbp.predict(X_test)
        dbp_pred_ori_randomforest = (dbp_pred_randomforest * (max(dbp_test_ori) - min(dbp_test_ori))) + min(dbp_test_ori)
        rmse_sbp_randomforest = mean_squared_error(sbp_test_ori, sbp_pred_ori_randomforest, squared=False)
        mae_sbp_randomforest = mean_absolute_error(sbp_test_ori, sbp_pred_ori_randomforest)
        rmse_dbp_randomforest = mean_squared_error(dbp_test_ori, dbp_pred_ori_randomforest, squared=False)
        mae_dbp_randomforest = mean_absolute_error(dbp_test_ori, dbp_pred_ori_randomforest)
        '''print("Root Mean Squared Error (RMSE):", rmse_sbp_randomforest)
        print("Root Mean Squared Error (RMSE):", rmse_dbp_randomforest)
        print("Mean Absolute Error (MAE):", mae_sbp_randomforest)
        print("Mean Absolute Error (MAE):", mae_dbp_randomforest)

        show_two(sbp_test, sbp_pred_randomforest)
        show_two(dbp_test_ori, dbp_pred_ori_randomforest)'''
        mae_sbp_linears.append(mae_sbp_linear)
        rmse_sbp_linears.append(rmse_sbp_linear)
        mae_sbp_randomforests.append(mae_sbp_randomforest)
        rmse_sbp_randomforests.append(rmse_sbp_randomforest)
        mae_dbp_linears.append(mae_dbp_linear)
        rmse_dbp_linears.append(rmse_dbp_linear)
        mae_dbp_randomforests.append(mae_dbp_randomforest)
        rmse_dbp_randomforests.append(rmse_dbp_randomforest)

    # wave
    import pandas as pd
    wave_ppg_ecg_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ori_ppg_cc', 'refbp_ecg_cc')))
    for i in range(28,30):#len(test_sbp)
        X_test = test_features_wave[i].T
        bp_test = test_bp[i]
        bp_test_ori = test_bp_ori[i]
        X_test = X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test
        bp_pred_randomforest = randomforest_model_bp.predict(X_test)
        # print(len(bp_pred_randomforest), bp_pred_randomforest)
        bp_pred_ori_randomforest = (bp_pred_randomforest * (max(bp_test_ori) - min(bp_test_ori))) + min(bp_test_ori)
        '''print(len(bp_pred_ori_randomforest), bp_pred_ori_randomforest)
        plt.figure(figsize=(30,6))
        plt.plot(bp_pred_randomforest)
        plt.plot(test_features_wave[i][0])
        plt.show()
        print(len(bp_test_ori), bp_test_ori)
        print(len(test_features_wave[i][0]), test_features_wave[i][0])
        print(len(ppg_signal[:len(bp_pred_ori_randomforest)]), ppg_signal[:len(bp_pred_ori_randomforest)])
        print(len(test_features_wave[i][1]), test_features_wave[i][1])'''
        correlation_matrix_1 = np.corrcoef(bp_test_ori, bp_pred_ori_randomforest)
        correlation_coefficient_1 = correlation_matrix_1[0, 1]
        correlation_matrix_2 = np.corrcoef(test_features_wave[i][0], bp_pred_ori_randomforest)
        correlation_coefficient_2 = correlation_matrix_2[0, 1]
        ppg_signal = data['p'][0][patient60s_saved1[400+i]][:, :3750][0]
        correlation_matrix_3 = np.corrcoef(ppg_signal[:len(bp_pred_ori_randomforest)], bp_pred_ori_randomforest)
        correlation_coefficient_3 = correlation_matrix_3[0, 1]
        correlation_matrix_4 = np.corrcoef(test_features_wave[i][1], bp_pred_ori_randomforest)
        correlation_coefficient_4 = correlation_matrix_4[0, 1]
        cc_list = [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
        wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = cc_list
        display(wave_ppg_ecg_df)
    wave_ppg_ecg_df.to_csv('wave_ppg_ecg_random90-60.csv')

    # wave
    import pandas as pd
    wave_ppg_ecg_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ori_ppg_cc', 'refbp_ecg_cc')))
    for i in range(82,90):#len(test_sbp)
        X_test = test_features_wave[i].T
        bp_test = test_bp[i]
        bp_test_ori = test_bp_ori[i]
        X_test = X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test
        bp_pred_randomforest = randomforest_model_bp.predict(X_test)
        # print(len(bp_pred_randomforest), bp_pred_randomforest)
        bp_pred_ori_randomforest = (bp_pred_randomforest * (max(bp_test_ori) - min(bp_test_ori))) + min(bp_test_ori)
        '''print(len(bp_pred_ori_randomforest), bp_pred_ori_randomforest)
        plt.figure(figsize=(30,6))
        plt.plot(bp_pred_randomforest)
        plt.plot(test_features_wave[i][0])
        plt.show()
        print(len(bp_test_ori), bp_test_ori)
        print(len(test_features_wave[i][0]), test_features_wave[i][0])
        print(len(ppg_signal[:len(bp_pred_ori_randomforest)]), ppg_signal[:len(bp_pred_ori_randomforest)])
        print(len(test_features_wave[i][1]), test_features_wave[i][1])'''
        correlation_matrix_1 = np.corrcoef(bp_test_ori, bp_pred_ori_randomforest)
        correlation_coefficient_1 = correlation_matrix_1[0, 1]
        correlation_matrix_2 = np.corrcoef(test_features_wave[i][0], bp_pred_ori_randomforest)
        correlation_coefficient_2 = correlation_matrix_2[0, 1]
        ppg_signal = data['p'][0][patient60s_saved1[400+i]][:, :3750][0]
        correlation_matrix_3 = np.corrcoef(ppg_signal[:len(bp_pred_ori_randomforest)], bp_pred_ori_randomforest)
        correlation_coefficient_3 = correlation_matrix_3[0, 1]
        correlation_matrix_4 = np.corrcoef(test_features_wave[i][1], bp_pred_ori_randomforest)
        correlation_coefficient_4 = correlation_matrix_4[0, 1]
        cc_list = [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
        wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = cc_list
        display(wave_ppg_ecg_df)
    wave_ppg_ecg_df.to_csv('wave_ppg_ecg_random60-70.csv')

    np.mean(mae_sbp_linears), np.mean(rmse_sbp_linears), np.mean(mae_dbp_linears), np.mean(rmse_dbp_linears)
    np.mean(mae_sbp_randomforests), np.mean(rmse_sbp_randomforests), np.mean(mae_dbp_randomforests), np.mean(rmse_dbp_randomforests)
    np.mean(correlation_coefficient_1), np.mean(correlation_coefficient_2), np.mean(correlation_coefficient_3), np.mean(correlation_coefficient_4)

    import pandas as pd
    wave_ppg_ecg_df = pd.read_csv(/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/4July23_ccmae/wave_ppg_ecg_random_20test_30s.csv)

    def cc_histogram(wave_ppg_ecg_df):
        plt.figure(figsize=(15, 6), dpi=300)
        # Set your color palette
        colors = ['pink', 'c', 'green', 'orange']

        # Specify data
        data1 = np.array(wave_ppg_ecg_df['refbp_estbp_cc'])
        data2 = np.array(wave_ppg_ecg_df['refbp_ppg_cc'])
        data3 = np.array(wave_ppg_ecg_df['refbp_ecg_cc'])

        # Create bins and histogram
        bins = np.linspace(0, 1, 11)
        counts1, _ = np.histogram(abs(data1), bins=bins)
        counts2, _ = np.histogram(abs(data2), bins=bins)
        counts3, _ = np.histogram(abs(data3), bins=bins)

        # Calculate frequencies
        freq1 = counts1 / len(data1)
        freq2 = counts2 / len(data2)
        freq3 = counts3 / len(data3)

        barWidth = 0.25
        r1 = np.arange(len(freq1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]

        plt.bar(r1, freq1, color=colors[0], width=barWidth, edgecolor='grey', label='refBP_estBP')
        plt.bar(r2, freq2, color=colors[1], width=barWidth, edgecolor='grey', label='refBP_PPG')
        plt.bar(r3, freq3, color=colors[2], width=barWidth, edgecolor='grey', label='refBP_ECG')

        # Adding labels
        for i in range(len(r1)):
            plt.text(x = r1[i] + barWidth/2 - 0.1 , y = freq1[i] + 0.02, s = f"{counts1[i]}", size = 10, ha='center')
            plt.text(x = r2[i] + barWidth/2 - 0.1 , y = freq2[i] + 0.02, s = f"{counts2[i]}", size = 10, ha='center')
            plt.text(x = r3[i] + barWidth/2 - 0.1 , y = freq3[i] + 0.02, s = f"{counts3[i]}", size = 10, ha='center')

        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.xticks([r + barWidth for r in range(len(freq1))], ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
        plt.legend()
        plt.show()
        return

    def cc_histogram(wave_ppg_ecg_df):
        plt.figure(figsize=(15, 6), dpi=300)
        # Set your color palette
        colors = ['pink', 'c', 'green', 'orange']

        # Specify data
        data1 = np.array(wave_ppg_ecg_df['refbp_estbp_cc'])
        data2 = np.array(wave_ppg_ecg_df['refbp_ppg_cc'])
        data3 = np.array(wave_ppg_ecg_df['refbp_ori_ppg_cc'])
        data4 = np.array(wave_ppg_ecg_df['refbp_ecg_cc'])

        # Create bins and histogram
        bins = np.linspace(0, 1, 11)
        counts1, _ = np.histogram(abs(data1), bins=bins)
        counts2, _ = np.histogram(abs(data2), bins=bins)
        counts3, _ = np.histogram(abs(data3), bins=bins)
        counts4, _ = np.histogram(abs(data4), bins=bins)

        # Calculate frequencies
        freq1 = counts1 / len(data1)
        freq2 = counts2 / len(data2)
        freq3 = counts3 / len(data3)
        freq4 = counts4 / len(data4)

        barWidth = 0.25
        r1 = np.arange(len(freq1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        r4 = [x + barWidth for x in r3]

        plt.bar(r1, freq1, color=colors[0], width=barWidth, edgecolor='grey', label='refBP_estBP')
        plt.bar(r2, freq2, color=colors[1], width=barWidth, edgecolor='grey', label='PPG_estBP')
        plt.bar(r3, freq3, color=colors[2], width=barWidth, edgecolor='grey', label='oriPPG_estBP')
        plt.bar(r4, freq4, color=colors[3], width=barWidth, edgecolor='grey', label='ECG_estBP')

        # Adding labels
        for i in range(len(r1)):
            plt.text(x = r1[i] + barWidth/2 - 0.1 , y = freq1[i] + 0.02, s = f"{counts1[i]}", size = 10, ha='center')
            plt.text(x = r2[i] + barWidth/2 - 0.1 , y = freq2[i] + 0.02, s = f"{counts2[i]}", size = 10, ha='center')
            plt.text(x = r3[i] + barWidth/2 - 0.1 , y = freq3[i] + 0.02, s = f"{counts3[i]}", size = 10, ha='center')
            plt.text(x = r4[i] + barWidth/2 - 0.1 , y = freq4[i] + 0.02, s = f"{counts4[i]}", size = 10, ha='center')

        plt.xlabel('500patients 30s Correlation Coefficient(Random Forest Model)')
        plt.ylabel('Frequency')
        plt.xticks([r + barWidth for r in range(len(freq1))], ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
        plt.legend()
        plt.show()
        return

    import pandas as pd
    wave_ppg_ecg_df = pd.read_csv('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/cc_mae/fourth_July_23_ccmae/wave_ppg_ecg_random_100test_30s.csv')
    wave_ppg_ecg_df = wave_ppg_ecg_df[:-1]
    wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = [0, 0.895563, 0.949758, -0.313758, -0.025052]
    wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = [0, 0.826622, 0.930730, -0.044141, -0.012152]
    wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = [0, 0.942088905, 0.962264971, -0.046002216, -0.030258016]
    wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = [0, 0.901039565, 0.972112837, -0.379322227, -0.114881816]
    wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = [0, 0.833227666, 0.946853212, -0.194686963, -0.08393826]
    wave_ppg_ecg_df

    # wave 20 test
    import pandas as pd
    wave_ppg_ecg_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ori_ppg_cc', 'refbp_ecg_cc')))
    for i in range(20):#len(test_bp)
        X_test = test_features_wave[i].T
        bp_test = test_bp[i]
        bp_test_ori = test_bp_ori[i]
        X_test = X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test
        bp_pred_randomforest = randomforest_model_bp80.predict(X_test)
        # print(len(bp_pred_randomforest), bp_pred_randomforest)
        bp_pred_ori_randomforest = (bp_pred_randomforest * (max(bp_test_ori) - min(bp_test_ori))) + min(bp_test_ori)
        '''print(len(bp_pred_ori_randomforest), bp_pred_ori_randomforest)
        plt.figure(figsize=(30,6))
        plt.plot(bp_pred_randomforest)
        plt.plot(test_features_wave[i][0])
        plt.show()
        print(len(bp_test_ori), bp_test_ori)
        print(len(test_features_wave[i][0]), test_features_wave[i][0])
        print(len(ppg_signal[:len(bp_pred_ori_randomforest)]), ppg_signal[:len(bp_pred_ori_randomforest)])
        print(len(test_features_wave[i][1]), test_features_wave[i][1])'''
        print(i)
        
        correlation_matrix_1 = np.corrcoef(bp_test_ori, bp_pred_ori_randomforest)
        correlation_coefficient_1 = correlation_matrix_1[0, 1]
        correlation_matrix_2 = np.corrcoef(test_features_wave[i][0], bp_pred_ori_randomforest)
        correlation_coefficient_2 = correlation_matrix_2[0, 1]
        ppg_signal = data['p'][0][patient60s_saved1[400+i]][:, :7500][0]
        correlation_matrix_3 = np.corrcoef(ppg_signal[:len(bp_pred_ori_randomforest)], bp_pred_ori_randomforest)
        correlation_coefficient_3 = correlation_matrix_3[0, 1]
        correlation_matrix_4 = np.corrcoef(test_features_wave[i][1], bp_pred_ori_randomforest)
        correlation_coefficient_4 = correlation_matrix_4[0, 1]
        cc_list = [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
        wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = cc_list
    display(wave_ppg_ecg_df)
    wave_ppg_ecg_df.to_csv('wave_ppg_ecg_random_20test_30s_21Jul.csv')
    # wave 20 test
    import pandas as pd
    wave_ppg_ecg_df = pd.DataFrame(columns=(('refbp_estbp_cc', 'refbp_ppg_cc', 'refbp_ori_ppg_cc', 'refbp_ecg_cc')))
    listselect = [1, 12, 9, 11]
    for i in listselect:#len(test_bp)
        X_test = test_features_wave[i].T
        bp_test = test_bp[i]
        bp_test_ori = test_bp_ori[i]
        X_test = X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test
        bp_pred_randomforest = randomforest_model_bp80.predict(X_test)
        # print(len(bp_pred_randomforest), bp_pred_randomforest)
        bp_pred_ori_randomforest = (bp_pred_randomforest * (max(bp_test_ori) - min(bp_test_ori))) + min(bp_test_ori)
        '''print(len(bp_pred_ori_randomforest), bp_pred_ori_randomforest)
        plt.figure(figsize=(30,6))
        plt.plot(bp_pred_randomforest)
        plt.plot(test_features_wave[i][0])
        plt.show()
        print(len(bp_test_ori), bp_test_ori)
        print(len(test_features_wave[i][0]), test_features_wave[i][0])
        print(len(ppg_signal[:len(bp_pred_ori_randomforest)]), ppg_signal[:len(bp_pred_ori_randomforest)])
        print(len(test_features_wave[i][1]), test_features_wave[i][1])'''
        print(i)
        
        correlation_matrix_1 = np.corrcoef(bp_test_ori, bp_pred_ori_randomforest)
        correlation_coefficient_1 = correlation_matrix_1[0, 1]
        correlation_matrix_2 = np.corrcoef(test_features_wave[i][0], bp_pred_ori_randomforest)
        correlation_coefficient_2 = correlation_matrix_2[0, 1]
        ppg_signal = data['p'][0][patient60s_saved1[400+i]][:, :7500][0]
        correlation_matrix_3 = np.corrcoef(ppg_signal[:len(bp_pred_ori_randomforest)], bp_pred_ori_randomforest)
        correlation_coefficient_3 = correlation_matrix_3[0, 1]
        correlation_matrix_4 = np.corrcoef(test_features_wave[i][1], bp_pred_ori_randomforest)
        correlation_coefficient_4 = correlation_matrix_4[0, 1]
        cc_list = [correlation_coefficient_1, correlation_coefficient_2, correlation_coefficient_3, correlation_coefficient_4]
        wave_ppg_ecg_df.loc[len(wave_ppg_ecg_df)] = cc_list
        print(cc_list)
        plt.figure(figsize=(15,6), dpi=300)
        plt.plot(bp_test_ori[:500], label = 'ref BP')
        plt.plot(bp_pred_ori_randomforest[:500], label = 'est BP')
        plt.legend(loc='upper center', prop={'size': 14})
        plt.title(f'CC between ref BP and est BP is {round(cc_list[0], 4)}', fontsize = 18)
        plt.show()
    #wave_ppg_ecg_df.to_csv('wave_ppg_ecg_random_20test_30s_21Jul.csv')
    return






