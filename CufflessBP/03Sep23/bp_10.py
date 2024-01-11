## 12 Sep 23 Modify:



'''## Check GPU 
import tensorflow as tf
tf.test.gpu_device_name()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"'''

## 1. Open a patient data from 1000 patients mat file of UCI dataset
import scipy.io
part = 1
data = scipy.io.loadmat(f'/home/hanjiechen/YanweiJIN/Data/Cuffless_BP_Estimation/part_{part}.mat')

## 2. Define Functions

import numpy as np
import matplotlib.pyplot as plt
import os

def get_ori_signal(patient_data, iftext=1, figtime=1, timelength=10):
    pt_ori = {'PPG': patient_data[0], 'BP': patient_data[1], 'ECG': patient_data[2], 'Type': 'Original'}
    if iftext == 1:
        print(f'Patient{patient} ori signal: {pt_ori}')
    if figtime != 0:
        show_3_signals(pt_ori, figtime, timelength)
    return pt_ori

def show_3_signals(signal, figtime=1, timelength=10):
    begin_index = (figtime-1) * timelength * fs
    end_index = figtime * timelength * fs
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(18, 6), dpi=96)
    axs[0].plot(list(signal.values())[0][begin_index : end_index])
    axs[0].set_title(list(signal.keys())[0])

    axs[1].plot(list(signal.values())[1][begin_index : end_index])
    axs[1].set_title(list(signal.keys())[1])

    axs[2].plot(list(signal.values())[2][begin_index : end_index])
    axs[2].set_title(list(signal.keys())[2])

    xtick_positions = np.arange(0, timelength * fs , fs)
    second_index = np.arange(begin_index, end_index, fs)
    xtick_labels = [f'{int(second/fs)}' for second in second_index]
    plt.xticks(xtick_positions, xtick_labels)
    plt.xlabel('Time (seconds)', fontsize=12)
    fig.suptitle(f'Patient{patient} {signal["Type"]} Signals')
    plt.tight_layout()
    plt.savefig(f'{patient_path}/Part{part}_Patient{patient}_{((figtime-1) * timelength)}to{figtime * timelength}Sec_{signal["Type"]}_signals.png', format='png')
    plt.show()
    return 

def get_norm_signal(pt_ori, iftext=0, figtime=0, timelength=10):
    pt_norm = {}
    pt_norm['PPG'] = (pt_ori['PPG']-min(pt_ori['PPG'])) / (max(pt_ori['PPG'])-min(pt_ori['PPG']))
    pt_norm['BP'] = (pt_ori['BP']-min(pt_ori['BP'])) / (max(pt_ori['BP'])-min(pt_ori['BP']))
    #pt_norm['BP'] = pt_ori['BP']
    pt_norm['ECG'] = (pt_ori['ECG']-min(pt_ori['ECG'])) / (max(pt_ori['ECG'])-min(pt_ori['ECG']))
    pt_norm['Type'] = 'Normlized'
    if iftext == 1:
        print(f'Patient{patient} norm signal: {pt_norm}')
    if figtime != 0:
        show_3_signals(pt_norm, figtime, timelength)
    return pt_norm

import pandas as pd
import math

def find_points_in_all_beats_index(signal, iftext=1, figtime=1, timelength=10, height_cf=0.8, p_lth=2, p_ht=0.17):
    points_in_all_beats_index = pd.DataFrame(columns=(['P start index', 'P peak index', 'Q peak index', 'R peak index', 'S peak index', 'S end index', 'T peak index', 'T end index', 'ECG Min index', 'DBP index', 'SBP index', 'PPG valley index', 'PPG Max Slope index', 'PPG peak index']))
    ## 1. Find ECG Points
    ecg_r_height = min(signal['ECG']) + ((max(signal['ECG']) - min(signal['ECG'])) * height_cf)
    #print(f'ECG R peak height > {ecg_r_height}')

    r_peak_index = []
    for i in range(1, (len(signal['ECG'])-1)):
        if (signal['ECG'][i] > signal['ECG'][i-1]) and (signal['ECG'][i] >= signal['ECG'][i+1]) and (signal['ECG'][i] >= ecg_r_height):
            r_peak_index.append(i)
    points_in_all_beats_index['R peak index'] = r_peak_index
    #print(f'{len(r_peak_index)} r peak index: {r_peak_index}')

    ecg_min_index = []
    for index0, index1 in zip(r_peak_index[:-1], r_peak_index[1:]):
        ecg_min_index.append(index0 + np.argmin(signal['ECG'][index0:index1]))
    ecg_min_index.append(r_peak_index[-1] + np.argmin(signal['ECG'][r_peak_index[-1]:]))
    points_in_all_beats_index['ECG Min index'] = ecg_min_index
    #print(f'{len(ecg_min_index)} ecg min index: {ecg_min_index}')   

    s_peak_index = []
    for index0, index1 in zip(r_peak_index, ecg_min_index):
        ecg_second_half = signal['ECG'][index0:index1]
        s_peak_in_one_beat_index = []
        s_peak_in_one_beat = []
        for i in range(index0+1, index1-1):
            if (signal['ECG'][i] < signal['ECG'][i-1]) and (signal['ECG'][i] < signal['ECG'][i+1]):
                s_peak_in_one_beat_index.append(i)
                s_peak_in_one_beat.append(signal['ECG'][i])
        if len(s_peak_in_one_beat_index) > 0:
            s_peak_index.append(s_peak_in_one_beat_index[s_peak_in_one_beat.index(min(s_peak_in_one_beat))])
        else:
            s_peak_index.append(index1)
    points_in_all_beats_index['S peak index'] = s_peak_index
    #print(f'{len(s_peak_index)} s peak index: {s_peak_index}')   

    p_peak_index = []
    p_peak_in_one_beat_index = []
    for i in range(1, r_peak_index[0]):
        if (signal['ECG'][i] > signal['ECG'][i-1]) and (signal['ECG'][i] > signal['ECG'][i+1]):
            p_peak_in_one_beat_index.append(i)
    if len(p_peak_in_one_beat_index) > 0:
        p_length = (s_peak_index[0] - r_peak_index[0]) * p_lth
        p_height = (signal['ECG'][r_peak_index[0]] - signal['ECG'][s_peak_index[0]]) * p_ht
        p_peak_right_index = []
        for p in p_peak_in_one_beat_index:
            if ((r_peak_index[0] - p) >= p_length) and ((signal['ECG'][p] - signal['ECG'][s_peak_index[0]]) >= p_height):
                p_peak_right_index.append(p)
        if len(p_peak_right_index) > 0:
            p_peak_index.append(p_peak_right_index[-1])
        else:
            p_peak_index.append(np.nan)
    else:
        p_peak_index.append(np.nan)

    for i in range(len(s_peak_index) - 1):
        #print(i)
        index_one = s_peak_index[i]
        index_two = r_peak_index[i + 1]
        p_peak_in_one_beat_index = []
        for j in range(index_one, index_two):
            if (signal['ECG'][j] >= signal['ECG'][j - 1]) and (signal['ECG'][j] > signal['ECG'][j + 1]):
                p_peak_in_one_beat_index.append(j)
        if len(p_peak_in_one_beat_index) > 0:
            #print(f'len(p_peak_in_one_beat_index) > 0: {p_peak_in_one_beat_index}')
            p_length = (s_peak_index[i+1] - r_peak_index[i+1]) * p_lth
            #print(f'p_length {p_length}')
            p_height = (signal['ECG'][r_peak_index[i+1]] - signal['ECG'][s_peak_index[i+1]]) * p_ht
            #print(f'p_height{p_height}')
            p_peak_right_index = []
            for p in p_peak_in_one_beat_index:
                #print(f'have a p:{p}')
                #print((r_peak_index[i+1] - p))
                if ((r_peak_index[i+1] - p) >= p_length) and ((signal['ECG'][p] - signal['ECG'][s_peak_index[i+1]]) >= p_height):
                    #print(f'have a p:{p}')
                    p_peak_right_index.append(p)
            if len(p_peak_right_index) > 0:
                #print('len(p_peak_right_index) > 0')
                p_peak_index.append(p_peak_right_index[-1])
            else:
                p_peak_index.append(np.nan)
        else:
            p_peak_index.append(np.nan)
    points_in_all_beats_index['P peak index'] = p_peak_index
    #print(f'{len(p_peak_index)} p peak index: {p_peak_index}')   

    q_peak_index = []
    for i in range(len(p_peak_index)):
        if math.isnan(p_peak_index[i]):
            q_peak_index.append(np.nan)
        else:
            q_peak_index.append(p_peak_index[i] + np.argmin(signal['ECG'][p_peak_index[i]:r_peak_index[i]]))
    points_in_all_beats_index['Q peak index'] = q_peak_index
    #print(f'{len(q_peak_index)} q peak index: {q_peak_index}')

    p_start_index = []
    p_start_in_one_beat_index = []
    if np.isnan(p_peak_index[0]):
        p_start_index.append(np.nan)
    else:
        for i in range(1, p_peak_index[0]):
            if signal['ECG'][i-1] == signal['ECG'][i] == signal['ECG'][i+1]:
                p_start_in_one_beat_index.append(i+1)
        if len(p_start_in_one_beat_index) > 0:
            p_start_index.append(p_start_in_one_beat_index[-1])
        else:
            p_start_index.append(p_peak_index[0] - (q_peak_index[0] - p_peak_index[0]))

    for i in range(len(s_peak_index) -1):
        if math.isnan(p_peak_index[i+1]):
            p_start_index.append(np.nan)
        else:
            p_start_in_one_beat_index = []
            for j in range((s_peak_index[i]+1),(p_peak_index[i+1]-1)):
                if signal['ECG'][j-1] == signal['ECG'][j] == signal['ECG'][j+1]:
                    p_start_in_one_beat_index.append(j+1)
            if len(p_start_in_one_beat_index) > 0:
                p_start_index.append(p_start_in_one_beat_index[-1])
            else:
                p_start_index.append(p_peak_index[i+1] - (q_peak_index[i+1] - p_peak_index[i+1]))
    points_in_all_beats_index['P start index'] = p_start_index
    #print(f'{len(p_start_index)} p start index: {p_start_index}')

    t_peak_index = []
    for i in range(len(s_peak_index)-1):
        t_peak_in_one_beat = 0
        if math.isnan(p_start_index[i+1]):
            index0 = s_peak_index[i]
            index1 = r_peak_index[i+1]
        else:
            index0 = s_peak_index[i]
            index1 = p_start_index[i+1]

        for j in range((index0+1), (index1-1)):
            if (signal['ECG'][j] > signal['ECG'][j-1]) and (signal['ECG'][j] > signal['ECG'][j+1]):
                t_peak_index.append(j)
                t_peak_in_one_beat += 1
                #print(j)
                break
        if t_peak_in_one_beat == 0:
            t_peak_index.append(np.nan)
            #print('nan')

    t_peak_in_one_beat = 0
    for i in range((s_peak_index[-1]+1),(len(signal['ECG'])-1)):
        if signal['ECG'][i] > signal['ECG'][i-1] and signal['ECG'][i] > signal['ECG'][i+1]:
            t_peak_index.append(i)
            t_peak_in_one_beat += 1
            break
    if t_peak_in_one_beat == 0:
        t_peak_index.append(np.nan)
    points_in_all_beats_index['T peak index'] = t_peak_index
    #print(f'{len(t_peak_index)} t peak index: {t_peak_index}')

    ## 2. Find BP Points
    sbp_index = []
    dbp_index = []
    for i in range(len(r_peak_index) - 1):
        index0 = r_peak_index[i]
        index1 = r_peak_index[i+1]
        sbp_index.append(index0 + np.argmax(signal['BP'][index0:index1]))
        if sbp_index[-1] > index0:
            dbp_index.append(index0 + np.argmin(signal['BP'][index0:sbp_index[-1]]))
        else:
            dbp_index.append(np.nan)
    for i in range(r_peak_index[-1]+1,len(signal['BP'])-1):
        if (signal['BP'][i] <= signal['BP'][i-1]) and (signal['BP'][i] < signal['BP'][i+1]):
            dbp_index.append(i)
            break
    if dbp_index[-1] > sbp_index[-1]:
            for i in range(dbp_index[-1]+1, len(signal['BP'])-1):
                if (signal['BP'][i] > signal['BP'][i-1]) and (signal['BP'][i] >= signal['BP'][i+1]):
                    sbp_index.append(i)
                    break
            if len(sbp_index) == len(dbp_index):
                pass
            else:
                sbp_index.append(np.nan)
    else:
        sbp_index.append(np.nan)
        dbp_index.append(np.nan)

    points_in_all_beats_index['SBP index'] = sbp_index
    points_in_all_beats_index['DBP index'] = dbp_index
    #print(f'{len(sbp_index)} sbp index: {sbp_index}, \n{len(dbp_index)} dbp index: {dbp_index}')

    ## 3. Find PPG Points
    ppg_peak_index = []
    ppg_valley_index = []
    ppg_max_slope_index = []

    def get_ppg_points(site=(len(sbp_index)-1)): #site=(len(sbp_index)-1)
        for i in range(site):
            if not np.isnan(dbp_index[i]):
                index0 = sbp_index[i]
                index1 = sbp_index[i+1]
                ppg_peak_index.append(index0 + np.argmax(signal['PPG'][index0:index1]))
                #print(ppg_peak_index)
                index0 = dbp_index[i]
                index1 = ppg_peak_index[-1]
            #if index1 > index0:
                ppg_valley_index.append(index0 + np.argmin(signal['PPG'][index0:index1]))
            #else:
                #ppg_valley_index.append(index0)
                max_slope = 0
                slope_index = ppg_valley_index[-1]
                for j in range(ppg_valley_index[-1],ppg_peak_index[-1]):
                    one_slope = signal['PPG'][j+1] - signal['PPG'][j]
                    if one_slope > max_slope:
                        max_slope = one_slope
                        slope_index = j
                ppg_max_slope_index.append(slope_index)
            else:
                ppg_peak_index.append(np.nan)
                ppg_max_slope_index.append(np.nan)
                ppg_valley_index.append(np.nan)
                
        for i in range(sbp_index[site]+1,len(signal['PPG'])-1):
            if (signal['PPG'][i] < signal['PPG'][i-1]) and (signal['PPG'][i] < signal['PPG'][i+1]):
                ppg_valley_index.append(i)
                break
        #print(f'ppg_valley_index[-1]:{ppg_valley_index[-1]}, ppg_peak_index[-1]:{ppg_peak_index[-1]}')
        if len(ppg_valley_index) > len(ppg_peak_index):
            #print('yes')
            for j in range(ppg_valley_index[-1]+1, len(signal['PPG'])-1):
                if (signal['PPG'][j] > signal['PPG'][j-1]) and (signal['PPG'][j] > signal['PPG'][j+1]):
                    ppg_peak_index.append(j)
                    break
            if ppg_peak_index[-1] > ppg_valley_index[-1]:
                max_slope = 0
                slope_index = ppg_valley_index[-1]
                for k in range(ppg_valley_index[-1], ppg_peak_index[-1]-1):
                    one_slope = signal['PPG'][k+1] - signal['PPG'][k]
                    if one_slope > max_slope:
                        max_slope = one_slope
                        slope_index = k
                ppg_max_slope_index.append(slope_index)
            else:
                ppg_peak_index.append(np.nan)
                ppg_max_slope_index.append(np.nan)
        else:
            ppg_peak_index.append(np.nan)
            ppg_max_slope_index.append(np.nan)
            ppg_valley_index.append(np.nan)
        
        return

    if not np.isnan(sbp_index[-1]):
        site = len(sbp_index) - 1
        get_ppg_points(site)

    else:
        site = len(sbp_index) - 2
        get_ppg_points(site)
        ppg_peak_index.append(np.nan)
        ppg_max_slope_index.append(np.nan)
        ppg_valley_index.append(np.nan)

    #print(f'{len(ppg_peak_index)} ppg peak index: {ppg_peak_index}, \n{len(ppg_valley_index)} ppg valley index: {ppg_valley_index}, \n{len(ppg_max_slope_index)} ppg max slope index: {ppg_max_slope_index}')
    points_in_all_beats_index['PPG peak index'] = ppg_peak_index
    points_in_all_beats_index['PPG valley index'] = ppg_valley_index
    points_in_all_beats_index['PPG Max Slope index'] = ppg_max_slope_index
    points_in_all_beats_index.to_csv(f'{patient_path}/Part{part}_Patient{patient}_{n_seconds_to_load}Sec_{len(points_in_all_beats_index)}points.csv', index=False)
    print(f'All points: {points_in_all_beats_index.shape}')
    if iftext == 1:
        print(points_in_all_beats_index)
    if figtime != 0:
        show_points_in_signals(points_in_all_beats_index, pt_ori, figtime, timelength)
    return points_in_all_beats_index

def show_points_in_signals(points, signal, figtime=1, timelength=10):
    begin_index = (figtime-1) * timelength * fs
    end_index = figtime * timelength * fs

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(18,6), dpi=96)
    axs[0].plot(signal['PPG'][begin_index : end_index])
    p_p_i = [int(i) for i in points['PPG peak index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[0].scatter(p_p_i, signal['PPG'][p_p_i])
    p_v_i = [int(i) for i in points['PPG valley index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[0].scatter(p_v_i, signal['PPG'][p_v_i])
    p_m_s_i = [int(i) for i in points['PPG Max Slope index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[0].scatter(p_m_s_i, signal['PPG'][p_m_s_i])
    axs[0].set_title(f'{list(signal.keys())[0]}: PPG peak, PPG valley, PPG Max Slope')

    axs[1].plot(signal['BP'][begin_index : end_index])
    s_i = [int(i) for i in points['SBP index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[1].scatter(s_i, signal['BP'][s_i])
    d_i = [int(i) for i in points['DBP index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[1].scatter(d_i, signal['BP'][d_i])
    axs[1].set_title(f'{list(signal.keys())[1]}: SBP, DBP')

    axs[2].plot(signal['ECG'][begin_index : end_index])
    p_s_i = [int(i) for i in points['P start index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[2].scatter(p_s_i, signal['ECG'][p_s_i], label='Start')
    p_pk_i = [int(i) for i in points['P peak index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[2].scatter(p_pk_i, signal['ECG'][p_pk_i], label='P peak')
    q_pk_i = [int(i) for i in points['Q peak index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[2].scatter(q_pk_i, signal['ECG'][q_pk_i], label='Q peak')
    r_pk_i = [int(i) for i in points['R peak index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[2].scatter(r_pk_i, signal['ECG'][r_pk_i], label='R peak')
    s_pk_i = [int(i) for i in points['S peak index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[2].scatter(s_pk_i, signal['ECG'][s_pk_i], label='S peak')
    t_pk_i = [int(i) for i in points['T peak index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    axs[2].scatter(t_pk_i, signal['ECG'][t_pk_i], label='T peak')
    #e_m_i = [int(i) for i in points['ECG Min index'] if ((not np.isnan(i)) and (begin_index <= i < end_index))]
    #axs[2].scatter(e_m_i, signal['ECG'][e_m_i], label='Ecg min')
    axs[2].set_title(f'{list(signal.keys())[2]}: Start, P-peak, QRS-peak, T-peak, ECG Min')
    axs[2].legend()

    xtick_positions = np.arange(0, timelength * fs, fs)
    second_index = np.arange(begin_index, end_index, fs)
    xtick_labels = [f'{int(second/fs)}' for second in second_index]
    plt.xticks(xtick_positions, xtick_labels)
    plt.xlabel('Time (seconeds)', fontsize=12)
    fig.suptitle(f'Part{part}_Patient{patient} key points {len(points)} groups')
    plt.tight_layout()
    plt.savefig(f'{patient_path}/Part{part}_Patient{patient}_{n_seconds_to_load}Sec_{len(points)}Group_Points.png', format='png')
    plt.show()
    return

## Save normal Points and get Features
def get_points_saved_index(points, iftext=1, figtime=1, timelength=10):
    points_saved_index = pd.DataFrame(columns=(['P start index', 'P peak index', 'Q peak index', 'R peak index', 'S peak index', 'S end index', 'T peak index', 'T end index', 'ECG Min index', 'DBP index', 'SBP index', 'PPG valley index', 'PPG Max Slope index', 'PPG peak index']))
    #display(points_saved_index)
    for i in range(len(points)-1):
        #print(f'index:{i}')
        r_r = (points['R peak index'][i+1] - points['R peak index'][i]) / fs
        #print(f'r_r:{r_r}')
        if 1.2 >= r_r >= 0.4:
            if not np.isnan(points['P peak index'][i]):
                p_wave = ((points['P peak index'][i] - points['P start index'][i]) / fs) * 2
                p_r = (points['R peak index'][i] - points['P peak index'][i]) / fs
                #print(f'p_r:{p_r}')
                qrs_wave = (points['S peak index'][i] - points['Q peak index'][i]) / fs
                #print(f'qrs_wave:{qrs_wave}')
                if (p_wave > 0.1) and ((r_r/6) <= p_r <= (r_r/3)) and (qrs_wave <= 0.12):
                    if not np.isnan(points['DBP index'][i]):
                        points_saved_index = pd.concat([points_saved_index, points.iloc[i:i+1]])
    points_saved_index.to_csv(f'{patient_path}/Part{part}_Patient{patient}_{n_seconds_to_load}Sec_{len(points_saved_index)}Savedpoints.csv', index=False)
    print(f'Saved points: {points_saved_index.shape}')
    if iftext == 1:
        print(points_saved_index)
    if figtime != 0:
        show_points_in_signals(points_saved_index, pt_norm, figtime, timelength)
    return points_saved_index
#points_saved_index = get_points_saved_index()

def get_features_bps(points, signal, iftext=1):
    points.reset_index(drop=True, inplace=True)
    features = pd.DataFrame()
    saved_features = pd.DataFrame()
    features['P wavelength'] = ((points['P peak index'] - points['P start index']) / fs) * 2
    features['P amplitude'] = [signal['ECG'][int(i)] for i in points['P peak index']]
    features['Q amplitude'] = [signal['ECG'][int(i)] for i in points['Q peak index']]
    features['R amplitude'] = [signal['ECG'][int(i)] for i in points['R peak index']]
    features['S amplitude'] = [signal['ECG'][int(i)] for i in points['S peak index']]
    features['QRS wavelength'] = (points['S peak index'] - points['Q peak index']) / fs
    features['P-R interval'] = (points['R peak index'] - points['P peak index']) / fs
    #features['R-R interval half'] = [((points['R peak index'][i+1] - points['R peak index'][i]) / (2 * fs)) for i in range(len(points)-1)]+[np.nan]
    features['PTT R-PPG peak'] = (points['PPG peak index'] - points['R peak index']) / fs
    features['PTT R-PPG Max Slope'] = (points['PPG Max Slope index'] - points['R peak index']) / fs
    features['PPG valley amplitude'] = [signal['PPG'][int(i)] for i in points['PPG valley index']]
    features['PPG peak amplitude'] = [signal['PPG'][int(i)] for i in points['PPG peak index']]
    features['PPG valley-slope interval'] = (points['PPG Max Slope index'] - points['PPG valley index']) / fs
    features['PPG valley-peak interval'] = (points['PPG peak index'] - points['PPG valley index']) / fs
    features['PPG slope'] = [((signal['PPG'][int(points['PPG peak index'][i])] - signal['PPG'][int(points['PPG valley index'][i])])/ (points['PPG peak index'][i] - points['PPG valley index'][i])) for i in range(len(points))]
    #features['PPG val-val interval half'] = [((points['PPG valley index'][i+1] - points['PPG valley index'][i]) / (2 * fs)) for i in range(len(points)-1)]+[np.nan]
    #features['PPG pk-pk interval half'] = [((points['PPG peak index'][i+1] - points['PPG peak index'][i]) / (2 * fs)) for i in range(len(points)-1)]+[np.nan]
    features['SBP'] = [signal['BP'][int(i)] for i in points['SBP index']]
    features['DBP'] = [signal['BP'][int(i)] for i in points['DBP index']]
    features['PBP'] = features['SBP'] - features['DBP']
    print(f'All features: {features.shape}')
    for i in range(len(features)):
        if (np.max(features.iloc[i, :-3]) <= 1) and (np.min(features.iloc[i, :-3]) >= 0):
            saved_features = pd.concat([saved_features, features.iloc[i:i+1]])
    saved_features.to_csv(f'{patient_path}/Part{part}_Patient{patient}_{n_seconds_to_load}Sec_{len(saved_features)}features.csv', index=False)
    print(f'Saved features: {saved_features.shape}')
    if iftext == 1:
        print(saved_features)
    return saved_features


import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
class ABPLoss(nn.Module):
    def __init__(self):
        super(ABPLoss, self).__init__()

    def forward(self, predicted, target):
        # Custom loss calculation
        pred_bp = predicted * (max_bp - min_bp) + min_bp
        targ_bp = target * (max_bp - min_bp) + min_bp
        mse_loss = nn.MSELoss()
        loss = mse_loss(pred_bp, targ_bp) # Define your loss calculation here
        #print(f'ABP loss: {loss.item()}')
        return loss
    
class PBPLoss(nn.Module):
    def __init__(self):
        super(PBPLoss, self).__init__()

    def forward(self, predicted, target):
        # Custom loss calculation
        pred_bp = predicted * (max_bp - min_bp)
        targ_bp = target * (max_bp - min_bp)
        mse_loss = nn.MSELoss()
        loss = mse_loss(pred_bp, targ_bp) # Define your loss calculation here
        return loss

def est_bp_lstm(features_bps):
    features = features_bps.iloc[:, :-3]
    bps = features_bps.iloc[:, -3:]
    #print(f'features shape: {features.shape}, \n{features} \nbps shape: {bps.shape}, \n{bps}')
    features_reshape = np.zeros(((features.shape[0] - time_step + 1), time_step, features.shape[1]))
    bps_reshape = np.zeros(((bps.shape[0] - time_step + 1), time_step, bps.shape[1]))
    for i in range(len(features_reshape)):
        features_reshape[i] = features[i : (time_step + i)]
        bps_reshape[i] = bps[i : (time_step + i)]
    train_length = int(split_rate * len(features_reshape))
    X_train = torch.Tensor(features_reshape[ : train_length])
    X_test = torch.Tensor(features_reshape[train_length: ])
    y_train = {}
    y_train['SBP'] = torch.Tensor(bps_reshape[ : train_length, : , 0])
    y_train['DBP'] = torch.Tensor(bps_reshape[ : train_length, : , 1])
    y_train['PBP'] = torch.Tensor(bps_reshape[ : train_length, : , 2])
    #y_train['Type'] = 'Train'
    y_test = {}
    y_test['SBP'] = torch.Tensor(bps_reshape[train_length : , : , 0])
    y_test['DBP'] = torch.Tensor(bps_reshape[train_length : , : , 1])
    y_test['PBP'] = torch.Tensor(bps_reshape[train_length : , : , 2])
    #y_test['Type'] = 'Test'

    sbp_model = LSTMModel(input_size=features_reshape.shape[2], hidden_size=64, num_layers=2, output_size=time_step)
    dbp_model = LSTMModel(input_size=features_reshape.shape[2], hidden_size=64, num_layers=2, output_size=time_step)
    pbp_model = LSTMModel(input_size=features_reshape.shape[2], hidden_size=64, num_layers=2, output_size=time_step)
    abp_loss_fn = ABPLoss()
    pbp_loss_fn = PBPLoss()

    ## Use GPU to run
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbp_model.to(device)
    dbp_model.to(device)
    pbp_model.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train['SBP'] = y_train['SBP'].to(device)
    #y_test['SBP'] = y_test['SBP'].to(device)
    y_train['DBP'] = y_train['DBP'].to(device)
    #y_test['DBP'] = y_test['DBP'].to(device)
    y_train['PBP'] = y_train['PBP'].to(device)
    #y_test['PBP'] = y_test['PBP'].to(device)
    abp_loss_function = abp_loss_fn.to(device)
    pbp_loss_function = pbp_loss_fn.to(device)

    ## Check if model is using GPU
    if next(sbp_model.parameters()).is_cuda:
        print("Model is using GPU.")
    else:
        print("Model is using CPU.")

    ## Fit BP models
    optimizer = torch.optim.Adam(sbp_model.parameters(), lr=0.001)
    num_epochs = 200
    for epoch in range(num_epochs):
        output = sbp_model(X_train)
        loss = abp_loss_function(output.squeeze(), y_train['SBP'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, loss: {loss.item()}')

    optimizer = torch.optim.Adam(dbp_model.parameters(), lr=0.001)
    num_epochs = 100
    for epoch in range(num_epochs):
        output = dbp_model(X_train)
        loss = abp_loss_function(output.squeeze(), y_train['DBP'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, loss: {loss.item()}')
        
    optimizer = torch.optim.Adam(pbp_model.parameters(), lr=0.001)
    num_epochs = 100
    for epoch in range(num_epochs):
        output = pbp_model(X_train)
        loss = pbp_loss_function(output.squeeze(), y_train['PBP'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, loss: {loss.item()}')

    ## Predict BP
    y_pred = {}
    y_pred['SBP'] = sbp_model(X_test)
    y_pred['DBP'] = dbp_model(X_test)
    y_pred['PBP'] = pbp_model(X_test)
    #y_pred['Type'] = 'Pred'


    results = {}
    results['BP'] = ['SBP', 'DBP', 'PBP']
    results['ref_BP'] = [y_test['SBP'][:, -1] * (max_bp - min_bp) + min_bp, y_test['DBP'][:, -1] * (max_bp - min_bp) + min_bp, y_test['PBP'][:, -1] * (max_bp - min_bp)]
    results['pred_BP'] = [(y_pred['SBP'][:, -1] * (max_bp - min_bp) + min_bp).cpu(), (y_pred['DBP'][:, -1] * (max_bp - min_bp) + min_bp).cpu(), (y_pred['PBP'][:, -1] * (max_bp - min_bp)).cpu()]
    results['MAE'] = ["{:.4f}".format(torch.mean(torch.abs(results['ref_BP'][i] - results['pred_BP'][i])).item()) for i in range(3)]
    results['RMSE'] = ["{:.4f}".format(torch.sqrt(torch.mean((results['ref_BP'][i] - results['pred_BP'][i]) ** 2)).item()) for i in range(3)]

    ## Compare BPs
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(18, 6), dpi=96)
    for i in range(3):
        axs[i].plot(results['ref_BP'][i], label='Reference BP')
        axs[i].plot(results['pred_BP'][i].cpu().detach().numpy(), label='Predict BP')
        axs[i].set_title(f'{results["BP"][i]} MAE: {results["MAE"][i]}, RMSE: {results["RMSE"][i]}')
        axs[i].legend()
    fig.suptitle(f'Part{part} Patient{patient} {n_seconds_to_load/60}min LSTM Model')
    plt.tight_layout()
    plt.savefig(f'{patient_path}/Part{part}_Patient{patient}_{n_seconds_to_load/60}min_LSTM.png', format='png')
    plt.show()
    return


## Run all process
fs = 125
n_seconds_to_load = 300 # 5min
signal_length = int(n_seconds_to_load * fs)
iftext = 0
figtime = 1
timelength = 20

time_step = 10
split_rate = 0.7
patient_list = [i for i in range(30)]
for patient in patient_list:
    patient_data = data['p'][0,patient]
    time_length = patient_data.shape[1] / fs
    if time_length < n_seconds_to_load:
        continue
    print(f'Patient{patient} time length: {time_length} sec, data length: {patient_data.shape[1]}')
    patient_path = f'/home/hanjiechen/YanweiJIN/BP_est/bp_result/part{part}/patient{patient}'
    os.makedirs(patient_path, exist_ok=True)
    patient_data = patient_data[:,:signal_length] # Shape:(3,37500)
    pt_ori = get_ori_signal(patient_data, iftext=iftext, figtime=figtime, timelength=timelength)
    max_bp = np.max(pt_ori['BP'])
    print(max_bp)
    min_bp = np.min(pt_ori['BP'])
    pt_norm = get_norm_signal(pt_ori, iftext=0, figtime=1, timelength=20)
    points_in_all_beats_index = find_points_in_all_beats_index(signal=pt_ori, iftext=iftext)
    points_saved_index = get_points_saved_index(points=points_in_all_beats_index, iftext=iftext)
    features_bps = get_features_bps(points=points_saved_index, signal=pt_norm, iftext=iftext) 
    if len(features_bps) >= 200:
        est_bp_lstm(features_bps)
