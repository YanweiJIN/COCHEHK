'''
Created data: 17 Aug 2023
Last modified: 21 Aug 2023
Editor: Yanwei JIN (HKCOCHE)
Supervisors: Prof. Beeluan KHOO, Prof. Rosa CHAN and Prof. Raymond CHAN
Introduction: Recognize ECG type, Find ECG signals points, Correct ECG to standarlize
'''

## Import packages and Load the patient data
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
data = scipy.io.loadmat('/Users/jinyanwei/Desktop/BP_Model/Data/Cuffless_BP_Estimation/part_1.mat')

## Choose the ECG data
patient = 0
patient_data = data['p'][0,patient][:,:1000]
ecg_ori = patient_data[2]

## Find height, R peak, min peak, S peak, P peak, Q peak, P start, S end, T peak, T end indexs.
def get_ecg_points_index(ecg_signal):
    ecg_signal = ecg_ori
    ecg_r_height = min(ecg_signal) + ((max(ecg_signal) - min(ecg_signal)) * 0.6)
    print(ecg_r_height)

    r_peak_index = []
    for i in range(1, (len(ecg_signal)-1)):
        if (ecg_signal[i] > ecg_signal[i-1]) and (ecg_signal[i] >= ecg_signal[i+1]) and (ecg_signal[i] >= ecg_r_height):
            r_peak_index.append(i)
    print(f'{len(r_peak_index)} r peak index: {r_peak_index}')

    ecg_min_index = []
    for index0, index1 in zip(r_peak_index[:-1], r_peak_index[1:]):
        #print(f'index0: {index0}, index1: {index1}')
        ecg_min_index.append(index0 + np.argmin(ecg_signal[index0:index1]))
    print(f'{len(ecg_min_index)} ecg min index: {ecg_min_index}')   

    s_peak_index = []
    for index0, index1 in zip(r_peak_index[:-1], ecg_min_index):
        ecg_second_half = ecg_signal[index0:index1]
        #s_peak_in_one_peak = []
        #s_peak_in_one_peak_index = []
        for i in range(1, len(ecg_second_half) - 1):
            if (ecg_second_half[i] < ecg_second_half[i - 1]) and (ecg_second_half[i] <= ecg_second_half[i + 1]):
                #s_peak_in_one_peak.append(ecg_second_half[i])
                #s_peak_in_one_peak_index.append(index0 + i)
                #s_peak_true = min(s_peak_in_one_peak)
                #s_peak_index.append(s_peak_in_one_peak_index[s_peak_in_one_peak.index(s_peak_true)])
                s_peak_index.append(index0 + i)
                break
    print(f'{len(s_peak_index)} s peak index: {s_peak_index}')   

    p_peak_index = []
    for i in range(min(len(ecg_min_index),len(r_peak_index)) - 1):
        index_one = ecg_min_index[i]
        index_two = r_peak_index[i + 1]
        find_p_peak_list = []
        find_p_peak_index_list = []
        for i in range(index_one, index_two):
            if (ecg_signal[i] > ecg_signal[i - 1]) and (ecg_signal[i] > ecg_signal[i + 1]):
                find_p_peak_list.append(ecg_signal[i])
                find_p_peak_index_list.append(i)
        if len(find_p_peak_list) >= 1:
            p_peak_index.append(find_p_peak_index_list[find_p_peak_list.index(max(find_p_peak_list))])
    print(f'{len(p_peak_index)} p peak index: {p_peak_index}')   

    q_peak_index = []
    r_loc = 0
    for index_p in p_peak_index:
        for index_r in r_peak_index[r_loc:]:
            if index_r > index_p:
                q_peak_index.append(index_p + np.argmin(ecg_signal[index_p:index_r]))
                r_loc = r_peak_index.index(index_r)
                break
    print(f'{len(q_peak_index)} q peak index: {q_peak_index}')

    p_start_index = []
    s_end_index = []
    t_peak_index = []
    t_end_index = []
    for i in range(len(p_peak_index)):
        p_peak_half = q_peak_index[i] - p_peak_index[i]
        p_start_i = p_peak_index[i] - p_peak_half
        p_start_index.append(p_start_i)
        s_peak_half = s_peak_index[i+1] - r_peak_index[i+1]
        s_end_i = s_peak_index[i+1] + s_peak_half
        s_end_index.append(s_end_i)
        t_peak_i = list(ecg_signal[s_end_i:ecg_min_index[i+1]]).index(max(ecg_signal[t_i] for t_i in list(range(s_end_i, ecg_min_index[i+1])))) + s_end_i
        t_peak_index.append(t_peak_i)
        t_end_i = t_peak_i + (t_peak_i - s_end_i)
        t_end_index.append(t_end_i)
    print(f'{len(p_start_index)} p start index: {p_start_index}')
    print(f'{len(s_end_index)} s end index: {s_end_index}')
    print(f'{len(t_peak_index)} t peak index: {t_peak_index}')
    print(f'{len(t_end_index)} t end index: {t_end_index}')

## Correct ECG to standarlize
def correct_ecg():
    ecg_corr = []
    for i in range(ecg_min_index[0]):
        ecg_corr.append(ecg_signal[i])
    for i in range(len(ecg_min_index) - 1):
        for ecg_i in range(ecg_min_index[i], p_start_index[i]):
            ecg_corr.append(ecg_signal[p_start_index[0]])
        ecg_move = ecg_signal[p_start_index[0]] - ecg_signal[p_start_index[i]]
        for ecg_i in range(p_start_index[i], s_peak_index[i+1]):
            ecg_corr.append(ecg_signal[ecg_i] + ecg_move)
        s_second_part_index = list(range(s_peak_index[i+1], s_end_index[i]))
        print(s_second_part_index)
        min_i = ecg_signal[s_peak_index[i+1]]
        max_i = ecg_signal[p_start_index[i]]
        for ecg_i in s_second_part_index:
            ecg_corr.append((((ecg_signal[ecg_i] - min_i) / (ecg_signal[s_end_index[i]] - min_i)) * (max_i - min_i) + min_i) + ecg_move)
        print(ecg_corr[-1])
        print(ecg_corr[p_start_index[0]])
        for ecg_i in range(s_end_index[i], t_peak_index[i]):
            ecg_corr.append(ecg_signal[ecg_i] + (max_i - ecg_signal[s_end_index[i]]) + ecg_move)
        print(ecg_corr[-1])
        print(ecg_corr[p_peak_index[i]])
        min_t = ecg_signal[s_end_index[i]]
        max_t = ecg_signal[t_peak_index[i]]
        t_second_half = []
        for ecg_i in range(t_peak_index[i], t_end_index[i]):
            t_second_half_i = max_t - (((max_t - ecg_signal[ecg_i]) / (max_t - ecg_signal[t_end_index[i]])) * (max_t - min_t))
            ecg_corr.append(t_second_half_i + (max_i - ecg_signal[s_end_index[i]]) + ecg_move)
        for ecg_i in range(t_end_index[i],ecg_min_index[i+1]):
            ecg_corr.append(ecg_signal[p_start_index[0]])
    for i in range(ecg_min_index[-1], len(ecg_signal)):
        ecg_corr.append(ecg_signal[i])

## Show two ECG comparition
fig = plt.figure(figsize=(30,6), dpi=96)
plt.plot(ecg_ori, color='blue', label='original ECG')
plt.plot(ecg_corr, color='orange', label='correted ECG')
plt.title('ECG comparison', fontsize=12)
plt.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.83, 1.155))
plt.show()
## Show ECG signal and points on it
ecg_signal = ecg_ori
fig = plt.figure(figsize=(30,6),dpi=96)
plt.plot(ecg_ori, color='blue')
#plt.plot(ecg_corr)
plt.scatter(r_peak_index, ecg_signal[r_peak_index], color='c', marker='o')
plt.scatter(ecg_min_index, ecg_signal[ecg_min_index], color='green', marker='o')
plt.scatter(s_peak_index, ecg_signal[s_peak_index], color='red', marker='o')
plt.scatter(p_peak_index, ecg_signal[p_peak_index], color='pink', marker='o')
plt.scatter(p_start_index, ecg_signal[p_start_index], color='blue', marker='o')
plt.scatter(s_end_index, ecg_signal[s_end_index], color='grey', marker='o')
plt.scatter(t_peak_index, ecg_signal[t_peak_index], color='orange', marker='o')
plt.scatter(t_end_index, ecg_signal[t_end_index], color='black', marker='o')
plt.scatter(q_peak_index, ecg_signal[q_peak_index], color='purple', marker='o')
plt.title('Sites in original ECG', fontsize=12)
#plt.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.83, 1.155))
plt.show()

ecg_signal = np.array(ecg_corr)
fig = plt.figure(figsize=(30,6), dpi=96)
#plt.plot(ecg_ori)
plt.plot(ecg_corr, color='orange')
plt.scatter(r_peak_index, ecg_signal[r_peak_index], color='c', marker='o')
plt.scatter(ecg_min_index, ecg_signal[ecg_min_index], color='green', marker='o')
plt.scatter(s_peak_index, ecg_signal[s_peak_index], color='red', marker='o')
plt.scatter(p_peak_index, ecg_signal[p_peak_index], color='pink', marker='o')
plt.scatter(p_start_index, ecg_signal[p_start_index], color='blue', marker='o')
plt.scatter(s_end_index, ecg_signal[s_end_index], color='grey', marker='o')
plt.scatter(t_peak_index, ecg_signal[t_peak_index], color='orange', marker='o')
plt.scatter(t_end_index, ecg_signal[t_end_index], color='black', marker='o')
plt.scatter(q_peak_index, ecg_signal[q_peak_index], color='purple', marker='o')
plt.title('Sites in corrected ECG', fontsize=12)
#plt.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.83, 1.155))
plt.show()


