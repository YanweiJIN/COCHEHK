'''
Created data: 28 Aug 2023
Last modified: 28 Aug 2023
Editor: Yanwei JIN (HKCOCHE)
Supervisors: Prof. Beeluan KHOO, Prof. Rosa CHAN and Prof. Raymond CHAN
Introduction: Corroct ECG signals to normal shape.
'''


## Correct ECG signal
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
    #print(s_second_part_index)
    min_i = ecg_signal[s_peak_index[i+1]]
    max_i = ecg_signal[p_start_index[i]]
    for ecg_i in s_second_part_index:
        ecg_corr.append((((ecg_signal[ecg_i] - min_i) / (ecg_signal[s_end_index[i]] - min_i)) * (max_i - min_i) + min_i) + ecg_move)
    #print(ecg_corr[-1])
    #print(ecg_corr[p_start_index[0]])
    for ecg_i in range(s_end_index[i], t_peak_index[i]):
        ecg_corr.append(ecg_signal[ecg_i] + (max_i - ecg_signal[s_end_index[i]]) + ecg_move)
    #print(ecg_corr[-1])
    #print(ecg_corr[p_peak_index[i]])
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
## Compare teo ECG signals    
fig = plt.figure(figsize=(30,6), dpi=96)
plt.plot(ecg_signal, color='blue', label='original ECG')
plt.plot(ecg_corr, color='orange', label='correted ECG')
plt.title('ECG comparison', fontsize=12)
plt.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.83, 1.155))
plt.show()
## Original ECG signals and points on it.
fig = plt.figure(figsize=(30,6),dpi=96)
plt.plot(ecg_norm, color='blue')
plt.scatter(r_peak_index, ecg_signal[r_peak_index], color='c', marker='o')
plt.scatter(ecg_min_index, ecg_signal[ecg_min_index], color='green', marker='o')
plt.scatter(s_peak_index, ecg_signal[s_peak_index], color='red', marker='o')
plt.scatter(p_peak_index, ecg_signal[p_peak_index], color='pink', marker='o')
plt.scatter(p_start_index, ecg_signal[p_start_index], color='blue', marker='o')
plt.scatter(s_end_index, ecg_signal[s_end_index], color='grey', marker='o')
plt.scatter(t_peak_index, ecg_signal[t_peak_index], color='orange', marker='o')
plt.scatter(t_end_index, ecg_signal[t_end_index], color='black', marker='o')
plt.scatter(q_peak_index, ecg_signal[q_peak_index], color='purple', marker='o')
plt.title('Sites in normed ECG', fontsize=12)
plt.show()
## Corrected ECG signals and points on it.
ecg_signal = np.array(ecg_corr)
fig = plt.figure(figsize=(30,6), dpi=96)
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
plt.show()
## Show three signal_corr
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,3),dpi=96)
plt.plot(ppg_norm, color='red', label='PPG_norm')
plt.plot(bp_norm, color='green', label='BP_norm')
plt.plot(ecg_corr, color='orange', label='ECG_corr')
xtick_positions = np.arange(0, patient_data.shape[1], fs)
xtick_labels = [f'{int(xtick_position/fs)}s' for xtick_position in xtick_positions]
plt.xticks(xtick_positions, xtick_labels)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Normalized Signal', fontsize=12)
plt.title(f'Signal of Patient {patient}\n', fontsize=12)
plt.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.83, 1.155))
plt.show()