'''
Created data: 28 Aug 2023
Last modified: 28 Aug 2023
Editor: Yanwei JIN (HKCOCHE)
Supervisors: Prof. Beeluan KHOO, Prof. Rosa CHAN and Prof. Raymond CHAN
Introduction: find signals points, extract features
'''
####------------------------------------------------------- Version 4 ---------------------------------------------------------####
####------------------------------------------------------- Part1: Find Points on Signals ---------------------------------------------------------####

import pandas as pd
import math
ppg_signal = ppg_norm
bp_signal = bp_norm
ecg_signal = ecg_norm

points_in_all_beat_index = pd.DataFrame(columns=(['P start index', 'P peak index', 'Q peak index', 'R peak index', 'S peak index', 'S end index', 'T peak index', 'T end index', 'ECG Min index', 'DBP index', 'SBP index', 'PPG valley index', 'PPG Max Slope index', 'PPG peak index']))
## 1. Find ECG Points
ecg_r_height = min(ecg_signal) + ((max(ecg_signal) - min(ecg_signal)) * 0.6)
print(f'ECG R peak height > {ecg_r_height}')

r_peak_index = []
for i in range(1, (len(ecg_signal)-1)):
    if (ecg_signal[i] > ecg_signal[i-1]) and (ecg_signal[i] >= ecg_signal[i+1]) and (ecg_signal[i] >= ecg_r_height):
        r_peak_index.append(i)
points_in_all_beat_index['R peak index'] = r_peak_index
print(f'{len(r_peak_index)} r peak index: {r_peak_index}')

ecg_min_index = []
for index0, index1 in zip(r_peak_index[:-1], r_peak_index[1:]):
    ecg_min_index.append(index0 + np.argmin(ecg_signal[index0:index1]))
ecg_min_index.append(r_peak_index[-1] + np.argmin(ecg_signal[r_peak_index[-1]:]))
points_in_all_beat_index['ECG Min index'] = ecg_min_index
print(f'{len(ecg_min_index)} ecg min index: {ecg_min_index}')   

s_peak_index = []
for index0, index1 in zip(r_peak_index, ecg_min_index):
    ecg_second_half = ecg_signal[index0:index1]
    s_peak_in_one_beat_index = []
    s_peak_in_one_beat = []
    for i in range(index0+1, index1-1):
        if (ecg_signal[i] < ecg_signal[i-1]) and (ecg_signal[i] < ecg_signal[i+1]):
            s_peak_in_one_beat_index.append(i)
            s_peak_in_one_beat.append(ecg_signal[i])
    if len(s_peak_in_one_beat_index) > 0:
        s_peak_index.append(s_peak_in_one_beat_index[s_peak_in_one_beat.index(min(s_peak_in_one_beat))])
    else:
        s_peak_index.append(index1)
points_in_all_beat_index['S peak index'] = s_peak_index
print(f'{len(s_peak_index)} s peak index: {s_peak_index}')   

p_peak_index = []
p_peak_in_one_beat_index = []
for i in range(1, r_peak_index[0]):
    if (ecg_signal[i] > ecg_signal[i-1]) and (ecg_signal[i] > ecg_signal[i+1]):
        p_peak_in_one_beat_index.append(i)
if len(p_peak_in_one_beat_index) > 0:
    p_peak_index.append(p_peak_in_one_beat_index[-1])
else:
    p_peak_index.append(np.nan)

for i in range(len(s_peak_index) - 1):
    index_one = s_peak_index[i]
    index_two = r_peak_index[i + 1]
    p_peak_in_one_beat_index = []
    for j in range(index_one, index_two):
        if (ecg_signal[j] > ecg_signal[j - 1]) and (ecg_signal[j] > ecg_signal[j + 1]):
            p_peak_in_one_beat_index.append(j)
    if len(p_peak_in_one_beat_index) > 0:
        p_peak_index.append(p_peak_in_one_beat_index[-1])
    else:
        p_peak_index.append(np.nan)
points_in_all_beat_index['P peak index'] = p_peak_index
print(f'{len(p_peak_index)} p peak index: {p_peak_index}')   

q_peak_index = []
for i in range(len(p_peak_index)):
    if math.isnan(p_peak_index[i]):
        q_peak_index.append(np.nan)
    else:
        q_peak_index.append(p_peak_index[i] + np.argmin(ecg_signal[p_peak_index[i]:r_peak_index[i]]))
points_in_all_beat_index['Q peak index'] = q_peak_index
print(f'{len(q_peak_index)} q peak index: {q_peak_index}')

p_start_index = []
p_start_in_one_beat_index = []
if math.isnan(p_peak_index[0]):
    p_start_index.append(np.nan)
else:
    for i in range(1, p_peak_index[0]):
        if ecg_signal[i-1] == ecg_signal[i] == ecg_signal[i+1]:
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
            if ecg_signal[j-1] == ecg_signal[j] == ecg_signal[j+1]:
                p_start_in_one_beat_index.append(j+1)
        if len(p_start_in_one_beat_index) > 0:
            p_start_index.append(p_start_in_one_beat_index[-1])
        else:
            p_start_index.append(p_peak_index[i+1] - (q_peak_index[i+1] - p_peak_index[i+1]))
points_in_all_beat_index['P start index'] = p_start_index
print(f'{len(p_start_index)} p start index: {p_start_index}')

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
        if (ecg_signal[j] > ecg_signal[j-1]) and (ecg_signal[j] > ecg_signal[j+1]):
            t_peak_index.append(j)
            t_peak_in_one_beat += 1
            #print(j)
            break
    if t_peak_in_one_beat == 0:
        t_peak_index.append(np.nan)
        #print('nan')

t_peak_in_one_beat = 0
for i in range((s_peak_index[-1]+1),(len(ecg_signal)-1)):
    if ecg_signal[i] > ecg_signal[i-1] and ecg_signal[i] > ecg_signal[i+1]:
        t_peak_index.append(i)
        t_peak_in_one_beat += 1
        break
if t_peak_in_one_beat == 0:
    t_peak_index.append(np.nan)
points_in_all_beat_index['T peak index'] = t_peak_index
print(f'{len(t_peak_index)} t peak index: {t_peak_index}')

## 2. Find BP Points
sbp_index = []
dbp_index = []
for i in range(len(r_peak_index) - 1):
    index0 = r_peak_index[i]
    index1 = r_peak_index[i+1]
    sbp_index.append(index0 + np.argmax(bp_signal[index0:index1]))
    dbp_index.append(index0 + np.argmin(bp_signal[index0:index1]))
'''index0 = r_peak_index[-1]
index1 = len(bp_signal) - 1
sbp_index.append(index0 + np.argmax(bp_signal[index0:index1]))
dbp_index.append(index0 + np.argmin(bp_signal[index0:index1]))'''
sbp_index.append(np.nan)
dbp_index.append(np.nan)
points_in_all_beat_index['SBP index'] = sbp_index
points_in_all_beat_index['DBP index'] = dbp_index
print(f'{len(sbp_index)} sbp index: {sbp_index}, \n{len(dbp_index)} dbp index: {dbp_index}')

## 3. Find PPG Points
ppg_peak_index = []
ppg_valley_index = []
ppg_max_slope_index = []
for i in range(len(sbp_index)-2):
    index0 = sbp_index[i]
    index1 = sbp_index[i+1]
    ppg_peak_index.append(index0 + np.argmax(ppg_signal[index0:index1]))
    index0 = dbp_index[i]
    index1 = ppg_peak_index[-1]
    ppg_valley_index.append(index0 + np.argmin(ppg_signal[index0:index1]))
    max_slope = 0
    slope_index = ppg_valley_index[-1]
    for j in range(ppg_valley_index[-1],ppg_peak_index[-1]):
        one_slope = ppg_signal[j+1] - ppg_signal[j]
        if one_slope > max_slope:
            max_slope = one_slope
            slope_index = j
    ppg_max_slope_index.append(slope_index)

ppg_peak_index.append(np.nan)
ppg_max_slope_index.append(np.nan)
ppg_valley_index.append(sbp_index[-2] + np.argmin(ppg_signal[sbp_index[-2]:]))
ppg_peak_index.append(np.nan)
ppg_max_slope_index.append(np.nan)
ppg_valley_index.append(np.nan)
points_in_all_beat_index['PPG peak index'] = ppg_peak_index
points_in_all_beat_index['PPG valley index'] = ppg_valley_index
points_in_all_beat_index['PPG Max Slope index'] = ppg_max_slope_index
print(f'{len(ppg_peak_index)} ppg peak index: {ppg_peak_index}, \n{len(ppg_valley_index)} ppg valley index: {ppg_valley_index}, \n{len(ppg_max_slope_index)} ppg max slope index: {ppg_max_slope_index}')

display(points_in_all_beat_index)

points_df = points_in_all_beat_index

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15,6), dpi=96)
axs[0].plot(ppg_signal)
p_p_i = [int(i) for i in points_df['PPG peak index'] if not np.isnan(i)]
axs[0].scatter(p_p_i, ppg_signal[p_p_i])
p_v_i = [int(i) for i in points_df['PPG valley index'] if not np.isnan(i)]
axs[0].scatter(p_v_i, ppg_signal[p_v_i])
p_m_s_i = [int(i) for i in points_df['PPG Max Slope index'] if not np.isnan(i)]
axs[0].scatter(p_m_s_i, ppg_signal[p_m_s_i])
axs[0].set_title('PPG signal: PPG peak, PPG valley, PPG Max Slope')

axs[1].plot(bp_signal)
s_i = [int(i) for i in points_df['SBP index'] if not np.isnan(i)]
axs[1].scatter(s_i, bp_signal[s_i])
d_i = [int(i) for i in points_df['DBP index'] if not np.isnan(i)]
axs[1].scatter(d_i, bp_signal[d_i])
axs[1].set_title('BP signal: SBP, DBP')

axs[2].plot(ecg_signal)
p_s_i = [int(i) for i in points_df['P start index'] if not np.isnan(i)]
axs[2].scatter(p_s_i, ecg_signal[p_s_i], label='Start')
p_pk_i = [int(i) for i in points_df['P peak index'] if not np.isnan(i)]
axs[2].scatter(p_pk_i, ecg_signal[p_pk_i], label='P peak')
q_pk_i = [int(i) for i in points_df['Q peak index'] if not np.isnan(i)]
axs[2].scatter(q_pk_i, ecg_signal[q_pk_i], label='Q peak')
r_pk_i = [int(i) for i in points_df['R peak index'] if not np.isnan(i)]
axs[2].scatter(r_pk_i, ecg_signal[r_pk_i], label='R peak')
s_pk_i = [int(i) for i in points_df['S peak index'] if not np.isnan(i)]
axs[2].scatter(s_pk_i, ecg_signal[s_pk_i], label='S peak')
t_pk_i = [int(i) for i in points_df['T peak index'] if not np.isnan(i)]
axs[2].scatter(t_pk_i, ecg_signal[t_pk_i], label='T peak')
#e_m_i = [int(i) for i in points_df['ECG Min index'] if not np.isnan(i)]
#axs[2].scatter(e_m_i, ecg_signal[e_m_i], label='Ecg min')
axs[2].set_title('ECG signal: Start, P-peak, QRS-peak, T-peak, ECG Min')
axs[2].legend()

xtick_positions = np.arange(0, len(ecg_signal), fs)
xtick_labels = [f'{int(xtick_position/fs)}s' for xtick_position in xtick_positions]
plt.xticks(xtick_positions, xtick_labels)
plt.xlabel('Time (seconeds)', fontsize=12)
fig.suptitle(f'Patient {patient} key points')
plt.tight_layout()
plt.show()


## 2.1 Method 1: DBP and SBP in one R-R peak.

## 2.2 Method2: DBP and SBP in two R-R peak.



## 3.1 Method 1: Find max slope, min ppg before it nad max ppg after it. All points in one R-R peak.
## 3.2 Method 2: Find min ppg after max ppgand max slope. Three points in two R-R peak.
####------------------------------------------------------- Part1: Find Points on Signals ---------------------------------------------------------####


## Find PPG key points, input ppg_signal
ppg_signal = ppg_norm

import numpy as np

ppg_min_index = []
ppg_max_index = []
ppg_height= (0.45 * (np.max(ppg_signal) - np.min(ppg_signal))) + np.min(ppg_signal)
for i in range(1, (len(ppg_signal)-1)):
    if (ppg_signal[i] > ppg_signal[i-1]) and (ppg_signal[i] >= ppg_signal[i+1]) and (ppg_signal[i] > ppg_height) :
        ppg_max_index.append(i)
for i in range(len(ppg_max_index[:-1])):
    ppg_min_index.append(ppg_max_index[i] + np.argmin(ppg_signal[ppg_max_index[i]:ppg_max_index[i+1]]))
print(f'{len(ppg_max_index)} ppg max index: {ppg_max_index}, \n{len(ppg_min_index)} ppg min index: {ppg_min_index}')

ppg_max_slopes_index = []
for i in range(len(ppg_min_index)):  
    ppg_segement = ppg_signal[ppg_min_index[i]:ppg_max_index[i+1]]
    ppg_slope_list = []
    for point in range(len(ppg_segement)-1):
        ppg_slope_list.append(ppg_segement[point+1]-ppg_segement[point])
    #print(i)
    #print(ppg_slope_list)
    if len(ppg_slope_list) > 0:
        ppg_max_slope = max(ppg_slope_list)
        ppg_max_slope_index_in_segement = ppg_slope_list.index(ppg_max_slope)
        ppg_max_slope_index = ppg_min_index[i] + ppg_max_slope_index_in_segement
        ppg_max_slopes_index.append(ppg_max_slope_index)
    else:
        ppg_max_slope_index = ppg_min_index[i]
        ppg_max_slopes_index.append(ppg_max_slope_index)
print(f'{len(ppg_max_slopes_index)} ppg max slope: {ppg_max_slopes_index}')
fig = plt.figure(figsize=(30,6))
plt.scatter(ppg_max_slopes_index, ppg_signal[ppg_max_slopes_index], color='orange', marker='o')
plt.scatter(ppg_max_index, ppg_signal[ppg_max_index], color='c', marker='o')
plt.scatter(ppg_min_index, ppg_signal[ppg_min_index], color='green', marker='o')
plt.plot(ppg_signal)
plt.show()

## Find BP key points, input bp_signal
bp_signal = bp_norm
sbp_height = np.min(bp_signal) + 0.6 * (np.max(bp_signal) - np.min(bp_signal))
sbp_index = []
for i in range(1, (len(bp_signal)-1)):
    if (bp_signal[i] > bp_signal[i-1]) and (bp_signal[i] >= bp_signal[i+1]) and (bp_signal[i] > sbp_height):
        sbp_index.append(i)

dbp_index = []
for index_one, index_two in zip(sbp_index[:-1], sbp_index[1:]):
    dbp_index.append(np.argmin(bp_signal[index_one:index_two]) + index_one)

print(f'{len(sbp_index)} sbp index: {sbp_index}, \n{len(dbp_index)} dbp index: {dbp_index}')
fig = plt.figure(figsize=(30,6))
plt.scatter(sbp_index, bp_signal[sbp_index], color='c', marker='o')
plt.scatter(dbp_index, bp_signal[dbp_index], color='green', marker='o')
plt.plot(bp_signal)
plt.show()


#def get_ecg_points_index(ecg_signal):
ecg_signal = ecg_norm
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



## Align signals from first p-peak
for r_i in range(len(r_peak_index) - 1):
    count_sbp = []
    for sbp_i in sbp_index:
        if r_peak_index[r_i] < sbp_i < r_peak_index[r_i+1]:
            count_sbp.append(sbp_i)
            print(f'first align bp: {count_sbp[0]}')
        elif sbp_i >= r_peak_index[r_i+1]:
            break
    if len(count_sbp) == 1:
        for slope_i in ppg_max_slopes_index:
            if slope_i > r_peak_index[r_i]:
                first_max_ppg = ppg_max_index[ppg_max_slopes_index.index(slope_i) + 1]
                print(f'first align max ppg: {first_max_ppg}')
                break
        ppgbp_distance = count_sbp[0] - first_max_ppg
        break
'''for ppg_start_location in ppg_max_index:
    if ppg_start_location > p_peak_index[0]:
        print(ppg_start_location)
        break
for bp_start_location in sbp_index:
    if bp_start_location > p_peak_index[0]:
        print(bp_start_location)
        break
ppgbp_distance = bp_start_location - ppg_start_location'''
print(ppgbp_distance)
if ppgbp_distance <= 0: # move ppg
    ppg_aligned = ppg_norm[-ppgbp_distance:]
    bp_aligned = bp_norm[:ppgbp_distance]
    bp_ori_aligned = bp_ori[:ppgbp_distance]
    ecg_aligned = ecg_norm[:ppgbp_distance]
    ppg_max_index_aligned = []
    ppg_max_slopes_index_aligned = []
    ppg_min_index_aligned = []
    for i in range(len(ppg_max_slopes_index)):
        if (ppg_min_index[i] - abs(ppgbp_distance)) >= 0:
            ppg_max_slopes_index_aligned.append(ppg_max_slopes_index[i] - abs(ppgbp_distance))
            ppg_max_index_aligned.append(ppg_max_index[i+1] - abs(ppgbp_distance))
            ppg_min_index_aligned.append(ppg_min_index[i] - abs(ppgbp_distance))
    sbp_index_aligned = sbp_index[1:]
    dbp_index_aligned = dbp_index
    p_peak_index_aligned = p_peak_index
    q_peak_index_aligned = q_peak_index
    r_peak_index_aligned = r_peak_index
    s_peak_index_aligned = s_peak_index
else: # move bp and ecg ahead
    ppg_aligned = ppg_norm[:-ppgbp_distance]
    bp_aligned = bp_norm[ppgbp_distance:]
    bp_ori_aligned = bp_ori[ppgbp_distance:]
    ecg_aligned = ecg_norm[ppgbp_distance:]
    ppg_max_index_aligned = ppg_max_index[1:]
    ppg_min_index_aligned = ppg_min_index
    ppg_max_slopes_index_aligned = ppg_max_slopes_index
    sbp_index_aligned = []
    dbp_index_aligned = []
    for i in range(len(dbp_index)):
        if (dbp_index[i] - ppgbp_distance) >= 0:    
            sbp_index_aligned.append(sbp_index[i+1] - ppgbp_distance)
            dbp_index_aligned.append(dbp_index[i] - ppgbp_distance)
    p_peak_index_aligned = []
    q_peak_index_aligned = []
    for i in range(len(p_peak_index)):
        if (p_peak_index[i] - abs(ppgbp_distance)) >= 0:
            p_peak_index_aligned.append(p_peak_index[i] - abs(ppgbp_distance))
            q_peak_index_aligned.append(q_peak_index[i] - abs(ppgbp_distance))
    r_peak_index_aligned = []
    s_peak_index_aligned = []
    for i in range(len(r_peak_index) - 1):
        if (r_peak_index[i] - abs(ppgbp_distance)) >= 0:
            r_peak_index_aligned.append(r_peak_index[i] - abs(ppgbp_distance))
            s_peak_index_aligned.append(s_peak_index[i] - abs(ppgbp_distance))
print(r_peak_index_aligned)
print(s_peak_index_aligned)
print(len(ppg_aligned), len(bp_aligned), len(bp_ori_aligned), len(ecg_aligned))
print(len(ppg_max_index_aligned), len(ppg_min_index_aligned), len(ppg_max_slopes_index_aligned), len(sbp_index_aligned), len(dbp_index_aligned), len(p_peak_index_aligned),len(q_peak_index_aligned), len(r_peak_index_aligned), len(s_peak_index_aligned))
## Show the signal_aligned
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,3),dpi=96)
plt.plot(ppg_aligned, label='PPG_aligned')
plt.plot(bp_aligned, label='BP_aligned')
plt.plot(ecg_aligned, label='ECG_aligned')
plt.scatter(r_peak_index_aligned, ecg_aligned[r_peak_index_aligned], color='c', marker='o')
#plt.scatter(ecg_min_index, ecg_signal[ecg_min_index], color='green', marker='o')
plt.scatter(s_peak_index_aligned, ecg_aligned[s_peak_index_aligned], color='red', marker='o')
plt.scatter(p_peak_index_aligned, ecg_aligned[p_peak_index_aligned], color='pink', marker='o')
plt.scatter(q_peak_index_aligned, ecg_aligned[q_peak_index_aligned], color='purple', marker='o')
xtick_positions = np.arange(0, len(ppg_aligned), fs)
xtick_labels = [f'{int(xtick_position/fs)}s' for xtick_position in xtick_positions]
plt.xticks(xtick_positions, xtick_labels)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Aligned Signal', fontsize=12)
plt.title(f'Signal of Patient {patient}\n', fontsize=12)
plt.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.83, 1.155))
plt.show()


## Cut signals and Save acceptable points
ppg_cut = []
bp_cut = []
bp_ori_cut = []
ecg_cut = []
ppg_max_index_cut = []
ppg_min_index_cut = []
ppg_max_slopes_index_cut = []
sbp_index_cut = []
dbp_index_cut = []
p_peak_index_cut = [0]
q_peak_index_cut = []
r_peak_index_cut = []
s_peak_index_cut = []

ppg_max_slopes_index_loc = 0
# ppg_min_index_loc = 0
sbp_index_loc = 0
# dbp_index_loc = 0
r_peak_index_loc = 0
for i in range(len(p_peak_index_aligned)-1):
    #index_duration = list(range(p_peak_index[i], p_peak_index[i+1]))
    ppg_temlist1 = []
    for ppg_i in ppg_max_slopes_index_aligned[ppg_max_slopes_index_loc:]:
        if p_peak_index_aligned[i] < ppg_i < p_peak_index_aligned[i+1]:
            ppg_temlist1.append(ppg_i)
        elif ppg_i >= p_peak_index_aligned[i+1]:
            ppg_max_slopes_index_loc = ppg_max_slopes_index_aligned.index(ppg_i)
            break
    '''ppg_min_temlist1 = []
    for ppg_i in ppg_min_index_aligned[ppg_min_index_loc:]:
        if p_peak_index_aligned[i] <= ppg_i < p_peak_index_aligned[i+1]:
            ppg_min_temlist1.append(ppg_i)
        elif ppg_i >= p_peak_index_aligned[i+1]:
            ppg_min_index_loc = ppg_min_index_aligned.index(ppg_i)
            break'''
    sbp_temlist1 = []
    for sbp_i in sbp_index_aligned[sbp_index_loc:]:
        if p_peak_index_aligned[i] <= sbp_i < p_peak_index_aligned[i+1]:
            sbp_temlist1.append(sbp_i)
        elif sbp_i >= p_peak_index_aligned[i+1]:
            sbp_index_loc = sbp_index_aligned.index(sbp_i)
            break
    '''dbp_temlist1 = []
    for dbp_i in dbp_index_aligned[dbp_index_loc:]:
        if p_peak_index_aligned[i] <= dbp_i < p_peak_index_aligned[i+1]:
            dbp_temlist1.append(dbp_i)
        elif dbp_i >= p_peak_index_aligned[i+1]:
            dbp_index_loc = dbp_index_aligned.index(dbp_i)
            break'''
    r_peak_temlist1 = []
    for r_peak_i in r_peak_index_aligned[r_peak_index_loc:]:
        if p_peak_index_aligned[i] <= r_peak_i < p_peak_index_aligned[i+1]:
            r_peak_temlist1.append(r_peak_i)
        elif r_peak_i >= p_peak_index_aligned[i+1]:
            r_peak_index_loc = r_peak_index_aligned.index(r_peak_i)
            break
    # if (len(ppg_max_temlist1) == 1) and (len(ppg_min_temlist1) == 1) and (ppg_min_temlist1[0] < ppg_max_temlist1[0]) and (len(sbp_temlist1) == 1) and (len(dbp_temlist1) == 1) and (dbp_temlist1[0] < sbp_temlist1[0]) and (len(r_peak_temlist1) == 1):
    if (len(ppg_temlist1) == 1) and (len(sbp_temlist1) == 1) and (len(r_peak_temlist1) == 1):
        ppg_cut = ppg_cut + list(ppg_aligned[p_peak_index_aligned[i]:p_peak_index_aligned[i+1]])
        bp_cut = bp_cut + list(bp_aligned[p_peak_index_aligned[i]:p_peak_index_aligned[i+1]])
        bp_ori_cut = bp_ori_cut + list(bp_ori_aligned[p_peak_index_aligned[i]:p_peak_index_aligned[i+1]])
        ecg_cut = ecg_cut + list(ecg_aligned[p_peak_index_aligned[i]:p_peak_index_aligned[i+1]])
        ppg_max_index_cut.append(p_peak_index_cut[-1] + (ppg_max_index_aligned[ppg_max_slopes_index_aligned.index(ppg_temlist1[0])] - p_peak_index_aligned[i]))
        ppg_min_index_cut.append(p_peak_index_cut[-1] + (ppg_min_index_aligned[ppg_max_slopes_index_aligned.index(ppg_temlist1[0])] - p_peak_index_aligned[i]))
        ppg_max_slopes_index_cut.append(p_peak_index_cut[-1] + (ppg_temlist1[0] - p_peak_index_aligned[i]))
        sbp_index_cut.append(p_peak_index_cut[-1] + (sbp_temlist1[0] - p_peak_index_aligned[i])) 
        dbp_index_cut.append(p_peak_index_cut[-1] + (dbp_index_aligned[sbp_index_aligned.index(sbp_temlist1[0])] - p_peak_index_aligned[i])) 
        q_peak_index_cut.append(p_peak_index_cut[-1] + (q_peak_index_aligned[i] - p_peak_index_aligned[i]))
        r_peak_index_cut.append(p_peak_index_cut[-1] + (r_peak_temlist1[0] - p_peak_index_aligned[i]))
        s_peak_index_cut.append(p_peak_index_cut[-1] + (s_peak_index_aligned[r_peak_index_aligned.index(r_peak_temlist1[0])] - p_peak_index_aligned[i]))
        p_peak_index_cut.append(len(ecg_cut)) #p_peak_index_aligned[i]

'''ppg_min_index_cut = ppg_min_index_cut[:-1]
ppg_max_slope_index_cut = ppg_max_slope_index_cut[:-1]
dbp_index_cut = dbp_index_cut[:-1]'''

print(f'ppg_cut: {len(ppg_cut)}, bp_cut: {len(bp_cut)}, ecg_cut: {(len(ecg_cut))}')
print(f' ppg_max_index_cut: {len(ppg_max_index_cut)}, {ppg_max_index_cut} \n ppg_min_index_cut: {len(ppg_min_index_cut)}, {ppg_min_index_cut} \n ppg_max_slopes_index_cut: {len(ppg_max_slopes_index_cut)}, {ppg_max_slopes_index_cut} \n sbp_index_cut: {len(sbp_index_cut)}, {sbp_index_cut} \n dbp_index_cut: {len(dbp_index_cut)}, {dbp_index_cut} \n p_peak_index_cut: {len(p_peak_index_cut)}, {p_peak_index_cut} \n q_peak_index_cut: {len(q_peak_index_cut)}, {q_peak_index_cut} \n r_peak_index_cut: {len(r_peak_index_cut)}, {r_peak_index_cut} \n s_peak_index_cut: {len(s_peak_index_cut)}, {s_peak_index_cut}')
## Show the signal_norm
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,3),dpi=96)
plt.plot(ppg_cut, label='PPG_norm')
plt.plot(bp_cut, label='BP_norm')
plt.plot(ecg_cut, label='ECG_norm')
plt.scatter(ppg_max_index_cut, np.array(ppg_cut)[ppg_max_index_cut])
plt.scatter(ppg_min_index_cut, np.array(ppg_cut)[ppg_min_index_cut])
plt.scatter(ppg_max_slopes_index_cut, np.array(ppg_cut)[ppg_max_slopes_index_cut])
plt.scatter(sbp_index_cut, np.array(bp_cut)[sbp_index_cut])
plt.scatter(dbp_index_cut, np.array(bp_cut)[dbp_index_cut])
plt.scatter(p_peak_index_cut[:-1], np.array(ecg_cut)[p_peak_index_cut[:-1]])
plt.scatter(q_peak_index_cut, np.array(ecg_cut)[q_peak_index_cut])
plt.scatter(r_peak_index_cut, np.array(ecg_cut)[r_peak_index_cut])
plt.scatter(s_peak_index_cut, np.array(ecg_cut)[s_peak_index_cut])
xtick_positions = np.arange(0, patient_data.shape[1], fs)
xtick_labels = [f'{int(xtick_position/fs)}s' for xtick_position in xtick_positions]
plt.xticks(xtick_positions, xtick_labels)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Normalized Signal', fontsize=12)
plt.title(f'Signal of Patient {patient}\n', fontsize=12)
plt.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.83, 1.155))
plt.show()


####------------------------------------------------------- Part2: Calculate Features of Signals ---------------------------------------------------------####
## Calculate features of signals
ptt_ppg_slope = []
ptt_ppg_max = []
sbp_norm = []
sbp_ori = []
dbp_norm = []
dbp_ori = []

for i in range(len(ppg_max_slopes_index_cut)):
    ptt_ppg_slope.append((ppg_max_slopes_index_cut[i] - r_peak_index_cut[i]) / fs)
    ptt_ppg_max.append((ppg_max_index_cut[i] - r_peak_index_cut[i]) / fs)
    sbp_norm.append(bp_cut[sbp_index_cut[i]])
    sbp_ori.append(bp_ori_cut[sbp_index_cut[i]])
    dbp_norm.append(bp_cut[dbp_index_cut[i]])
    dbp_ori.append(bp_ori_cut[dbp_index_cut[i]])


