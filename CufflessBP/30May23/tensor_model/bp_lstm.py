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
import sys
sys.path.append('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/')
from read_data import open_data
from filter_and_clean_data import clean_data
from segment_and_features import features_data

all_data = open_data('/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_1.mat')

class SelectData:
    def __init__(self, data_array, data_begin=0, data_end=-1):
        self.array = data_array[data_begin:data_end]

    def signal_data(self): # PPG, BP, ECG
        return self.array[:,0], self.array[:,1], self.array[:,2]

    def show_data(self, show_begin=0, show_end=-1):
        fig, ax = plt.subplots(figsize=(30,6))
        ax.plot(self.array[show_begin:show_end])
        plt.show(fig)

def find_first_peak(signal):
    # Calculate the first order difference
    diff_signal = np.diff(signal)

    # Find the index of the first positive-to-negative transition
    for i in range(len(diff_signal) - 1):
        if diff_signal[i] > 0 and diff_signal[i + 1] < 0:
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