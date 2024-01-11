#### dataset: UCI Machine Learning Repository_Cuff-Less Blood Pressure Estimation Data Set
#### data address: https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation
#### Citation Request: M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time, IEEE International Symposium on Circuits and Systems (ISCAS'15), 2015.
####    The data set is in matlab's v7.3 mat file, accordingly it should be opened using new versions of matlab or HDF libraries in other environments.(Please refer to the Web for more information about this format) 
####    This database consist of a cell array of matrices, each cell is one record part. 
####    In each matrix each row corresponds to one signal channel: 
####        1: PPG signal, FS=125Hz; photoplethysmograph from fingertip 
####        2: ABP signal, FS=125Hz; invasive arterial blood pressure (mmHg) 
####        3: ECG signal, FS=125Hz; electrocardiogram from channel II


################################################################################################################################################################################
#### Packages:


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



################################################################################################################################################################################



import sys
sys.path.append('/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/')
from read_data import open_data
from filter_and_clean_data import clean_data
from segment_and_features import features_data
from random_forest import run_random_forest

#### Read all of data.
all_data = open_data('/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_1.mat')

data_ready = pd.DataFrame()
saved_number = []
for patient_number in range(len(all_data)):
    if len(all_data[patient_number]) > 18000 :  # 5min-300beats-18000sample at least
        clean_data(all_data[patient_number], patient_number)
        features_data("/Users/jinyanwei/Desktop/BP_Model/Model_record/cleaned_data", patient_number)
        features_file = f'/Users/jinyanwei/Desktop/BP_Model/Model_record/features_data/Part1_feature{patient_number}.csv'
        if not os.path.exists(features_file):
            continue       
        data_ready = pd.concat([data_ready, pd.read_csv(features_file)])
        saved_number.append(patient_number)

data_ready.to_csv(f'/Users/jinyanwei/Desktop/BP_Model/Model_record/data_ready{len(saved_number)}.csv')

run_random_forest(data_ready)  ## Can't split data to train and test, can't write right result.



################################################################################################################################################################################

