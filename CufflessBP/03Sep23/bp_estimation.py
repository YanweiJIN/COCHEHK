'''
Created data: 18 Apr 2023
Last modified: 27 Aug 2023
Editor: Yanwei JIN (HKCOCHE)
Supervisors: Prof. Beeluan KHOO, Prof. Rosa CHAN and Prof. Raymond CHAN
Introduction: find signals points, extract features, built models
'''

####------------------------------------------------------- Part1: Read Data ---------------------------------------------------------####

####------------------------------------------------ Part2: Preprocess PPG-BP-ECG signals --------------------------------------------####

####----------------------------------------------------- Part3: Extract Features ----------------------------------------------------####

####----------------------------------------------------- Part4: Build Models ----------------------------------------------------####


## Check if GPU is available and if CUDA is detected
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
