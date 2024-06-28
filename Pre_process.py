"""
Explainable AI

Project II

Conference Paper

Author: Saleh ValizadehSotubadi

Research Lab: Automation in Smart Manufacturing

MEEM Department at Michigan Technological University
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from scipy import signal
from image_import import img_import

filter_order = 3

cut_off_freq = 0.4

b, a = signal.butter(filter_order, cut_off_freq)

"""
def integrate(time_step, temp):
    
    Temp = 0
    
    for i in range(len(temp)):
        
        Temp += time_step*temp[i]
        
    return Temp
"""
def Import_Data(status, transfer_learning):

    def zero_order_hold(temp):
        
        for i in range(len(temp) - 1):
            
            if (np.abs(temp[i+1] - temp[i]) > 0.75):
                
                temp[i+1] = temp[i]
                
        return temp
    
    def Subtract(time, vol):
        
        fit_data = np.poly1d(np.polyfit(time, vol, 5))
        
        volume = vol - fit_data(time)
        
        return volume
       
    """
    def features(acc_x, acc_y, acc_z, vol_1, vol_2, temp, T):
        
        vol_2 = Subtract(T, vol_2)
        
        temp = zero_order_hold(temp)
        
        acc_x = np.var(acc_x)
        acc_y = np.var(acc_y)
        acc_z = np.var(acc_z)
        vol_1 = np.var(vol_1)
        vol_2 = np.var(vol_2)
        #temp = np.mean(temp)
        temp = integrate(time_step, temp)
        
        return acc_x, acc_y, acc_z, vol_1, vol_2, temp
    """ 
    def filter_data(x, b, a):
        
        filtered = signal.filtfilt(b, a, x)
        
        return filtered
        
    def normalize_time_series(X):
        
        normalized = (X - np.min(X))/(np.max(X) - np.min(X))
        
        return normalized 
    
    def normalize_machine_data(X):
        
        normalized = (X - 700)/(1300 - 700)
        
        return normalized 
    
    def normalize_images(X):
        
        normalized = (X - 0)/(255 - 0)
        
        return normalized
    
    win = 4
    
    Worn = '_WORN'
    
    if status == 'Train':
    
        os.chdir('C:\\Users\\svalizad\\Desktop\\First Paper\\101622')
    
    elif status == 'Test':
        
        os.chdir('C:\\Users\\svalizad\\Desktop\\First Paper\\test\\101622')
    #os.chdir('C:\\Users\\svalizad\\Desktop\\PhD_Valizadeh_Sal-20221014T140915Z-001\\PhD_Valizadeh_Sal\\MATLAB Code\\101622')
    
    #os.chdir('C:\\Users\\svalizad\\Desktop\\PhD_Valizadeh_Sal-20221020T161538Z-001\\PhD_Valizadeh_Sal\\MATLAB Code\\101622')
    
    Path = os.getcwd()
    
    feature = []
    
    features = []
    
    classifier_features = []
    
    classifier_feature = []
    
    labels = []
    
    Cats = []
    
    PATH = os.listdir()
    
    k = 0
    
    for p in PATH:
        
        if '1100' in p:
            speed = (1100 - 700) / (1300 - 700)
        elif '900' in p:
            speed = (900 - 700) / (1300 - 700)
        elif '700' in p:
            speed = 0
        elif '800' in p:
            speed = (800 - 700) / (1300 - 700)
        elif '900' in p:
            speed = (900 - 700) / (1300 - 700)
        elif '1150' in p:
            speed = (1150 - 700) / (1300 - 700)
        elif '1200' in p:
            speed = (1200 - 700) / (1300 - 700)
        elif '1300' in p:
            speed = 1
        elif '1000' in p:
            speed = (1000 - 700) / (1300 - 700)
            
        
        if '002' in p:
            feed = 1
        else:
            feed = 0
        
        PAth = os.path.join(Path, p) 
        
        print(p)
        
        for path in os.listdir(PAth):
            
            k += 1
            
            print(path)
        
            if Worn in path:
                cat = 1
            else:
                cat = 0
                
            Cats.extend([cat] * win)
            
            data = pd.read_csv(os.path.join(PAth, path))
            data_np = np.array(data)
            
            time_np = data_np[:,0] 
            data_np = data_np[:,1:]
            
            volume = data_np[:,4]
            
            index = np.array(np.where(np.abs(volume - np.mean(volume) > 5)))
            
            start, end = index[0][0], index[0][0] + 240
            
            acc_x, acc_y, acc_z, vol_2, vol_1, temp = filter_data(data_np[start:end, 0], b, a), filter_data(data_np[start:end, 1], b, a), filter_data(data_np[start:end, 2], b, a), filter_data(data_np[start:end, 3], b, a), filter_data(data_np[start:end, 4], b, a), data_np[start:end, 5] 
            
            time_np = time_np[start:end]  
            
            vol_2 = Subtract(time_np, vol_2)
            
            temp = zero_order_hold(temp)
            
            acc_x, acc_y, acc_z, vol_2, vol_1, temp = normalize_time_series(acc_x), normalize_time_series(acc_y), normalize_time_series(acc_z), normalize_time_series(vol_2), normalize_time_series(vol_1), normalize_time_series(temp)
            
            tot, chunk = len(acc_x), math.floor(len(acc_x)/win)
            
            for i in range(win):
                
                T = time_np[i*chunk: (i+1)*chunk]
                
                ACC_X = acc_x[i*chunk: (i+1)*chunk]
                ACC_Y = acc_y[i*chunk: (i+1)*chunk]
                ACC_Z = acc_z[i*chunk: (i+1)*chunk]
                VOL_1 = vol_1[i*chunk: (i+1)*chunk]
                VOL_2 = vol_2[i*chunk: (i+1)*chunk]
                TEMP = temp[i*chunk: (i+1)*chunk]
                
                classifier_feature.extend([speed,feed])
                
                feature.extend([ACC_X, ACC_Y, ACC_Z, VOL_1, VOL_2, TEMP])
                
                features.append(np.array(feature))
                
                classifier_features.append(np.array(classifier_feature))
                
                feature = []
                
                classifier_feature = []
                
                labels.append(cat)
    
    Training_data = np.array(features)
    Training_data_machine_rate = np.array(classifier_features)
    
    rows = int(Training_data.shape[0])
    
    #Training_data = np.reshape(Training_data,[rows,6])
    
    stat = status
    
    flank, rake = img_import(stat, transfer_learning)
    
    flank, rake = normalize_images(flank), normalize_images(rake)
    
    if stat == "Train":
    
        shuffle = np.random.permutation(rows)
        
        machine_param_data = []
    
        time_data = []
    
        Flank_img = []
    
        Rake_img = []
        
        label = []
    
        for indx,_ in enumerate(shuffle):
            
            index = shuffle[indx]
            
            time_data.append(Training_data[index])
            
            Flank_img.append(flank[index])
            
            Rake_img.append(rake[index])
            
            machine_param_data.append(Training_data_machine_rate[index])
            
            label.append(labels[index])
        
        if transfer_learning == True:
            flank_image = np.array(Flank_img, dtype=np.float32).reshape(-1, 224, 224, 3)
            rake_image = np.array(Rake_img, dtype=np.float32).reshape(-1, 224, 224, 3)
            
        elif transfer_learning == False:
            flank_image = np.array(Flank_img, dtype=np.float32).reshape(-1, 224, 224, 3)
            rake_image = np.array(Rake_img, dtype=np.float32).reshape(-1, 224, 224, 3)
        
        time_series = np.array(time_data)
        
        machine_params_data_np = np.array(machine_param_data)
        """
        for i in range(machine_params_data_np.shape[0]):
            
            machine_params_data_np[i,0] = normalize_machine_data(machine_params_data_np[i,0])
            
            """
        
        #machine_params_data_np_normalized = normalize_machine_data(machine_params_data_np[:,0])
        
        #machine_params_data_np[:,0] = machine_params_data_np_normalized
        
        #time_series = np.expand_dims(time_series, axis = 1)
        
        Labels = np.array(label, dtype = np.int16)
        
        return machine_params_data_np, time_series, flank_image, rake_image, Labels
    
    elif stat == "Test":
        
        shuffle = np.random.permutation(rows)
    
        time_data = []
        
        machine_param_data = []
    
        Flank_img = []
    
        Rake_img = []
        
        label = []
    
        for indx,_ in enumerate(shuffle):
            
            index = shuffle[indx]
            
            time_data.append(Training_data[index])
            
            Flank_img.append(flank[index])
            
            machine_param_data.append(Training_data_machine_rate[index])
            
            Rake_img.append(rake[index])
            
            label.append(labels[index])
        
        if transfer_learning == True:
            flank_image = np.array(Flank_img, dtype=np.float32).reshape(-1, 224, 224, 3)
            rake_image = np.array(Rake_img, dtype=np.float32).reshape(-1, 224, 224, 3)
            
        elif transfer_learning == False:
            flank_image = np.array(Flank_img, dtype=np.float32).reshape(-1, 224, 224, 3)
            rake_image = np.array(Rake_img, dtype=np.float32).reshape(-1, 224, 224, 3)
        
        time_series = np.array(time_data)
        
        machine_params_data_np = np.array(machine_param_data)
    
        #machine_params_data_np_normalized = normalize_machine_data(machine_params_data_np[:,0])
        
        #machine_params_data_np[:,0] = machine_params_data_np_normalized
        
        #time_series = np.expand_dims(time_series, axis = 1)
        
        Labels = np.array(label, dtype = np.int16)
        
        return machine_params_data_np, time_series, flank_image, rake_image, Labels
        