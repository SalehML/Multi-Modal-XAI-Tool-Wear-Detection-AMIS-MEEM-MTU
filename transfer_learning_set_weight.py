# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:43:43 2024

@author: Saleh
"""

import numpy as np
import pandas as pd
import tensorflow as tf

feature_neurons = 15

def set_weights_transfer_learning_LSTM(index, CNN_1, CNN_2, RNN, Latent, classifier, Weights):
    
    if index == 0:
            
        print("CNN_I_Submodel_Transfer_Learning!")
            
        weights = {20: Weights[41],
                   23: Weights[50], 
                       26: [Weights[60][0][:feature_neurons],Weights[60][1][:feature_neurons],Weights[60][2][:feature_neurons],Weights[60][3][:feature_neurons]], 
                       27: [Weights[61][0][:feature_neurons,:],Weights[61][1]]}
            
        weights_keys = list(weights.keys())
            
        for i in range(len(weights_keys)):
                
            CNN_1.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
                
    elif index == 1:
            
        print("CNN_II_Submodel_Transfer_Learning!")
            
        weights = {20: Weights[42], 
                   23: Weights[51], 
                   26: [Weights[60][0][feature_neurons:feature_neurons*2],Weights[60][1][feature_neurons:feature_neurons*2],Weights[60][2][feature_neurons:feature_neurons*2],Weights[60][3][feature_neurons:feature_neurons*2]], 
                   27: [Weights[61][0][feature_neurons:feature_neurons*2,:],Weights[61][1]]}
            
            
        weights_keys = list(weights.keys())
            
        for i in range(len(weights_keys)):
                
            CNN_2.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
                
    elif index == 2:
            
        print("RNN_Submodel_Transfer_Learning!")
            
        weights = {1: Weights[43], 
                       2: Weights[46], 
                       4: Weights[52],
                       7: [Weights[60][0][feature_neurons * 2:feature_neurons * 3],Weights[60][1][feature_neurons * 2:feature_neurons * 3],Weights[60][2][feature_neurons * 2:feature_neurons * 3],Weights[60][3][feature_neurons * 2:feature_neurons * 3]], 
                       8: [Weights[61][0][feature_neurons * 2:feature_neurons * 3,:],Weights[61][1]]}
            
        weights_keys = list(weights.keys())
            
        for i in range(len(weights_keys)):
                
            RNN.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
                
    elif index == 3:
            
        print("Latent_Space_Submodel_Transfer_Learning!")
            
        weights = {41: Weights[41],
                   50: Weights[50], 
                   42: Weights[42], 
                   51: Weights[51],
                   43: Weights[43], 
                   46: Weights[46], 
                   52: Weights[52],
                   60: Weights[60]}
            
        weights_keys = list(weights.keys())
            
        for i in range(len(weights_keys)):
                
            Latent.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
              
    elif index == 4:
            
        print("Classifier_Submodel_Transfer_Learning!")
            
        weights = Weights[-1]
            
        classifier.layers[-1].set_weights(weights)
        
def set_weights_transfer_learning_Transformer(index, CNN_1, CNN_2, RNN, Latent, classifier, Weights):
    
    if index == 0:
            
        print("CNN_I_Submodel_Transfer_Learning!")
            
        weights = {20: Weights[57],
                   23: Weights[66], 
                       26: [Weights[76][0][:feature_neurons],Weights[76][1][:feature_neurons],Weights[76][2][:feature_neurons],Weights[76][3][:feature_neurons]], 
                       27: [Weights[77][0][:feature_neurons,:],Weights[77][1]]}
            
        weights_keys = list(weights.keys())
            
        for i in range(len(weights_keys)):
                
            CNN_1.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
                
    elif index == 1:
            
        print("CNN_II_Submodel_Transfer_Learning!")
            
        weights = {20: Weights[58], 
                   23: Weights[67], 
                   26: [Weights[76][0][feature_neurons:feature_neurons*2],Weights[76][1][feature_neurons:feature_neurons*2],Weights[76][2][feature_neurons:feature_neurons*2],Weights[76][3][feature_neurons:feature_neurons*2]], 
                   27: [Weights[77][0][feature_neurons:feature_neurons*2,:],Weights[77][1]]}
            
            
        weights_keys = list(weights.keys())
            
        for i in range(len(weights_keys)):
                
            CNN_2.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
                
    elif index == 2:
            
        print("RNN_Submodel_Transfer_Learning!")
            
        weights = {1: Weights[11], 
                       3: Weights[17], 
                       5: Weights[23],
                       7: Weights[29],
                       8: Weights[32],
                       10: Weights[38],
                       12: Weights[44],
                       14: Weights[50],
                       16: Weights[56],
                       17: Weights[59],
                       20: Weights[68],
                       23: [Weights[76][0][feature_neurons * 2:feature_neurons * 3],Weights[76][1][feature_neurons * 2:feature_neurons * 3],Weights[76][2][feature_neurons * 2:feature_neurons * 3],Weights[76][3][feature_neurons * 2:feature_neurons * 3]], 
                       24: [Weights[77][0][feature_neurons * 2:feature_neurons * 3,:],Weights[77][1]]}
            
        weights_keys = list(weights.keys())
            
        for i in range(len(weights_keys)):
                
            RNN.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
                
    elif index == 3:
            
        print("Latent_Space_Submodel_Transfer_Learning!")
            
        weights = {11: Weights[11], 
                       17: Weights[17], 
                       23: Weights[23],
                       29: Weights[29],
                       32: Weights[32],
                       38: Weights[38],
                       44: Weights[44],
                       50: Weights[50],
                       56: Weights[56],
                       59: Weights[59],
                       68: Weights[68],
                       76: Weights[76],
                       58: Weights[58], 
                       67: Weights[67], 
                       57: Weights[57],
                       66: Weights[66]}
            
        weights_keys = list(weights.keys())
            
        for i in range(len(weights_keys)):
                
            Latent.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
              
    elif index == 4:
            
        print("Classifier_Submodel_Transfer_Learning!")
            
        weights = Weights[-1]
            
        classifier.layers[-1].set_weights(weights)
           
def set_weights_LSTM(index, CNN_1, CNN_2, RNN, Latent, classifier, Weights):
        
    if index == 0:
            
            weights = {1: Weights[2], 
                       2: Weights[4], 
                       4: Weights[8],
                       5: Weights[10],
                       7: Weights[14],
                       8: Weights[16],
                       10: Weights[20],
                       11: Weights[23],
                       14: Weights[32],
                       17: [Weights[42][0][:feature_neurons],Weights[42][1][:feature_neurons],Weights[42][2][:feature_neurons],Weights[42][3][:feature_neurons]], 
                       18: [Weights[43][0][:feature_neurons],Weights[43][1]]}
            
            weights_keys = list(weights.keys())
            
            for i in range(len(weights_keys)):
                
                CNN_1.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
                
    elif index == 1:
            
            weights = {1: Weights[3], 
                       2: Weights[5], 
                       4: Weights[9],
                       5: Weights[11],
                       7: Weights[15],
                       8: Weights[17],
                       10: Weights[21],
                       11: Weights[24],
                       14: Weights[33],
                       17: [Weights[42][0][feature_neurons:feature_neurons*2],Weights[42][1][feature_neurons:feature_neurons*2],Weights[42][2][feature_neurons:feature_neurons*2],Weights[42][3][feature_neurons:feature_neurons*2]], 
                       18: [Weights[43][0][feature_neurons:feature_neurons*2],Weights[43][1]]}
            
            weights_keys = list(weights.keys())
            
            for i in range(len(weights_keys)):
                
                CNN_2.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
                
    elif index == 2:
            
            weights = {1: Weights[25], 
                       2: Weights[28],
                       4: Weights[34],
                       7: [Weights[42][0][feature_neurons * 2:feature_neurons * 3],Weights[42][1][feature_neurons * 2:feature_neurons * 3],Weights[42][2][feature_neurons * 2:feature_neurons * 3],Weights[42][3][feature_neurons * 2:feature_neurons * 3]], 
                       8: [Weights[43][0][feature_neurons * 2:feature_neurons * 3],Weights[43][1]]}
            
            weights_keys = list(weights.keys())
            
            for i in range(len(weights_keys)):
                
                RNN.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
         
    elif index == 3:
            
            print("Latent_Space_Submodel_Transfer_Learning!")
            
            weights = {2: Weights[2],
                       3: Weights[3],
                       4: Weights[4],
                       5: Weights[5],
                       8: Weights[8],
                       9: Weights[9],
                       10: Weights[10],
                       11: Weights[11],
                       14: Weights[14],
                       15: Weights[15],
                       16: Weights[16],
                       17: Weights[17],
                       20: Weights[20],
                       21: Weights[21],
                       23: Weights[23],
                       24: Weights[24],
                       25: Weights[25],
                       28: Weights[28],
                       32: Weights[32],
                       33: Weights[33],
                       34: Weights[34],
                       42: Weights[42]
                     }
            
            weights_keys = list(weights.keys())
            
            for i in range(len(weights_keys)):
                
                Latent.layers[weights_keys[i]].set_weights(weights[weights_keys[i]])
            
    elif index == 4:
            
            weights = Weights[-1]
            
            classifier.layers[-1].set_weights(weights)