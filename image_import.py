
"""
Explainable AI

Project II

Conference Paper

Author: Saleh ValizadehSotubadi

Research Lab: Automation in Smart Manufacturing

MEEM Department at Michigan Technological University
"""

import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt

def img_import(status, transfer_learning):

    
    if status == 'Train':
        image_directory = 'C:\\Users\\svalizad\\Desktop\\First Paper\\Pics'
        show = 0
    elif status == 'Test':
        image_directory = 'C:\\Users\\svalizad\\Desktop\\First Paper\\test\\Pics'
        show = 1
    
    os.chdir(image_directory)
    
    rake = []
    
    flank = []
    
    img_size = 224 if transfer_learning == True else 224
    
    PATH = os.listdir()
    
    for path in PATH:
        
        print(path)
        
        image_path = os.path.join(image_directory, path)
        
        k = 1
        
        for img in os.listdir(path):
            
            image = cv2.imread(os.path.join(image_path, img))
            
            if show == 1:
                plt.figure()
                
                plt.imshow(image)
            
            image = cv2.resize(image,(img_size, img_size))
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image2 = cv2.flip(image, 0)
            
            image3 = cv2.flip(image, 1)
            
            image4 = cv2.flip(image2, 1)
            
            #image5, image6, image7 = np.copy(image2), np.copy(image3), np.copy(image4)
            
            #image5[50:75][150:175] = np.array([255, 255, 255])
            
            
            if np.mod(k, 2) == 0:
                
                rake.extend([image, image2, image3, image4])
                
            elif np.mod(k, 2) == 1:
                
                flank.extend([image, image2, image3, image4])
                
                
            k += 1
            
            
    
    image_flank = np.array(flank, dtype = np.float32)
    
    image_rake = np.array(rake, dtype = np.float32)
    
    return image_flank, image_rake
