#khai bao thu vien
import os 
import cv2 
import math 
import pafy
import random 
import numpy as np
import datetime as dt 
import tensorflow as tf 
from moviepy.editor import * 
from collections import deque 
import matplotlib.pyplot as plt 
#%matplotlib inline

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import * 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.utils import plot_model

seed_constant = 23 
np.random.seed(seed_constant) 
random.seed(seed_constant) 
tf.random.set_seed(seed_constant)



#truc quan hoa du lieu bang nhan
# Create a Matplotlib figure 
plt.figure(figsize = (30, 30))

# Get Names of all classes in UCF50 
all_classes_names = os.listdir("UCF50")

# Generate a random sample of images each time the cell runs 
random_range = random.sample(range(len(all_classes_names)), 20)

# Iterating through all the random samples 
for counter, random_index in enumerate(random_range, 1): 
    # Getting Class Name using Random Index 
    selected_class_Name = all_classes_names[random_index]
    
    # Getting a list of all the video files present in a Class Directory 
    video_files_names_list = os.listdir(f"UCF50/{selected_class_Name}")
    
    # Randomly selecting a video file 
    selected_video_file_name = random.choice(video_files_names_list)
    
    # Reading the Video File Using the Video Capture 
    video_reader = cv2.VideoCapture(f"UCF50/{selected_class_Name}/{selected_video_file_name}")
     
     # Reading The First Frame of the Video File 
    _, bgr_frame = video_reader.read() 
    
    # Closing the VideoCapture object and releasing all resources.  
    video_reader.release()
    
    # Converting the BGR Frame to RGB Frame  
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) 
    
    # Adding The Class Name Text on top of the Video Frame. 
    cv2.putText(rgb_frame, selected_class_Name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Assigning the Frame to a specific position of a subplot 
    plt.subplot(5, 4, counter) 
    plt.imshow(rgb_frame) 
    plt.axis('off')
