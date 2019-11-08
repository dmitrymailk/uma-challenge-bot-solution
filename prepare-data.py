import numpy as np
import os
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

men_classes = pd.read_csv(".\\images_labelling.csv")
# print(len(set(men_classes["label"])), set(men_classes["label"]))

DATADIR = "C:\\Users\\dmitry\\Documents\\Data Science\\Competitions\\1__uma-challenge\\1__machine-learning\\images"

IMG_HEIGHT = 65
IMG_WIDTH = 100

training_data = []

def create_training_data():
  path = DATADIR 
  for img in tqdm(os.listdir(DATADIR)):  
    try:
        class_num = int(men_classes["label"][men_classes["boxid"] == int(img[:img.find(".")])])
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
        new_array = cv2.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))  # resize to normalize data size
        training_data.append([new_array, class_num]) 
    except Exception as e:  
        pass

create_training_data()

import random
random.shuffle(training_data)
X = []
y = []
from keras.utils import to_categorical

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
y = to_categorical(y, 25)
import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()