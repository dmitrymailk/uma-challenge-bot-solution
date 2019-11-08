import cv2 
import tensorflow as tf
import pandas as pd
import pickle
man_classes = pickle.load(open("man-list.pickle", "rb"))
model = tf.keras.models.load_model("64x3-CNN.model")

def prepare(filepath):
    img_height = 65
    img_width = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (img_height, img_width))
    return new_array.reshape(-1, img_height, img_width, 1)


def main():
    prediction = model.predict([prepare('img/image.jpg')])
    result = man_classes[list(prediction[0]).index(max(prediction[0]))]
    return result




