import cv2
import tensorflow as tf
import pandas as pd
CATEGORIES = ["Dog", "Cat"]

men_classes = pd.read_csv(".\\images_labelling.csv")
# print(len(set(men_classes["label"])), set(men_classes["label"]))

DATADIR = "C:\\Users\\dmitry\\Documents\\Data Science\\Competitions\\1__uma-challenge\\1__machine-learning\\images"

IMG_HEIGHT = 65
IMG_WIDTH = 100

def prepare(filepath):
  IMG_HEIGHT = 65
  IMG_WIDTH = 100
  img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  img_array = img_array/255.0
  new_array = cv2.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))
  return new_array.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)


model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('./images/40498.png')])

print(list(prediction[0]).index(max(prediction[0])))
print(men_classes['class_'][men_classes['label'] == list(prediction[0]).index(max(prediction[0]))][:1])

prediction = model.predict([prepare('./images/19903.png')])

print(list(prediction[0]).index(max(prediction[0])))
print(men_classes['class_'][men_classes['label'] == list(prediction[0]).index(max(prediction[0]))][:1])
