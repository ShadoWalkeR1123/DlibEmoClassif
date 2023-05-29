from PIL import Image
import os
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
import cv2
from skimage.transform import resize
import numpy as np
import scipy
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import cv2
import dlib
import numpy as np 
import sys
import os
from imutils import face_utils
import PIL
from PIL import Image 
import random

dataset1 = "directory_name"
dataset2 = "directory_name"

train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset1,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(96, 96),
  batch_size=64)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset2,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(96, 96),
  batch_size=64)


print(train_ds)
class_names = train_ds.class_names
print(class_names)



predictor_path = "/content/drive/MyDrive/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)



class_names = val_ds.class_names
lst = range(6000)
for image,label in val_ds:
  for i in range(len(image)-1):
    img = image[i].numpy().astype('uint8')
    det = detector(img,1)
    for k,d in enumerate(det):
      shape = predictor(img,d)
      shape = face_utils.shape_to_np(shape)
      
      for (x,y) in shape:
        cv2.circle(img, (x,y),1,(0,255,0),-1)
      img = Image.fromarray(img)
      
      img.save("/content/drive/MyDrive/{}/{}{}.jpeg".format(class_names[label[i]],class_names[label[i]],random.choices(lst)),"JPEG")
