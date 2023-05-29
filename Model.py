from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import albumentations as albu
from skimage.transform import resize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import rcParams
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB7 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization,Flatten,concatenate
from tensorflow.keras import regularizers
import deepstack
from deepstack.base import KerasMember
from deepstack.ensemble import DirichletEnsemble
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from deepstack.ensemble import StackEnsemble
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from deepstack.base import KerasMember
import scipy

from google.colab import drive
drive.mount('/content/drive')
dataset1 = "/content/drive/MyDrive/Training"

img_height,img_width=96,96
batch_size=16
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset1,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)


dataset2 = "/content/drive/MyDrive/Testing"
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset2,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

train_datagen = ImageDataGenerator(#rotation_range = 180,
                                         width_shift_range = 0.1,          #Specifying the changes to be made on the images for training 
                                         height_shift_range = 0.1,
                                         horizontal_flip = True,     #Image Mirroring
                                         rescale = 1./255,           #Rescaling the images so as to get values between 0-1
                                         #zoom_range = 0.2,
                                         validation_split = 0.2     #Split the images for validation by 20%
                                        )
validation_datagen = ImageDataGenerator(rescale = 1./255,
                                         validation_split = 0.2)        #Creating different versions of images for training


train_generator = train_datagen.flow_from_directory(directory = dataset1,                          #Specifying the changes to be made on the images for validation
                                                    target_size = (96,96),  #Target size for images to be resized
                                                    batch_size = 64,                    #Size for imamges to be grouped
                                                    #color_mode = "grayscale",           #I
                                                    class_mode = "categorical",
                                                    subset = "training"
                                                   )
validation_generator = validation_datagen.flow_from_directory( directory = dataset2,
                                                              target_size = (96,96),        #Creating different versions of images for validation
                                                              batch_size = 64,                          #Creating groups of images for train/val
                                                              #color_mode = "grayscale",                 #Converting BGR to Gray 
                                                              class_mode = "categorical",               #Classification
                                                              subset = "validation"
                                                             )


efnb0 = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(96,96,3), classes = 4)
#resnt = tf.keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = (96,96,3), classes = 4)
vg = tf.keras.applications.VGG16(weights = 'imagenet', include_top=False, input_shape=(96,96,3),classes = 4)




model1 = Sequential()
model1.add(efnb0)
model1.add(GlobalAveragePooling2D())
model1.add(BatchNormalization())
model1.add(Dropout(0.20))
model1.add(Flatten())
model1.add(Dense(4, activation='softmax'))



"""model2=Sequential()
model2.add(resnt)
model2.add(GlobalAveragePooling2D())
model2.add(BatchNormalization())
model2.add(Dropout(0.20))
model2.add(Flatten())
model2.add(Dense(4,activation='softmax'))"""

model3 = Sequential()
model3.add(vg)
model3.add(GlobalAveragePooling2D())
model3.add(BatchNormalization())
model3.add(Dropout(0.20))
model3.add(Flatten())
model3.add(Dense(4,activation='softmax'))


member1 = KerasMember(name='model1',keras_model=model1,train_batches = train_generator, val_batches = validation_generator)
#member2 = KerasMember(name='model2',keras_model=model2,train_batches = train_generator, val_batches = validation_generator)
member3 = KerasMember(name='model3',keras_model=model3,train_batches = train_generator, val_batches = validation_generator)




stack = DirichletEnsemble(N=1000)


estimators = [
    ('rf', RandomForestClassifier(verbose=0, n_estimators=400, max_depth=15, n_jobs=20, min_samples_split=30)),
    ('etr', ExtraTreesClassifier(verbose=0, n_estimators=200, max_depth=10, n_jobs=20, min_samples_split=20))
]

clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

stack.model = clf
stack.add_members([member1, member3])
stack.fit()
stack.describe()

