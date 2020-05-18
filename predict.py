# -*- coding: utf-8 -*-
"""predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gnuitzlvR6sZCUW-55a6_K0TVjeXN7dE
"""

import tensorflow as tf
import pickle
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/content/drive/My Drive/projects/pneumonia xray/predtest/'

img_size = 120

testing_data = []
def create_testing_data():
  for img in path:
    try:
      image_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
      new_array = cv2.resize(image_array, (image_size, image_size))
      testing_data.append([new_array])
    except Exception as e:
      pass   

create_testing_data()

print(len(testing_data))