# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

PATH = '/content/drive/My Drive/datasets/xray'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_normal_dir = os.path.join(train_dir, 'normal')
train_pneumonia_dir = os.path.join(train_dir, 'pneumonia')
validation_normal_dir = os.path.join(validation_dir, 'normal')
validation_pneumonia_dir = os.path.join(validation_dir, 'pneumonia')

# Understanding the data

num_normal_tr = len(os.listdir(train_normal_dir))
num_pneumonia_tr = len(os.listdir(train_pneumonia_dir))

num_normal_val = len(os.listdir(validation_normal_dir))
num_pneumonia_val = len(os.listdir(validation_pneumonia_dir))

total_train = num_normal_tr + num_pneumonia_tr
total_val = num_normal_val + num_pneumonia_val

print('normal train:', num_normal_tr)
print('pneumonia train:', num_pneumonia_tr)
print('normal val:', num_normal_val)
print('pneumonia val:', num_pneumonia_val)
print('total train:', total_train)
print('total validation:', total_val)

batch_size = 128
epochs = 10
IMG_HEIGHT = 100
IMG_WIDTH = 100

# Data preparation 
# Reading data from the disk, decode contents of images and convert it into proper grid format as per RGB content
# Convert them into floating point tensors
# Rescale the tensors from values between 0 to 255 to values between 0 and 1, as neural networks prefer small input values
# Fortunately all these tasks can be done with ImageDataGenerator class by tf.keras
# Adding rotation, shifting, flipping and zoom for more accuracy

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           horizontal_flip=True,
                                           zoom_range=0.5)
validation_image_generator = ImageDataGenerator(rescale=1./255)

# After defining generators, the flow_from_directory method loads images from the disk, applies rescaling and resizing the images into the requires dimensions

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=train_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                            class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=validation_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')

# Visualizing training images
# Next function returns a batch from the dataset. The return value of next function is in form of(x_train, y_train) where x_train is training features nad y_tarin is it labels
sample_training_images, _ = next(train_data_gen) # discarding labels to onlu visualize images

# This function will plot the images in the form of a grid

def plotImages(images_arr):
  fig, axes = plt.subplots(1, 5, figsize=(20,20))
  axes = axes.flatten()
  for img, ax in zip( images_arr, axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()

plotImages(sample_training_images[:5])

# Creating the model
# Adding Dropouts to first and last max pool layers. Applying dropouts will randomly set 20% of the neurons to zero during each training epoch. This helps to avoid overfitting
model = Sequential([
                    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                    MaxPooling2D(),
                    Dropout(0.2),
                    Conv2D(32, 3, padding='same', activation='relu'),
                    MaxPooling2D(),
                    Conv2D(64, 3, padding='same', activation='relu'),
                    MaxPooling2D(),
                    Dropout(0.2),
                    Flatten(),
                    Dense(512, activation='relu'),
                    Dense(1)
])

# Compile the model

model.compile(optimizer='Adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model Summary

model.summary()

# Train the Model
# Use fit_generator method of the ImageDataGenerator class to train the metwork

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train,
    epochs = epochs,
    validation_data = val_data_gen,
    validation_steps = total_val
)

model.save('/content/drive/My Drive/projects/pneumonia xray/last_weights.h5')

# Visualizing the training results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

