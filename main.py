import os
import tkinter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
# we just load data from tf directly, no need of csv files

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# splitting not needed in mnist functions

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# normalization directly

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# flattens into a layer. 1st we just add flat layer
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
# softmax gives the probability of each digit to be the right answer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwriting.model')



model = tf.keras.models.load_model('handwriting.model')
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)
