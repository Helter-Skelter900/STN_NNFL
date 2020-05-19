import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from mnist_keras_load import starter

x_train, y_train, x_val, y_val, x_test, y_test = starter()

model = models.Sequential()

model.add(layers.Conv2D)