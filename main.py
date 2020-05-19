import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from mnist_keras_load import starter
import numpy as np

x_train, y_train, x_val, y_val, x_test, y_test = starter()

model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size = (3,3), activation='relu', padding = 'same', input_shape = (28,28)))

model.add(layers.MaxPooling2D(pool_size = (2,2)))

model.add(layers.Conv2D(filters=64, kernel_size = (3,3), activation = 'relu', padding = 'same'))

model.add(layers.MaxPooling2D(pool_size = (2,2)))

model.add(layers.Conv2D(filters=64, kernel_size = (3,3), activation = 'relu', padding = 'same'))

model.add(layers.MaxPooling2D(pool_size = (2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation = 'relu'))

model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())

results = model.fit(x_train, y_train, epochs=15, batch_size = 64, validation_data = (x_val, y_val))

test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=128)

print('test loss, test acc:', results)

predictions = model.predict(x_test[:25])

first20_preds = np.argmax(predictions, axis=1)[:25]
first20_true = np.argmax(y_test, axis=1)[:25]

print(first20_preds)
print(first20_true)
