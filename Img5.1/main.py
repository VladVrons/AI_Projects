import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

train_path = "Test/"
test_path = "Training/"

img = load_img("Training/Lemon/0_100.jpg")
plt.imshow(img)
plt.axis("on")
plt.show()

img = img_to_array(img)
img.shape

model = Sequential()
model.add(Conv2D(128, 3, activation="relu", input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation="relu"))
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(5000, activation="relu"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(131, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.3,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   zoom_range=0.3
                                   )
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(100, 100),
                                                    batch_size=32,
                                                    color_mode="rgb",
                                                    class_mode="categorical")
test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=(100, 100),
                                                  batch_size=32,
                                                  color_mode="rgb",
                                                  class_mode="categorical")

hist = model.fit_generator(generator=train_generator,
                           steps_per_epoch=50,
                           epochs=50,
                           validation_data=test_generator,
                           validation_steps=50)

#from keras.models import load_model

#model.save("Fruitmodel.h5")
