import os
import cv2
import glob
import numpy as np
import pandas as pd

from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

#set paras
dir = "G:/project/distracted_driver_detection/date/img"
model_image_size = (320, 480)
fine_tune_layer = 172
final_layer = 314
visual_layer = 311
batch_size = 64

#preparation
train_gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    rotation_range=10.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
)
gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
)

train_generator = train_gen.flow_from_directory(os.path.join(dir, 'train'),  model_image_size, shuffle=True, batch_size=batch_size, class_mode="categorical")
#print("subdior to train type {}".format(train_generator.class_indices))
valid_generator = gen.flow_from_directory(os.path.join(dir, 'valid'),  model_image_size, shuffle=True, batch_size=batch_size, class_mode="categorical")
#print("subdior to valid type {}".format(valid_generator.class_indices))

input_tensor = Input((*model_image_size, 3))
x = input_tensor
x = Lambda(inception_v3.preprocess_input)(x)

base_model = InceptionV3(input_tensor=x, weights='imagenet', include_top=False)

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)
model = Model(base_model.input, x)

print("total layer count {}".format(len(base_model.layers)))

for i in range(fine_tune_layer):
    model.layers[i].trainable = False

print("train_generator.samples = {}".format(train_generator.samples))
print("valid_generator.samples = {}".format(valid_generator.samples))
steps_train_sample = train_generator.samples // 64 + 1
steps_valid_sample = valid_generator.samples // 64 + 1

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=8, validation_data=valid_generator, validation_steps=steps_valid_sample)
model.save("models/InceptionV3.h5".format(fine_tune_layer))
print("model saved!")

model.compile(optimizer=RMSprop(lr=1*0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=3, validation_data=valid_generator, validation_steps=steps_valid_sample)
model.save("models/InceptionV3.h5".format(fine_tune_layer))
print("model saved!")