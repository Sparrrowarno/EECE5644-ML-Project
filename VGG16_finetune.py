import os
import cv2
import glob
import numpy as np
import pandas as pd

from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import *

#set paras
dir = "G:/project/distracted_driver_detection/data/imgs"
model_image_size = (224, 224)
fine_tune_layer = 18
final_layer = 21
visual_layer = 18
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


base_model = VGG16(input_tensor=Input((*model_image_size, 3)), weights='imagenet', include_top=False,classes=10)



x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)

#x = Flatten()(base_model.output)
#x = Dense(4096, activation='relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(4096, activation='relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(1000, activation='relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(10, activation='softmax')(x)
model = Model(base_model.input, x)




#base_model.summary()
#model.summary()
print("total layer count {}".format(len(base_model.layers)))

for i in range(fine_tune_layer):
    model.layers[i].trainable = False
    print (i)

print("train_generator.samples = {}".format(train_generator.samples))
print("valid_generator.samples = {}".format(valid_generator.samples))
steps_train_sample = train_generator.samples // 64 + 1
steps_valid_sample = valid_generator.samples // 64 + 1

model.compile(optimizer='adam'(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=14, validation_data=valid_generator, validation_steps=steps_valid_sample)
model.save("models/VGG16.h5".format(fine_tune_layer))
print("model saved!")



################# code from here are data visualization attempt but failed #################
################# code from here are data visualization attempt but failed #################



from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import *

model = load_model("models/vgg16-mymodel.h5")
print("load successed")
SVG(model_to_dot(model).create(prog='dot', format='svg'))
z = zip([x.name for x in model.layers], range(len(model.layers)))
for k, v in z:
    print("{} - {}".format(k,v))

import matplotlib.pyplot as plt
import random
% matplotlib
inline
% config
InlineBackend.figure_format = 'retina'


def show_heatmap_image(model_show, weights_show):
    test_dir = os.path.join(basedir, "test", "test")
    image_files = glob.glob(os.path.join(test_dir, "*"))
    print(len(image_files))

    plt.figure(figsize=(12, 14))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        img = cv2.imread(image_files[2000 * i + 113])
        img = cv2.resize(img, (model_image_size, model_image_size))
        x = img.copy()
        x.astype(np.float32)
        out, predictions = model_show.predict(np.expand_dims(x, axis=0))
        predictions = predictions[0]
        out = out[0]

        max_idx = np.argmax(predictions)
        prediction = predictions[max_idx]

        status = ["safe driving", " texting - right", "phone - right", "texting - left", "phone - left",
                  "operation radio", "drinking", "reaching behind", "hair and makeup", "talking"]

        plt.title('c%d |%s| %.2f%%' % (max_idx, status[max_idx], prediction * 100))

        cam = (prediction - 0.5) * np.matmul(out, weights_show)
        cam = cam[:, :, max_idx]
        cam -= cam.min()
        cam /= cam.max()
        cam -= 0.2
        cam /= 0.8

        cam = cv2.resize(cam, (model_image_size, model_image_size))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam <= 0.2)] = 0

        out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)

        plt.axis('off')
        plt.imshow(out[:, :, ::-1])


print("done")

weights = model.layers[21].get_weights()[0]
layer_output = model.layers[18].output
model2 = Model(model.input, [layer_output, model.output])
print("layer_output {0}".format(layer_output))
print("weights shape {0}".format(weights.shape))
show_heatmap_image(model2, weights)

plt.show()

