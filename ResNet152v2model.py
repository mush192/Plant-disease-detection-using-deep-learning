import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from glob import glob


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential


train_path = 'PlantVillage1/train'
test_path = 'PlantVillage1/val'


image_size = [224,224]


resnet152V2  = ResNet152V2(input_shape = image_size + [3], weights = 'imagenet', include_top = False)


for layer in resnet152V2.layers:
    layer.trainable = False


folders = glob('PlantVillage1/train/*')


folders
len(folders)


resnet152V2.output


x = Flatten()(resnet152V2.output)


prediction = Dense(len(folders), activation='softmax')(x)


model = Model(inputs = resnet152V2.input, outputs = prediction)


model.summary()


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),loss='categorical_crossentropy', metrics=["accuracy"])


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   shear_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


train_set = train_datagen.flow_from_directory('PlantVillage1/train',
                                              target_size=(224, 224),
                                              batch_size = 32,
                                              class_mode = 'categorical')
                                          
test_set = test_datagen.flow_from_directory('PlantVillage1/val',
                                            target_size = (224,224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


r = model.fit_generator(train_set, 
                        validation_data= test_set, 
                        epochs=10, 
                        steps_per_epoch= len(train_set), 
                        validation_steps = len(test_set))

