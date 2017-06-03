
# coding: utf-8

# In[ ]:

import csv
import cv2
import numpy as np 
import sklearn
import os
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli

lines =[]
with open('data/driving_log.csv') as csvfile:
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
        

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)        

def resize(image, size=(50, 50)):
#It transform the image into a squared one.
    return scipy.misc.imresize(image, size)

def crop(image, top_crop =0.35, bottom_crop=0.1): 
    
#Source: https://github.com/upul/Behavioral-Cloning
#You may introduce the percetage in the top and the bottom for cropping the image

    top = int(np.ceil(image.shape[0] * top_crop))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_crop))

    return image[top:bottom, :]

def mirror(image, steering_angle, prob = 0.5):
    #Source: https://github.com/upul/Behavioral-Cloning
    #It randomly flips the images
    
    mirror = bernoulli.rvs(prob)
    if mirror:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle
    


def random_gamma(image):
    
    #Source:  http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    #This way the Network detects the borders way better

    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")


    return cv2.LUT(image, table)


def generator(samples, batch_size=8):
    num_samples=len(samples)
 
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                name1 = 'data/IMG/'+batch_sample[0].split('/')[-1]
                angle = float(batch_sample[3])
                #name2 = 'data/IMG/'+batch_sample[1].split('/')[-1]
                #name3 = 'data/IMG/'+batch_sample[2].split('/')[-1]
                
                center_image = cv2.imread(name1)
                #left_image = cv2.imread(name2)
                #right_image = cv2.imread(name3)
                
                center_image = crop(center_image)
                #left_image = crop(left_image)
                #right_image = crop(right_image)
                
                #center_image, angle = mirror(center_image, angle)
                #left_image = mirror(left_image)
                #right_image = mirror(right_image)
                
                center_image = random_gamma(center_image)
                
                #center_image, angle = random_shear(center_image, angle)
                
                center_image = resize(center_image, (50, 50))
                #left_image = resize(left_image, (50, 50))
                #right_image = resize(right_image, (50, 50))
            
                
                images.append(center_image)
                angles.append(angle)
                
            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield (X_train, y_train)
            
train_generator = generator(train_samples, batch_size=50)
validation_generator = generator(validation_samples, batch_size=50)
ch, row, col = 3, 50, 50

            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, MaxPooling2D
from keras.backend import tf as ktf


model = Sequential()
#model.add(Lambda(lambda x: (x/255)-0.5))
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))


model.add(Conv2D(24, (5, 5), padding="valid"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(36, (5, 5), padding="valid"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(48, (5, 5), padding="valid" ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, (3, 3), padding="valid"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, (3, 3), padding="valid"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=1000, validation_data=validation_generator, nb_val_samples=len(validation_samples), verbose=1, nb_epoch=3)

model.save('mymodel.h5')




