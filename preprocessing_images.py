'''

Database generation takes place in 3 steps
1. Augmenting initial image dataset using keras image preprocessing to create 6 new images from 1.
2. Extracting new images containing only the faces cropped from augmented images usig haarcascade face classifier.
3. Creating the embedding vector of every extracted face image using the pretrained model and complinig them up in a single pickle file for use in the project.

'''


# Installing dependencies
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from utils import triplet_loss
import pickle
import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('nn4.small2.lrn.h5',custom_objects = {'triplet_loss' : triplet_loss,'tf':tf})
database = {}

datagen = ImageDataGenerator(horizontal_flip=True,
                            vertical_flip=False,
                            rotation_range=20, 
                            fill_mode='nearest',
                            zoom_range=0.3,
                            brightness_range=[0.5,2.0]
                            )


# step 1 :- 

names = os.listdir('pictures')
for name in names:
    img = cv2.imread('pictures/{}.jpg'.format(name))
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    it = datagen.flow(samples, batch_size=1)
    for i in range(1,6):
        image = next(it)[0].astype('uint8')
        cv2.imwrite('augmented_images/{}-{}.jpg'.format(name,i),image)


# step 2 :-

aug_names = os.listdir('augmented_images')
for name in aug_names:
    img = cv2.imread('augmented_images/{}.jpg'.format(name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces)!=0:
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = img[y:y+h, x:x+w]
            if len(roi_color)!=0:
                img = cv2.resize(roi_color, (96,96), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite('cascade_faces/{}.jpg'.format(name),img)


# step 3 :-

def img_to_encoding(image_path, model):
    img = cv2.imread(image_path, 1)
    img = img[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

names_faces = os.listdir('cascade_faces')
for i,name in enumerate(names_faces) :
    filepath = 'cascade_faces/{}.jpg'.format(name)
    database[name] = img_to_encoding(filepath, model)
try:
    geeky_file = open('geekyfile_cascade_processed', 'wb')
    pickle.dump(database, geeky_file)
    geeky_file.close()
except:
    print("Something went wrong")
