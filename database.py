# from fr_utils import img_to_encoding
from triplet_loss import triplet_loss
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np
import os

def img_to_encoding(image_path, model):

    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (96,96))
    img = img[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)

    return embedding




model = load_model('nn4.small2.lrn.h5',custom_objects = {'triplet_loss' : triplet_loss,'tf':tf})

database = {}

dir_names = os.listdir('friends_new')
names =  [dir_name.replace('.JPG','') for dir_name in dir_names]
for i,name in enumerate(names) :
    filepath = 'friends_new/{}.JPG'.format(name)
    database[name] = img_to_encoding(filepath, model)


  
try:
    geeky_file = open('db_geekyfile_friends_new', 'wb')
    pickle.dump(database, geeky_file)
    geeky_file.close()
  
except:
    print("Something went wrong")
