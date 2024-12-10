import cv2
from keras.models import load_model
from PIL import Image
import pickle
import os
import numpy as np
model=load_model("D:\Brain-Tumour-Detection-Flask-Web-App-master\model2.h5")

img=cv2.imread('data/no/4 no.jpg')
img=Image.fromarray(img)
img=img.resize((64,64))
img=np.array(img)
# print(img)

img=np.expand_dims(img,axis=0)
result=model.predict_step(img)
print(result)
# print(np.argmax(model.predict(img),axis=1))
