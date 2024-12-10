import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from PIL import  Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Flatten,Dense
image_path='data/'
dataset=[]
labels=[]

image_no=os.listdir(image_path+'no/')
for i,image_name in enumerate(image_no):
    if image_name.split('.')[1]=='jpg':
        image=cv2.imread(image_path+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64)) #to have images with homogenous dimensions
        labels.append(0)
        dataset.append((np.array(image)))

image_yes=os.listdir(image_path+'yes/')
for i,image_name in enumerate(image_yes):
    if image_name.split('.')[1]=='jpg':
        image=cv2.imread(image_path+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        labels.append(1)
        dataset.append((np.array(image)))

dataset=np.array(dataset)
labels=np.array(labels)
x_train,x_test,y_train,y_test=train_test_split(dataset,labels,test_size=0.2)

x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

# Building Model

model=Sequential()
model.add(Conv2D(32,3,3,input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=9,validation_data=(x_test,y_test),shuffle=False)
loss,acc=model.evaluate(x_test,y_test,verbose=1)
print("accuracy is:",acc)
print("loss is :",loss)
model.save('model3.h5')