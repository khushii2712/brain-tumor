from flask import Flask,render_template,request
from flask import jsonify
import os
import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model
import requests
from PIL import Image
app=Flask(__name__)
model=load_model(r"D:\Brain-Tumour-Detection-Flask-Web-App-master\model3.h5")


@app.route('/')
def home():
    return render_template(r"home.html")


#brain tumour detection   

@app.route('/',methods=['GET','POST'])
def brain():
    if request.method=='POST':
        f=request.files['file']
        base_path=os.path.dirname(__file__)
        file_path=os.path.join(
            base_path,'uploads'
        )
        f.save(file_path)
        value=getResult(file_path)
        ans=get_classname(value)
        return render_template("home.html",key=ans)
   

#Used for getting the response
@app.route('/upload',methods=['GET','POST'])
def upload():
    f=request.files['file']
    base_path=os.path.dirname(__file__)
    file_path=os.path.join(
        base_path,'uploads'
    )
    f.save(file_path)
    value=getResult(file_path)
    ans=get_classname(value)
    return jsonify({"ans":ans})
    # return "hello"


def get_classname(classno):
    if classno==0:
        return "No Brain Tumor"
    else:
        return "Yes Brain Tumor"

def getResult(img):
    image=cv2.imread(img)
    image=Image.fromarray(image,'RGB')
    image=image.resize((64,64))
    image=np.array(image)
    input_image=np.expand_dims(image,axis=0)
    result=model.predict(input_image)
    return result



















if __name__ == '__main__':
    app.run(debug=True)