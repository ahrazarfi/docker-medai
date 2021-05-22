import asyncio
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image
import json
import config
import tensorflow as tf
import keras

# previous imports
from flask import Flask, jsonify, request, render_template, flash, redirect, url_for,abort
from flask_cors import CORS
import tf.keras.backend as K
from datetime import datetime as dt
import numpy as np
import cv2
from cv2 import resize, INTER_AREA
import uuid
from PIL import Image
import os
import tempfile
from keras.models import load_model
import imageio
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

import os
import urllib.request


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/{disease}")
def get_image(disease: str, file: UploadFile = File(...)):
    model = config.DISEASES[disease]
    spooledImageFile = file.file #changed because compute to be done later on image
    final_predictions = inference(model, spooledImageFile)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    print(f"name: {name}")
    # name = file.file.filename
    cv2.imwrite(name, output)
    return {"name": name}


def inference(model, image):
    data = None
    final_json = []

    if(model=='dia'):
        test_image = Image.open(image)
        test_image = test_image.resize((128,128), Image.ANTIALIAS)
        test_image = np.array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)
        data = test_image
    
    elif(model=='oct'):
        test_image = imageio.imred(image)
        test_image = resize_image_oct(test_image)
        test_image = np.array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        data = test_image
    

    elif(model=='mal'):
        #test_image = image.load_img(name, target_size = (128, 128))
        test_image = Image.open(image)
        test_image = test_image.resize((128,128))        
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        data=test_image  

    else:
      warn = "Feeding blank image won't work. Please enter an input image to continue."
      pred_val =" "
      final_json.append({"pred_val": warn,"mild": " ","mod": " ","norm": " ", "severe":" ",
                         "cnv": " ","dme": " ","drusen": " ","normal": " ","para": " ",
                         "unin": " "})       

    #K.clear_session()                          

    loaded_model = loadModel(model)[0]

    if(model=='dia'):
        preds, pred_val = translate_retinopathy(loaded_model["model"].predict_proba(data))
        final_json.append({"empty": False, "type":model["type"],
                            "mild":preds[0],
                            "mod":preds[1],
                            "norm":preds[2],
                            "severe":preds[3],
                            "pred_val": pred_val})

    elif(model=='oct'):
        preds, pred_val = translate_oct(model["model"].predict(data))
        final_json.append({"empty": False, "type":model["type"],
                            "cnv":preds[0],
                            "dme":preds[1],
                            "drusen":preds[2],
                            "normal":preds[3],
                            "pred_val": pred_val})
                        
    elif(model=='mal'):
        preds, pred_val = translate_malaria(model["model"].predict_proba(data))
        final_json.append({"empty": False, "type":model["type"], 
                            "para":preds[0], 
                            "unin":preds[1],
                            "pred_val": pred_val})

    json_obj = json.loads(final_json)
    print(json.dumps(json_obj, indent= 1))

    return jsonify(final_json)


def resize_image_oct(image):
    resized_image = cv2.resize(image, (128,128))
    if(len(resized_image.shape)!=3):
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB) 
    return resized_image




def loadModel(model):
    model_name = []
    model_path = f"{config.MODEL_PATH}{model}.h5"
    model_name.append({"model": load_model(model_path), "type": model})
    return model_name


    # model_name = f"{config.MODEL_PATH}{model}.h5"
    # model = load_model(model_name)
    # return model



def translate_retinopathy(preds):
    y_proba_Class0 = preds.flatten().tolist()[0] * 100
    y_proba_Class1 = preds.flatten().tolist()[1] * 100
    y_proba_Class2 = preds.flatten().tolist()[2] * 100
    y_proba_Class3 = preds.flatten().tolist()[3] * 100

    mild="Probability of the input image to have Mild Diabetic Retinopathy: {:.2f}%".format(y_proba_Class0)
    mod="Probability of the input image to have Moderate Diabetic Retinopathy: {:.2f}%".format(y_proba_Class1)
    norm="Probability of the input image to be Normal: {:.2f}%".format(y_proba_Class2)
    severe="Probability of the input image to have Severe Diabetic Retinopathy: {:.2f}%".format(y_proba_Class3)

    total = [mild,mod,norm,severe]

    list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
    statements = ["Inference: The image has high evidence for Mild Nonproliferative Diabetic Retinopathy Disease.",
                "Inference: The image has high evidence for Moderate Nonproliferative Diabetic Retinopathy Disease.",
                "Inference: The image has no evidence for Nonproliferative Diabetic Retinopathy Disease.",
                "Inference: The image has high evidence for Severe Nonproliferative Diabetic Retinopathy Disease."]

    index = list_proba.index(max(list_proba))
    prediction = statements[index]

    return total, prediction

def translate_malaria(preds):
    y_proba_Class0 = preds.flatten().tolist()[0] * 100
    y_proba_Class1 = 100.0-y_proba_Class0

    para_prob="Probability of the cell image to be Parasitized: {:.2f}%".format(y_proba_Class1)
    unifected_prob="Probability of the cell image to be Uninfected: {:.2f}%".format(y_proba_Class0)

    total = para_prob + " " + unifected_prob
    total = [para_prob,unifected_prob]

    if (y_proba_Class1 > y_proba_Class0):
        prediction="Inference: The cell image shows strong evidence of Malaria."
        return total,prediction
    else:
        prediction="Inference: The cell image shows no evidence of Malaria."
        return total,prediction

def translate_oct(preds):
    y_proba_Class0 = preds.flatten().tolist()[0] * 100
    y_proba_Class1 = preds.flatten().tolist()[1] * 100
    y_proba_Class2 = preds.flatten().tolist()[2] * 100
    y_proba_Class3 = preds.flatten().tolist()[3] * 100


    cnv="Probability of the input image to have Hyperdynamic Circulation: {:.2f}%".format(y_proba_Class0)
    dme="Probability of the input image to have Normal Ejection Fraction: {:.2f}%".format(y_proba_Class1)
    drusen="Probability of the input image to have Moderate Ejection Fraction: {:.2f}%".format(y_proba_Class2)
    normal="Probability of the input image to have Severe Ejection Fraction: {:.2f}%".format(y_proba_Class3)

    total = [cnv,dme,drusen,normal]

    list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
    statements = ["Inference: The image has high evidence of Hyperdynamic circular.",
                "Inference: The image has high evidence of Normal Ejection Fraction.",
                "Inference: The image has high evidence of Mild Ejection Fraction .",
                "Inference: The image has high evidence of Moderate Ejection Fraction."]


    index = list_proba.index(max(list_proba))
    prediction = statements[index]

    return total, prediction

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)