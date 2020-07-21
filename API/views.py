import datetime
import pickle
import json
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from Disease_Detection.settings import BASE_DIR
from PIL import Image
from custom_code import image_converter
import base64

@api_view(['GET'])
def __index__function(request):
    start_time = datetime.datetime.now()
    elapsed_time = datetime.datetime.now() - start_time
    elapsed_time_ms = (elapsed_time.days * 86400000) + (elapsed_time.seconds * 1000) + (elapsed_time.microseconds / 1000)
    return_data = {
        "error" : "0",
        "message" : "Successful",
        "restime" : elapsed_time_ms
    }
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')

def predict_plant_disease(request):
    with open(f"{BASE_DIR}/download.jpg", "rb") as f:
        image_data = base64.b64encode(f.read())
    image_array, err_msg = image_converter.convert_image(image_data)
    model_file = f"{BASE_DIR}/ml_files/cnn_model.pkl"
    saved_classifier_model = pickle.load(open(model_file,'rb'))
    prediction = saved_classifier_model.predict(image_array) 
    label_binarizer = pickle.load(open(f"{BASE_DIR}/ml_files/label_transform.pkl",'rb'))
    return_data = {
        "error" : "0",
        "data" : f"{label_binarizer.inverse_transform(prediction)[0]}"
        }
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')