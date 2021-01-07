# +
import sys
import os
import joblib
import numpy as np
import tensorflow as tf 
from azureml.core.model import Model
import json

# Called when the service is loaded
def init():
    global model
    print("Executing init() method...")
    model_path = Model.get_model_path('fashion_mnist_model')
    print("got model...")
    model = tf.keras.models.load_model(model_path)
    print("loaded model")

# Called when a request is received
def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    data = np.reshape(data, (1,28,28,1))
    predictions = model.predict(data)
    print("Executed predictions...")
    return json.dumps(predictions.tolist())
