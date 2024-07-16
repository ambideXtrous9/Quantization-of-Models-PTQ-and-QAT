import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import pandas as pd
import warnings
import torch


interpreter = tf.lite.Interpreter(model_path='QATModel.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare the image
def preprocess_image(image_path, input_shape):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_shape[1], input_shape[2]))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = img / 128 - 1
    print(img.shape)
    return img

def modelpred(image_path):
    # Load and preprocess the image
    input_shape = input_details[0]['shape']
    image = preprocess_image(image_path, input_shape)

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run the inference
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    output = torch.from_numpy(output_data[0])
    confidence, predicted_class = torch.max(output, 0)
    return confidence, predicted_class