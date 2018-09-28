import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import cv2
from flask import request
from flask import jsonify
from flask import Flask
from mrcnn.visualize import save_image, return_image
from flask import send_file
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO

#from evaluate import ffwd_to_img


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.plate import plate1

#matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "trained_model")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
print(os.path)
PLATE_WEIGHTS_PATH = "/trained_model/mask_rcnn_plate_0010.h5"  # TODO: update this path

#############################################################
#Flask App
#############################################################
app = Flask(__name__)


#############################################################
#CONFIG
#############################################################
config = plate1.PlateConfig()
#PLATE_DIR = os.path.join(ROOT_DIR, "datasets/plate")


#############################################################
#INFERENCE CONFIG
#############################################################
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

#############################################################
#CREATE DEVICE
#############################################################
# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
#DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
#TEST_MODE = "inference"


#############################################################
# Create model in inference mode
#############################################################
#with tf.device(DEVICE):
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print('Chaitvirta MASKRCNN !!!!!!!!!!!!!!')

#############################################################
# LOAD WEIGHTS
#############################################################
# Set path to plate weights file
#weights_path = os.path.join(ROOT_DIR, "/trained_model/mask_rcnn_plate_0002.h5")

weights_path = 'trained_model/mask_rcnn_plate_0010.h5'
# Load weights
print("Loading weights... ", weights_path)
model.load_weights(weights_path, by_name=True)
print('Chaitvirta modeli !!!!!!!!!!!!!!!!!!!!')

@app.route('/hello', methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting': 'Hello, ' + name + '!'
    }
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def take_image():
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request1")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")
    # if not allowed_file(filename):
    #     return BadRequest("Invalid file type")

    # input_filepath = os.path.join('./input_images/', filename)
    # output_filepath = os.path.join('./output_images/', filename)
    # input_file.save(input_filepath)
    #
    # # Get checkpoint filename from la_muse
    # checkpoint = request.form.get("checkpoint", "/checkpoints")
    # ffwd_to_img(input_filepath, output_filepath, checkpoint, '/gpu:0')
    # return send_file(output_filepath, mimetype='image/jpg')
    #image = cv2.imread(input_file)
    #image = input_file
    #print(image)
    image = skimage.io.imread(input_file)
    #image = Image.open(input_file)
    #width, height = image.size
    print('Image shape is: ', image.shape)
    print('111')

    #RESIZE
    #
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image, window, scale, padding, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    rv = detect(image)
    print('Opanaaaaaaaaaaaaa')
    return rv



def detect(image):
    print('Shemovedi')
    results = model.detect([image], verbose=1)
    r = results[0]
    print('sdadasdada')
    save_image(image, 'image_name', r['rois'], r['masks'], r['class_ids'], r['scores'], ['BG', 'Plate'],
               filter_classs_names=None,
               scores_thresh=0.1, save_dir='images888/', mode=0)

    # masked_image = return_image(image, 'kkkk', r['rois'], r['masks'], r['class_ids'], r['scores'], ['BG', 'Plate'],
    #                           filter_classs_names=None,
    #                           scores_thresh=0.1, save_dir=None, mode=0)

    #return send_file(BytesIO(masked_image), attachment_filename='pic',as_attachment=True, mimetype='image/jpeg')
    print('PAAAAAAAAAAAAAAAATH!: ', os.path)
    try:
        return send_file('C:/Users/iliac/Documents/Machine Learning/FlaskTest/Cars/images888/image_name.jpg', attachment_filename='python.jpg')
    except:
        print('e raa')


    #return masked_image
    #print(image.filename)
    #return 'shevinaxe'

if __name__ == "__main__":
    app.run()