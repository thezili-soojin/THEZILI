#! /usr/bin/python
# -*- coding: cp949 -*-

from __future__ import print_function
import numpy as np
import os
import sys
import cv2
import random
import io
import picamera
import time
import tensorflow as tf
from scipy import ndimage
from six.moves import cPickle as pickle

# Data Path define
REAL_DATA_PATH = '/home/pi/thezili/mirror-mirror-/talking-mirror/smart-mirror/model/data/real.jpg'
CASCADE_DATA_PATH = '/home/pi/thezili/mirror-mirror-/talking-mirror/smart-mirror/model/'

# Classification Number define
NUM_CLASSES = 3
# Image define
IMAGE_SIZE = 48
IMAGE_CHANNELS = 1
PIXEL_DEPTH = 255.0
# Train define
BATCH_SIZE = 160
PATCH_SIZE = 3
DEPTH = 16
NUM_HIDDEN = 64
NUM_STEPS = 30000

# Data shuffle
'''def dataShuffle(dataset, labels):
    zip_data = list(zip(dataset, labels))
    random.shuffle(zip_data)
    dataset, labels = zip(zip_data)
    ran = random.random()
    random.shuffle(dataset, lambda : ran)
    random.shuffle(labels, lambda : ran)
    return dataset, labels

train_dataset, train_labels = dataShuffle(train_dataset, train_labels)
print('dataShuffle', train_dataset, train_labels)
print('dataShuffle.shape', train_dataset.shape, train_labels.shape)'''

# Model
def model(data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, 
layer3_weights, layer3_biases, layer4_weights, layer4_biases, p_keep_input, p_keep_hidden):
    data = tf.nn.dropout(data, p_keep_input)
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    hidden = tf.nn.dropout(hidden, p_keep_hidden)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    hidden = tf.nn.dropout(hidden, p_keep_hidden)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

# Error Display
def show_usage():
    print('Usage : python Mirror_Mirror.py')
    print('\t Mirror_Mirror.py train \t Trains and saves model with saved dataset')
    print('\t Mirror_Mirror.py poc \t\t Trains and  Launch the proof of concept')
    
# make file list before classicfy 
def facecrop(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        img,
        scaleFactor = 1.3,
        minNeighbors = 5
    )    

    if not len(faces) > 0:
        return None
    print(faces)    
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    face = max_area_face
    img = img[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_CUBIC)
    return img    

# Get the picture(low resolution, so it should be quite fast)
def get_picture():
    # Create a memory stream so photos doesn't need to be saved in a file
    stream = io.BytesIO()

    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.capture(stream, format='jpeg')
        
        # Convert the picture into a numpy array
        buff = np.fromstring(stream.getvalue(), dtype=np.uint8)
        
        # Now creates an OpenCV image
        image = cv2.imdecode(buff, 1)
        
        # Save the result image
        cv2.imwrite(REAL_DATA_PATH, image)

#  Main Funtions
if __name__ == "__main__":
    '''if len(sys.argv) <= 1:
        show_usage()
        exit()
    
    if sys.argv[1] == 'train':
        print('argv[1] : train')
        mTrain = True
    elif sys.argv[1] == 'poc':
        print('argv[1] : poc')
        mTrain = False
    else :
        show_usage()
        exit() '''       

    graph = tf.Graph()

    with graph.as_default():
        # Saver Init
        ckpt_dir = '/home/pi/thezili/mirror-mirror-/talking-mirror/smart-mirror/model/ckpt_dir'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        global_step = tf.Variable(0, name='global_step', trainable=False)   

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES))

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, IMAGE_CHANNELS, DEPTH], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([DEPTH]))
        layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
        layer3_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
        layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]))

        # Training computation.
        logits = model(tf_train_dataset, layer1_weights, layer1_biases, layer2_weights, layer2_biases, 
        layer3_weights, layer3_biases, layer4_weights, layer4_biases, 0.8, 0.5)
       # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        #optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        train_prediction = tf.argmax(logits, 1)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
            # Save picture
            get_picture()
            # Sleep
            time.sleep(1)
            
            if(os.path.exists(REAL_DATA_PATH)):
                CASC_PATH = CASCADE_DATA_PATH + 'haarcascade_frontalface_default.xml'
                cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

                print('Full Filename : ' + REAL_DATA_PATH)
                img = cv2.imread(REAL_DATA_PATH)
                img = facecrop(img)

                if img is None:
                    print('Face Detacting Error')
                    exit()
                else:
                    print(img.shape)

                if len(img.shape) == 2:
                    image = img.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1])
                    image = (image.astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
                    image = np.float32(image)

                    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(session, ckpt.model_checkpoint_path) # restore all variables
                        start = global_step.eval() # get last global_step
                        global_step.assign(start).eval()
                        logits = model(image, layer1_weights, layer1_biases, layer2_weights, 
                        layer2_biases, layer3_weights, layer3_biases, layer4_weights, layer4_biases, 1.0, 1.0)
                        train_prediction = session.run(tf.argmax(logits, 1))
                        print(train_prediction)
                    else:
                        print('saver data load Fail\t')
                else:
                    print('Prediction \t Fail')               
		
		#return 0
