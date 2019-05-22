#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from keras.models import load_model, Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import cv2

id2label = {}

def preprocess_input(x):
    x_copy = np.copy(x)
    x_copy -= 0.5
    x_copy *= 2.
    return x_copy

def find_top_pred(scores):
    # only return 4 of this
    # hamburger 53
    # french_fries 40 
    # grbeet_salad 5
    # sushi 95
    # print ('bug', scores)
    tmp = [scores[0][5], scores[0][40], scores[0][53], scores[0][95]]
    print (tmp)
    top_label_ix = np.argmax(scores)
    max(scores)  # label 95 is Sushi, label 33 is donuts
    confidence = scores[0][top_label_ix]
    name = ['hamburger' ,'french_fries', 'grbeet_salad', 'sushi']
    pos, max_sco = 0, 0
    for i,x in enumerate(tmp):
        if x > max_sco:
            max_sco = x
            pos = i
    # for i ,x in enumerate(tmp):
    #     pass
    print('Id:{},\tLabel: {},\tConfidence: {}'.format(
        pos, name[pos], max_sco))
    print('Id:{},\tLabel: {},\tConfidence: {}'.format(top_label_ix, id2label[top_label_ix], confidence))
    return name[pos]

def load(fileName):
    global id2label 
    with open(fileName,'r+',encoding='utf-8') as f:
        for num, x in enumerate(f):
            id2label[num] = x.strip()

def test():
    sess = tf.Session()
    K.set_session(sess)
    model = load_model('./model4b.10-0.68.hdf5')

    gd = sess.graph.as_graph_def()
    print(len(gd.node), 'Nodes')

    x = tf.placeholder(tf.float32, shape=model.get_input_shape_at(0))
    y = model(x)

    # img = plt.imread('sushi.png')
    img = Image.open('sushi.png')
    # img = plt.imread('donuts.png')
    # img = Image.open('donuts.png')
    plt.imshow(img)
    plt.show()

    img = img.convert('RGBA')
    r, g, b, alpha = img.split()
    img = Image.merge('RGB',(r,g,b))  

    plt.imshow(img)
    plt.show()

    print (type(img))
    # img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img = np.array(img, dtype=np.float32)
    # print(img.dtype.name)
    # img.dtype = 'uint8'
    amin, amax = 0, 255 # 求最大最小值
    img = (img-amin)/(amax-amin)
    print (img)
    print (img[0][0].dtype.name)
    print (type(img))
    

    plt.imshow(img)
    plt.show()

    img_processed = preprocess_input(img)
    plt.imshow(img_processed)
    plt.show()
    # print(img_processed.shape)
    imgs = np.expand_dims(img_processed, 0)
    imgs = imgs.reshape((1,299,299,3))
    print(imgs.shape)
    orig_scores = sess.run(y, feed_dict={x: imgs, K.learning_phase(): False})

    find_top_pred(orig_scores)

sess = tf.Session()
K.set_session(sess)
model = load_model('./model4b.10-0.68.hdf5')

gd = sess.graph.as_graph_def()
print(len(gd.node), 'Nodes')

x = tf.placeholder(tf.float32, shape=model.get_input_shape_at(0))
y = model(x)

def loop_test():

    # %matplotlib inline
    while True:
        file_name = input('Input pic name : ')#picture location
        file_name = os.getcwd() + '\\' + file_name
        print('Dir :{}'.format(file_name))
        if file_name == 'Q':
            break
        if os.path.exists(file_name)==False:
            print('{} doesn\'t exists.'.format(file_name))
            continue

        img = Image.open(file_name)
        # img = plt.imread('donuts.png')
        # img = Image.open('donuts.png')
        plt.imshow(img)
        # plt.show()

        img = img.convert('RGBA')
        r, g, b, alpha = img.split()
        img = Image.merge('RGB',(r,g,b))  

        plt.imshow(img)
        # plt.show()

        # print (type(img))
        # img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img = np.array(img, dtype=np.float32)
        # print(img.dtype.name)
        # img.dtype = 'uint8'
        amin, amax = 0, 255 # normalize
        img = (img-amin)/(amax-amin)
        # print (img)
        # print (img[0][0].dtype.name)
        # print (type(img))
        

        # plt.imshow(img)
        # plt.show()

        img_processed = preprocess_input(img)
        plt.imshow(img_processed)
        # plt.show()
        # print(img_processed.shape)
        imgs = np.expand_dims(img_processed, 0)
        imgs = imgs.reshape((1,299,299,3))
        print(imgs.shape)
        orig_scores = sess.run(y, feed_dict={x: imgs, K.learning_phase(): False})

        label_prediction = find_top_pred(orig_scores)
        print('prediction:',label_prediction)

if __name__ == '__main__':
    load('labels.txt')
    # test()
    loop_test()
