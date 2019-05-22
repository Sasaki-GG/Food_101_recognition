"""server"""
import base64
import io
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, url_for
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image, ImageFilter

import tensorflow as tf
from keras.models import load_model, Model
from keras  import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import cv2

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# def print_image(raw_image_data):
#   """print image with '+' & '-'"""
#   one_sample_image = np.array(raw_image_data).reshape([28, 28])
#   for index_a in range(len(one_sample_image)):
#     for index_b in range(len(one_sample_image[index_a])):
#       if index_b == 27:
#         print('')

#       point = one_sample_image[index_a][index_b]

#       if point > 0.3:
#         print("+", end="")
#       elif point > 0 and point <= 0.3:
#         print("-", end="")
#       else:
#         print(" ", end="")


# ### tf start
# ## init

# # placeholders
# x = tf.placeholder(tf.float32, [None, 784])
# y_ = tf.placeholder(tf.float32, [None, 10])
# x_to_be_tested = tf.placeholder(tf.float32, [None, 784])

# # variables
# W = tf.Variable(tf.zeros([784, 10]), name='Weight') # weight
# b = tf.Variable(tf.zeros([10])) # bias

# # the model
# y = tf.nn.softmax(tf.matmul(x, W) + b)

# init = tf.global_variables_initializer()
# sess = tf.InteractiveSession()
# sess.run(init)


# def train():
#   print('start training ...')
#   cross_entropy = tf.reduce_mean(-tf.reduce_sum(
#       y_ * tf.log(y), reduction_indices=[1]))

#   train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#   for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#   print('end training')


# def test_image(raw_image):
#   raw_image_32 = [np.float(raw_image_p) for raw_image_p in raw_image]
#   np_img = np.array(raw_image_32)
#   result = tf.nn.softmax(tf.matmul(x_to_be_tested, W) + b)
#   final_result = tf.argmax(result, 1)
#   py_res = sess.run(final_result, feed_dict={x_to_be_tested: [np_img]})[0]

#   return py_res

### tf end


app = Flask(__name__)
# train()

@app.route('/')
def index():
  return render_template('index.html', title='recog food')


@app.route('/recog', methods=['POST'])
def upload_img():
  file_name = 'tmp_pic.png'
  img = request.form['img'].split(',')[1].encode('utf-8')
  # print (img)
  # img = img.resize((299, 299))
  # img.save(file_name)
  missing_padding = 4 - len(img) % 4
  if missing_padding:
    img += b'=' * missing_padding
  with open(file_name, 'wb') as f:
    f.write(base64.b64decode(img))
  ans = loop_test(file_name)
  print ('Ans :', ans)
  return  jsonify(ans)
  # img_str = request.form['img'][22:]
  # img_data = base64.b64decode(img_str)
  # img_obj = Image.open(io.BytesIO(img_data))
  # img_obj = img_obj.resize((28, 28))
  # img_pix = img_obj.load()
  # img_arr = []

  # for index_b in range(28):
  #   for index_a in range(28):
  #     pix_l = img_pix[index_a, index_b][3]/255
  #     img_arr.append(pix_l)

  # print_image(img_arr)
  # num = test_image(img_arr)
  # num = num.astype(np.float)

  # return jsonify(ok=1, num=num)



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
    print(tmp)
    top_label_ix = np.argmax(scores)
    max(scores)  # label 95 is Sushi, label 33 is donuts
    confidence = scores[0][top_label_ix]
    name = ['hamburger', 'french_fries', 'grbeet_salad', 'sushi']
    pos, max_sco = 0, 0
    for i, x in enumerate(tmp):
        if x > max_sco:
            max_sco = x
            pos = i
    # for i ,x in enumerate(tmp):
    #     pass
    print('Id:{},\tLabel: {},\tConfidence: {}'.format(
        pos, name[pos], max_sco))
    print('Id:{},\tLabel: {},\tConfidence: {}'.format(
        top_label_ix, id2label[top_label_ix], confidence))
    return name[pos]


def load(fileName):
    global id2label
    with open(fileName, 'r+', encoding='utf-8') as f:
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
    img = Image.merge('RGB', (r, g, b))

    plt.imshow(img)
    plt.show()

    print(type(img))
    # img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img = np.array(img, dtype=np.float32)
    # print(img.dtype.name)
    # img.dtype = 'uint8'
    amin, amax = 0, 255  # 求最大最小值
    img = (img - amin) / (amax - amin)
    print(img)
    print(img[0][0].dtype.name)
    print(type(img))

    plt.imshow(img)
    plt.show()

    img_processed = preprocess_input(img)
    plt.imshow(img_processed)
    plt.show()
    # print(img_processed.shape)
    imgs = np.expand_dims(img_processed, 0)
    imgs = imgs.reshape((1, 299, 299, 3))
    print(imgs.shape)
    orig_scores = sess.run(y, feed_dict={x: imgs, K.learning_phase(): False})

    find_top_pred(orig_scores)

load('labels.txt')
sess = tf.Session()
K.set_session(sess)
gd = sess.graph.as_graph_def()
model = load_model('./model4b.10-0.68.hdf5')

gd = tf.get_default_graph()
# print(len(gd.node), 'Nodes')

model._make_predict_function() 
# %matplotlib inline
x = tf.placeholder(tf.float32, shape=model.get_input_shape_at(0))
y = model(x)

def loop_test(file_name):
      # global x, y
      global gd
      with gd.as_default():
        # K.clear_session()
        # with gd.as_default():
        # tf.sesscl
        # tf.keras.backend.clear_session()
        print ('BUG')
        # file_name = input('Input pic name : '.format(file_name))  # picture location
        file_name = os.getcwd() + '\\' + file_name
        # print('Dir :{}'.format(file_name))
        if file_name == 'Q':
            return
        if os.path.exists(file_name) == False:
            print('{} doesn\'t exists.'.format(file_name))
            return

        img = Image.open(file_name)
        img = img.resize((299, 299))
        # img = plt.imread('donuts.png')
        # img = Image.open('donuts.png')
        plt.imshow(img)
        # plt.show()

        img = img.convert('RGBA')
        r, g, b, alpha = img.split()
        img = Image.merge('RGB', (r, g, b))

        plt.imshow(img)
        # plt.show()

        # print (type(img))
        # img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img = np.array(img, dtype=np.float32)
        # print(img.dtype.name)
        # img.dtype = 'uint8'
        amin, amax = 0, 255  # normalize
        img = (img - amin) / (amax - amin)
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
        imgs = imgs.reshape((1, 299, 299, 3))
        print(imgs.shape)
        # with tf.Session() as sess:
        orig_scores = sess.run(
            y, feed_dict={x: imgs, K.learning_phase(): False})

        label_prediction = find_top_pred(orig_scores)
        print('prediction:', label_prediction)
        # K.clear_session()
        return label_prediction



if __name__ == '__main__':
  app.run(debug=False, use_reloader=False, threaded=False)
