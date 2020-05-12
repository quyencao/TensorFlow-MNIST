import numpy as np
import os
from PIL import Image, ImageFilter
from random import randint
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# Utility packages
class TFUtils:
    def __init__(self):
        return

    # Xavier initialization
    @staticmethod
    def xavier_init(shape, name='', uniform=True):
        num_input = sum(shape[:-1])
        num_output = shape[-1]

        if uniform:
            init_range = tf.sqrt(6.0 / (num_input + num_output))
            init_value = tf.random_uniform_initializer(-init_range, init_range)
        else:
            stddev = tf.sqrt(3.0 / (num_input + num_output))
            init_value = tf.truncated_normal_initializer(stddev=stddev)

        return tf.get_variable(name, shape=shape, initializer=init_value)

    @staticmethod
    def conv2d(X, W, strides=None, padding='SAME'):
        if strides is None:
            strides = [1, 1, 1, 1]

        return tf.nn.conv2d(X, W, strides=strides, padding=padding)

    @staticmethod
    def max_pool(X, ksize=None, strides=None, padding='SAME'):
        if ksize is None:
            ksize = [1, 2, 2, 1]

        if strides is None:
            strides = [1, 2, 2, 1]

        return tf.nn.max_pool(X, ksize=ksize, strides=strides, padding=padding)

    @staticmethod
    def build_cnn_layer(X, W, p_dropout=1., pool=True, reshape=None):
        L = tf.nn.relu(TFUtils.conv2d(X, W))

        if pool is True:
            L = TFUtils.max_pool(L)

        if reshape is not None:
            L = tf.reshape(L, reshape)

        if p_dropout == 1:
            return L
        else:
            return tf.nn.dropout(L, p_dropout)


# MNIST base class
# main purpose is building cnn model
# can add other models
class MNIST:
    model_path = None
    data_path = None

    sess = None
    model = None
    mnist = None

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])

    p_keep_conv = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)

    def __init__(self, model_path=None, data_path=None):
        self.model_path = model_path
        self.data_path = data_path

    def init_session(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def print_status(self, text):
        print('---')
        print(text)

    def build_feed_dict(self, X, Y, p_keep_conv=1., p_keep_hidden=1.):
        return {
            self.X: X,
            self.Y: Y,
            self.p_keep_conv: p_keep_conv,
            self.p_keep_hidden: p_keep_hidden
        }

    # define model
    def build_cnn_model(self, p_keep_conv=1., p_keep_hidden=1.):
        W1 = TFUtils.xavier_init([3, 3, 1, 32], 'W1')
        W2 = TFUtils.xavier_init([3, 3, 32, 64], 'W2')
        W3 = TFUtils.xavier_init([3, 3, 64, 128], 'W3')
        W4 = TFUtils.xavier_init([128 * 4 * 4, 625], 'W4')
        W5 = TFUtils.xavier_init([625, 10], 'W5')

        with tf.name_scope('layer1') as scope:
            # L1 Conv shape=(?, 28, 28, 32)
            #    Pool     ->(?, 14, 14, 32)
            L1 = TFUtils.build_cnn_layer(self.X, W1, p_keep_conv)
        with tf.name_scope('layer2') as scope:
            # L2 Conv shape=(?, 14, 14, 64)
            #    Pool     ->(?, 7, 7, 64)
            L2 = TFUtils.build_cnn_layer(L1, W2, p_keep_conv)
        with tf.name_scope('layer3') as scope:
            # L3 Conv shape=(?, 7, 7, 128)
            #    Pool     ->(?, 4, 4, 128)
            #    Reshape  ->(?, 625)
            reshape = [-1, W4.get_shape().as_list()[0]]
            L3 = TFUtils.build_cnn_layer(L2, W3, p_keep_conv, reshape=reshape)
        with tf.name_scope('layer4') as scope:
            # L4 FC 4x4x128 inputs -> 625 outputs
            L4 = tf.nn.relu(tf.matmul(L3, W4))
            L4 = tf.nn.dropout(L4, p_keep_hidden)

        # Output(labels) FC 625 inputs -> 10 outputs
        self.model = tf.matmul(L4, W5, name='model')

        return self.model

    def save_model(self):
        if self.model_path is not None:
            self.print_status('Saving my model..')

            saver = tf.train.Saver(tf.global_variables())
            saver.save(self.sess, self.model_path)

    def load_model(self):
        self.build_cnn_model()

        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def check_accuracy(self, test_feed_dict=None):
        check_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
        accuracy_rates = self.sess.run(accuracy, feed_dict=test_feed_dict)

        return accuracy_rates

# MNIST Prediction class
# check accuracy of test set
# predict random number from test set
# predict number from image
class MNISTPrediction(MNIST):
    def __init__(self, model_path=None, data_path=None):
        MNIST.__init__(self, model_path, data_path)

        self.init()

    def init(self):
        self.print_status('Loading a model..')

        self.init_session()

        self.load_model()

        if self.data_path is not None:
            self.load_training_data(self.data_path)

    def classify(self, feed_dict):
        number = self.sess.run(tf.argmax(self.model, 1), feed_dict)[0]
        accuracy = self.sess.run(tf.nn.softmax(self.model), feed_dict)[0]

        return number, accuracy[number]

    def accuracy_of_testset(self):
        self.print_status('Calculating accuracy of test set..')

        X = self.mnist.test.images.reshape(-1, 28, 28, 1)
        Y = self.mnist.test.labels
        test_feed_dict = self.build_feed_dict(X, Y)

        accuracy = self.check_accuracy(test_feed_dict)

        self.print_status('CNN accuracy of test set: %f' % accuracy)

    def predict_random(self, show_image=False):
        num = randint(0, self.mnist.test.images.shape[0])
        image = self.mnist.test.images[num]
        label = self.mnist.test.labels[num]

        feed_dict = self.build_feed_dict(image.reshape(-1, 28, 28, 1), [label])

        (number, accuracy) = self.classify(feed_dict)
        label = self.sess.run(tf.argmax(label, 0))

        self.print_status('Predict random item: %d is %d, accuracy: %f' %
                                              (label, number, accuracy))

    def predict(self, filename):
        data = self.load_image(filename)

        number, accuracy = self.classify({self.X: data})

        self.print_status('%d is %s, accuracy: %f' % (number, os.path.basename(filename), accuracy))

    def load_image(self, filename):
        img = Image.open(filename).convert('L')

        # resize to 28x28
        img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        # normalization : 255 RGB -> 0, 1
        data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

        # reshape -> [-1, 28, 28, 1]
        return np.reshape(data, (-1, 28, 28, 1)).tolist()

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = script_dir + '/models/mnist-cnn'

mnist = MNISTPrediction(model_path)
mnist.predict(script_dir + '/imgs/digit-4.png')
mnist.predict(script_dir + '/imgs/digit-2.png')
mnist.predict(script_dir + '/imgs/digit-5.png')