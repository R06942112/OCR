from __future__ import absolute_import

import numpy as np
import tensorflow as tf


def var_random(name, shape, regularizable=False):
    '''
    Initialize a random variable using xavier initialization.
    Add regularization if regularizable=True
    :param name:
    :param shape:
    :param regularizable:
    :return:
    '''
    v = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if regularizable:
        with tf.name_scope(name + '/Regularizer/'):
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(v))
    return v


def max_2x2pool(incoming, name):
    '''
    max pooling on 2 dims.
    :param incoming:
    :param name:
    :return:
    '''
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def max_2x1pool(incoming, name):
    '''
    max pooling only on image width
    :param incoming:
    :param name:
    :return:
    '''
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 2, 1, 1), strides=(1, 2, 1, 1), padding='SAME')


def ConvRelu(incoming, num_filters, filter_size, name):
    '''
    Add a convolution layer followed by a Relu layer.
    :param incoming:
    :param num_filters:
    :param filter_size:
    :param name:
    :return:
    '''
    num_filters_from = incoming.get_shape().as_list()[3]
    with tf.variable_scope(name):
        conv_W = var_random(
            'W',
            tuple(filter_size) + (num_filters_from, num_filters),
            regularizable=True
        )

        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 1, 1, 1), padding='SAME')

        return tf.nn.relu(after_conv)


def batch_norm(incoming, is_training):
    '''
    batch normalization
    :param incoming:
    :param is_training:
    :return:
    '''
    return tf.contrib.layers.batch_norm(incoming, is_training=is_training, scale=True, decay=0.99)


def ConvReluBN(incoming, num_filters, filter_size, name, is_training):
    '''
    Convolution -> Batch normalization -> Relu
    :param incoming:
    :param num_filters:
    :param filter_size:
    :param name:
    :param is_training:
    :return:
    '''
    num_filters_from = incoming.get_shape().as_list()[3]
    with tf.variable_scope(name):
        conv_W = var_random(
            'W',
            tuple(filter_size) + (num_filters_from, num_filters),
            regularizable=True
        )

        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 1, 1, 1), padding='SAME')

        after_bn = batch_norm(after_conv, is_training)

        return tf.nn.relu(after_bn)


def dropout(incoming, is_training, keep_prob=0.5):
    return tf.contrib.layers.dropout(incoming, keep_prob=keep_prob, is_training=is_training)


def tf_create_attention_map(incoming):
    '''
    flatten hight and width into one dimention of size attn_length
    :param incoming: 3D Tensor [batch_size x cur_h x cur_w x num_channels]
    :return: attention_map: 3D Tensor [batch_size x attn_length x attn_size].
    '''
    shape = incoming.get_shape().as_list()
    return tf.reshape(incoming, (-1, np.prod(shape[1:3]), shape[3]))


class CNN(object):
    """
    Usage for tf tensor output:
    o = CNN(x).tf_output()
    """

    def __init__(self, input_tensor, is_training):
        self._build_network(input_tensor, is_training)

    def _build_network(self, input_tensor, is_training):
        """
        https://github.com/bgshih/crnn/blob/master/model/crnn_demo/config.lua
        :return:
        """
        inputs = tf.placeholder(shape=(batch_size, height, width, channels),name='input_images',dtype=tf.float32)
        with tf.variable_scope('cnn') :
        net = tf.add(input_tensor, (-128.0))
        net = tf.multiply(net, (1/128.0))

        net = ConvReluBN(net, 64, (3, 3), 'conv_conv1', is_training)
        net = ConvReluBN(net, 64, (3, 3), 'conv_conv2', is_training)
        pool1 = max_2x2pool(net, 'conv_pool1')

        net = ConvReluBN(pool1, 128, (3, 3), 'conv_conv3', is_training)
        net = ConvReluBN(net, 128, (3, 3), 'conv_conv4', is_training)
        pool2 = max_2x2pool(net, 'conv_pool2')

        net = ConvReluBN(pool2, 256, (3, 3), 'conv_conv5', is_training)
        net = ConvReluBN(net, 256, (3, 3), 'conv_conv6', is_training)
        net = ConvReluBN(net, 256, (3, 3), 'conv_conv7', is_training)
        pool3 = max_2x2pool(net, 'conv_pool3')
        
        deconv = tf.nn.conv2d_transpose(c, [5, 5, 256, 256],tf.shape(pool2) )



        #net = tf.squeeze(net, axis=1)

        self.model = net

    def tf_output(self):
        # if self.input_tensor is not None:
        return self.model

    # def __call__(self, input_tensor):
    #     return self.model(input_tensor)

    def save(self):
        pass