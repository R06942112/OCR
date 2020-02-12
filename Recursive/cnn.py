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
        conv_b = var_random('b',[num_filters])

        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 1, 1, 1), padding='SAME')
        after_conv = tf.nn.bias_add(after_conv, conv_b)
        after_bn = batch_norm(after_conv, is_training)

        return tf.nn.relu(after_bn)
    
def ConvReluRecursive(incoming, num_filters, filter_size, name, is_training):
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
        conv_b = var_random('b',[num_filters])
        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 1, 1, 1), padding='SAME')
        after_conv = tf.nn.bias_add(after_conv, conv_b)
        after_conv = batch_norm(after_conv, is_training)
        after_conv = tf.nn.conv2d(after_conv, conv_W, strides=(1, 1, 1, 1), padding='SAME')
        after_conv = tf.nn.bias_add(after_conv, conv_b)
#        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 1, 1, 1), padding='SAME')
#        after_conv = tf.nn.bias_add(after_conv, conv_b)
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




def build_cnn(is_training, batch_size, height, width, channels):
    """
    https://github.com/bgshih/crnn/blob/master/model/crnn_demo/config.lua
    :return:
    """
    inputs = tf.placeholder(shape=(batch_size, height, width, channels),name='input_images',dtype=tf.float32)


    net = ConvRelu(inputs, 64, (3, 3), 'conv_conv1')
    net = ConvReluRecursive(net, 64, (3, 3), 'conv_conv2', is_training)
    net = max_2x2pool(net, 'conv_pool1')

    net = ConvReluBN(net, 128, (3, 3), 'conv_conv3', is_training)
    net = ConvReluRecursive(net, 128, (3, 3), 'conv_conv4', is_training)
    net = max_2x2pool(net, 'conv_pool2')

    net = ConvReluBN(net, 256, (3, 3), 'conv_conv5', is_training)
    net = ConvReluRecursive(net, 256, (3, 3), 'conv_conv6', is_training)
    net = max_2x1pool(net, 'conv_pool3')

    net = ConvReluBN(net, 512, (3, 3), 'conv_conv7', is_training)
    net = ConvReluRecursive(net, 512, (3, 3), 'conv_conv8', is_training)
    net = max_2x1pool(net, 'conv_pool4')
    
    net = ConvReluBN(net, 512, (3, 3), 'conv_conv9', is_training)
    net = max_2x1pool(net, 'conv_pool5')

    net = dropout(net, is_training)

    net = tf.squeeze(net, axis=1)

    return net, inputs

