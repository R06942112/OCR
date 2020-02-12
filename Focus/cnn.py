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

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

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
        return tf.nn.max_pool(incoming, ksize=(1, 2, 2, 1), strides=(1, 2, 1, 1), padding='SAME')


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

def ConvRelu_a(incoming, num_filters, filter_size, name):
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

        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 2, 1, 1), padding='VALID')

        return tf.nn.relu(after_conv)
    
def ConvRelu_b(incoming, num_filters, filter_size, name):
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

        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 1, 1, 1), padding='VALID')

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
    


def dropout(incoming, is_training, keep_prob=0.5):
    return tf.contrib.layers.dropout(incoming, keep_prob=keep_prob, is_training=is_training)


def identity_block(X_input,  in_filter, out_filter, stage, block, training):
    """
    Implementation of the identity block as defined in Figure 3
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        #first
        W_conv1 = weight_variable([1, 1, in_filter, out_filter])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #second
        W_conv2 = weight_variable([3, 3, out_filter, out_filter])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        #final step
        add = tf.add(X, X_shortcut)
        add_result = tf.nn.relu(add)

    return add_result


def convolutional_block(X_input,  in_filter,
                        out_filter, stage, block, training, stride=[1,1]):
    """
    Implementation of the convolutional block as defined in Figure 4
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test
    stride -- Integer, specifying the stride to be used
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):

        x_shortcut = X_input
        #first
        W_conv1 = weight_variable([3, 3, in_filter, out_filter])
        X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride[0], stride[1], 1],padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #second
        W_conv2 = weight_variable([3, 3, out_filter, out_filter])
        X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)


        #shortcut path
        W_shortcut = weight_variable([3, 3, in_filter, out_filter])
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride[0], stride[1], 1], padding='SAME')

        #final
        add = tf.add(x_shortcut, X)
        add_result = tf.nn.relu(add)

    return add_result




def build_cnn(is_training, batch_size, height, width, channels):
    """
    https://github.com/bgshih/crnn/blob/master/model/crnn_demo/config.lua
    :return:
    """
    inputs = tf.placeholder(shape=(batch_size, height, width, channels),name='input_images',dtype=tf.float32)


    net = ConvRelu(inputs, 32, (3, 3), 'conv_conv1')
    net = ConvRelu(net, 64, (3, 3), 'conv_conv2')

    net = max_2x2pool(net, 'conv_pool1')
    net = convolutional_block(net,  64, 128, 3, 'conv_res1', is_training)
    net = ConvRelu(net, 128, (3, 3), 'conv_conv3')
    
    net = max_2x2pool(net, 'conv_pool2')
    net = convolutional_block(net,  128, 256, 3, 'conv_res2', is_training)
    net = identity_block(net, 256, 256, 3, 'conv_res3', is_training)
    net = ConvRelu(net, 256, (3, 3), 'conv_conv4')
    paddings=[[0,0],[0,0],[1,0],[0,0]]
    net = tf.pad(net,paddings,"CONSTANT")
    net = max_2x1pool(net, 'conv_pool3')
    net = convolutional_block(net,  256, 512, 3, 'conv_res4', is_training)
    net = identity_block(net, 512, 512, 3, 'conv_res5', is_training)
    net = identity_block(net, 512, 512, 3, 'conv_res6', is_training)
    net = identity_block(net, 512, 512, 3, 'conv_res7', is_training)
    net = identity_block(net, 512, 512, 3, 'conv_res8', is_training)

    net = ConvRelu(net, 512, (3, 3), 'conv_conv5')
    paddings=[[0,0],[0,0],[1,1],[0,0]]
    net = identity_block(net, 512, 512, 3, 'conv_res9', is_training)
    net = identity_block(net, 512, 512, 3, 'conv_res10', is_training)
    net = identity_block(net, 512, 512, 3, 'conv_res11', is_training)
#    
    net = tf.pad(net,paddings,"CONSTANT")
    net = ConvRelu_a(net, 512, (2, 2), 'conv_conv6')
    net = ConvRelu_b(net, 512, (2, 2), 'conv_conv7')




    net = tf.squeeze(net, axis=1)

    return net, inputs




#def build_cnn(x,training,keep_prob):
#    """
#    Implementation of the popular ResNet50 the following architecture:
#    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
#    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
#    Arguments:
#    Returns:
#    """
#
#    with tf.variable_scope('cnn') :
#
#        #stage 1
#        w_conv1 = weight_variable([3, 3, 1, 32])
#        x = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
#        x = tf.layers.batch_normalization(x, axis=3, training=training)
#        x = tf.nn.relu(x)
#
#
#        #stage 2
#        x = convolutional_block(x, 32, 32, 2, 'a', training)
#        x = identity_block(x,  32, 32, 2, 'b', training)
#        x = identity_block(x, 32, 32, 2, 'c', training)
#        
#        x = convolutional_block(x, 32, 64, 3, 'a', training)
#        x = identity_block(x,  64, 64, 3, 'b', training)
#        x = identity_block(x, 64, 64, 3, 'c', training)
#        
#        x = convolutional_block(x, 64, 128, 4, 'a', training, stride=[2,1])
#        x = identity_block(x,  128, 128, 4, 'b', training)
#        x = identity_block(x, 128, 128, 4, 'c', training)
#        
#        x = convolutional_block(x, 128, 256, 5, 'a', training, stride=[2,1])
#        x = identity_block(x,  256, 256, 5, 'b', training)
#        x = identity_block(x, 256, 256, 5, 'c', training)
#        x2 = x
#        
#        x = convolutional_block(x, 256, 512, 6, 'a', training, stride=[2,1])
#        x = identity_block(x,  512, 512, 6, 'b', training)
#        x = identity_block(x, 512, 512, 6, 'c', training)
#        x = tf.squeeze(x, axis=1)
#        
#        x2 = convolutional_block(x2, 256, 512, 7, 'a', training, stride=[2,1])
#        x2 = identity_block(x2,  512, 512, 7, 'b', training)
#        x2 = identity_block(x2, 512, 512, 7, 'c', training)
#        x2 = tf.layers.flatten(x2)
#        x2 = tf.layers.dense(x2, units=6400, activation=tf.nn.relu)
#        x2 = tf.nn.dropout(x2, keep_prob)
#        x2 = tf.layers.dense(x2, units=50, activation=tf.nn.relu)
#        x2 = tf.nn.dropout(x2, keep_prob)
#        x2 = tf.layers.dense(x2, units=2, activation=tf.nn.softmax)
#        
#        return x, x2

