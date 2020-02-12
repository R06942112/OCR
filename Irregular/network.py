import tensorflow as tf
from seq2seq_model import build_seq2seq
from six.moves import xrange
from scipy.stats import wasserstein_distance

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


def build_cnn(is_training,batch_size, img_size, channels):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    Returns:
    """
    inputs = tf.placeholder(shape=(batch_size, img_size, img_size, channels),name='input_images',dtype=tf.float32)
    with tf.variable_scope('cnn') :
        net = ConvReluBN(inputs, 64, (3, 3), 'conv_conv1', is_training)
        net = ConvReluBN(net, 64, (3, 3), 'conv_conv2', is_training)
        net = max_2x2pool(net, 'conv_pool1')
    
        net = ConvReluBN(net, 128, (3, 3), 'conv_conv3', is_training)
        net = ConvReluBN(net, 128, (3, 3), 'conv_conv4', is_training)
        pool2 = max_2x2pool(net, 'conv_pool2')
    
        net = ConvReluBN(pool2, 256, (3, 3), 'conv_conv5', is_training)
        net = ConvReluBN(net, 256, (3, 3), 'conv_conv6', is_training)
        net = ConvReluBN(net, 256, (3, 3), 'conv_conv7', is_training)
        net = max_2x2pool(net, 'conv_pool3')
        

            
        return net, inputs
    
def build_deconv(is_training,inputs,batch_size):
    with tf.variable_scope('deconv') :
        net = ConvReluBN(inputs, 256, (3, 3), 'deconv_conv1', is_training)
        weights1 = tf.Variable(tf.random_normal([5, 5, 256, 256]))
        deconv = tf.nn.conv2d_transpose(net, weights1,tf.stack([batch_size,32,32,256]),
                                        [1, 2, 2, 1], padding='SAME', name=None )
        
        deconv = ConvReluBN(deconv, 256, (3, 3), 'deconv_conv2', is_training)
        deconv = ConvReluBN(deconv, 256, (3, 3), 'deconv_conv3', is_training)
        deconv = ConvReluBN(deconv, 256, (3, 3), 'deconv_conv4', is_training)
        
        weights2 = tf.Variable(tf.random_normal([5, 5, 128, 256]))
        deconv = tf.nn.conv2d_transpose(deconv, weights2,tf.stack([batch_size,64,64,128]),
                                        [1, 2, 2, 1], padding='SAME', name=None )
        deconv = ConvReluBN(deconv, 128, (3, 3), 'deconv_conv5', is_training)
        deconv = ConvReluBN(deconv, 128, (3, 3), 'deconv_conv6', is_training)
        
        weights3 = tf.Variable(tf.random_normal([5, 5, 64, 128]))
        deconv = tf.nn.conv2d_transpose(deconv, weights3,tf.stack([batch_size,128,128,64]),
                                        [1, 2, 2, 1], padding='SAME', name=None )
        deconv = ConvReluBN(deconv, 64, (3, 3), 'deconv_conv7', is_training)
        deconv = ConvReluBN(deconv, 1, (3, 3), 'deconv_conv8', is_training)
        
        return deconv
    


    
def build_decoder(tgt_vocab_size,
                  decoder_inputs,
                  encoder_outputs,
                  decoder_length,
                  encoder_length,
                  attn_num_hidden,
                  forward_only
                  ):
    attn_num_layers = 1

  
    outputs, attention_weights_history = build_seq2seq(encoder_outputs,
                                                         decoder_inputs,
                                                         tgt_vocab_size,
                                                         encoder_length,
                                                         decoder_length,
                                                         attn_num_layers,
                                                         attn_num_hidden,
                                                         forward_only
                                                         )
    
    
    num_feed = []

    for line in xrange(len(outputs)):
        guess = tf.argmax(outputs[line], axis=1)
        num_feed.append(guess)
        
    return num_feed, outputs, attention_weights_history


    
    

def wasserstein_distance_tf(tensor_a, tensor_b):
    distance = tf.py_func(wasserstein_distance, [tensor_a, tensor_b], tf.double)
    distance = tf.cast(distance,tf.float32)
    return distance
     
def build_network(encoder_outputs,
                  forward_only,
                  batch_size,
                  decoder_length,
                  tgt_vocab_size,
                  attn_num_hidden,
                  encoder_length,
                  max_gradient_norm,
                  f_size,
                  att_loss,
                  img_size,
                  deconv
                  ):
    

    
    decoder_inputs = tf.placeholder(tf.int32, shape=(decoder_length, batch_size), name="decoder_inputs")
    decoder_inputs_u = tf.unstack(decoder_inputs)
    
    target_labels = tf.placeholder(tf.int32,shape=(batch_size,decoder_length), name="target_label")
    
    sample_ids, logits, attention_weights_history = build_decoder(tgt_vocab_size,
                                      decoder_inputs_u,
                                      encoder_outputs,
                                      decoder_length,
                                      encoder_length,
                                      attn_num_hidden,
                                      forward_only
                                      )
    logits = tf.stack(logits,axis=0)
    logits = tf.transpose(logits, [1,0,2])
    attention_weights_history = tf.stack(attention_weights_history,axis=0)
    attention_weights_history = tf.transpose(attention_weights_history, [1,0,2])
    attention_weights_history = tf.reshape(attention_weights_history, shape=(batch_size, decoder_length, f_size, f_size))
    
    
    if att_loss:
        att_label = tf.placeholder(tf.float32, shape=(batch_size,decoder_length,f_size,f_size), name="att_label")
        att_mask = tf.placeholder(tf.float32, shape=(batch_size,decoder_length,f_size,f_size), name="att_mask")
        input_seg = tf.placeholder(shape=(batch_size, img_size, img_size, 1),name='input_mask',dtype=tf.float32)
    else:
        att_label = tf.no_op()
        att_mask = tf.no_op()
        input_seg = tf.no_op()
    

    


    if att_loss:
        att = tf.multiply(attention_weights_history,att_mask)
        att_x = tf.reduce_sum(att,2)
        att_y = tf.reduce_sum(att,3)
        label_x = tf.reduce_sum(att_label,2)
        label_y = tf.reduce_sum(att_label,3)
        
        att_add = tf.add(att_x, att_y)
        att_sub = tf.subtract(att_x, att_y)
        
        label_add = tf.add(label_x, label_y)
        label_sub = tf.subtract(label_x, label_y)
    
    if not forward_only:  # train
    
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        lamda = tf.placeholder(tf.float32, shape=[], name="lamda")

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_labels, logits=logits)
        
        loss = tf.reduce_sum(loss)
        
        if att_loss:
            loss_w = 0
            for i in range(batch_size):
                for j in range(decoder_length):
    
                    loss_w = loss_w + lamda * (wasserstein_distance_tf(label_x[i,j,:],att_x[i,j,:]) + wasserstein_distance_tf(label_y[i,j,:],att_y[i,j,:]) \
                         + 0.5*(wasserstein_distance_tf(label_add[i,j,:],att_add[i,j,:]))+ 0.5*(wasserstein_distance_tf(label_sub[i,j,:],att_sub[i,j,:])))
            
            loss_m = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_seg,logits=deconv)
            loss_m = tf.reduce_mean(loss_m)
            loss = loss + loss_w + loss_m
            
            
        params = tf.trainable_variables()

        
        
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=initial_learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params)),
                        
    else:         
        loss = tf.no_op()
        loss_w = tf.no_op()
        train_op = tf.no_op()
        learning_rate = tf.no_op()
        lamda = tf.no_op()
        
    return train_op, loss, sample_ids, logits, decoder_inputs, target_labels, learning_rate,attention_weights_history,att_label,lamda,att_mask,input_seg
