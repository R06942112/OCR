import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import tools
import numpy as np

def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
        tensor: A tensor of any type.
    Returns:
        A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_shape[index])
    return combined_shape

def _weight(shape, trainable=True, name='weights', initializer=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    w = tf.get_variable(
        name=name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable
    )
    return w

def _bias(shape, trainable=True, name='biases', initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0)
    b = tf.get_variable(
        name=name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable
    )
    return b

def _fc(layer_name, inputs, out_nodes):
    """
    Args:
        inputs: 4D, 3D or 2D tensor, if 4D tensor,
        out_nodes: number of output neutral units
    """
    shape = combined_static_and_dynamic_shape(inputs)
    if len(shape) == 4:
        size = shape[1] * shape[2] * shape[3]
    else:  # convert the last dimention to out_nodes size
        size = shape[-1]

    with tf.variable_scope(layer_name):
        w = _weight(shape=[size, out_nodes])
        b = _bias(shape=[out_nodes])
        flat_x = tf.reshape(inputs, [-1, size])
        x = tf.matmul(flat_x, w, name='matmul')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x)
        return x
    
def _conv(layer_name, inputs, out_channels, kernel_size=[3, 3], strides=[1, 1], paddings=[1, 1], trainable=True, reuse=None):
    """convolution layer with relu and batch normalization
    Args:
        layer_name: e.g. conv1, conv2
        x: input_tensor, [b, h, w, c]
        reuse: if reuse==tf.AUTO_REUSE: this convolution layer is parameter shared layer
    Returns:
        4D tensor
    """
    in_channels = combined_static_and_dynamic_shape(inputs)[-1]
    strides = [1, strides[0], strides[1], 1]
    p_h, p_w = paddings[0], paddings[1]
    paddings = [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]]

    with tf.variable_scope(layer_name, reuse=reuse):
        w = _weight(shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], trainable=trainable)
        b = _bias(shape=[out_channels], trainable=trainable)
        x = tf.pad(inputs, paddings=paddings)
        x = tf.nn.conv2d(input=x, filter=w, strides=strides, padding='VALID', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        x = tf.layers.batch_normalization(inputs=x, axis=-1)  # In channel 
        return x
    
def _max_pool(layer_name, inputs, paddings, strides, ksize=[2, 2]):
    ksize = [1, ksize[0], ksize[1], 1]
    strides = [1, strides[0], strides[1], 1]
    p_h, p_w = paddings[0], paddings[1]
    paddings = [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]]

    with tf.variable_scope(layer_name):
        x = tf.pad(inputs, paddings=paddings)
        max_pool_ = tf.nn.max_pool(value=x, ksize=ksize, strides=strides, padding='VALID', name='max_pool')
        return max_pool_
    
def _bilstm(layer_name, inputs, hidden_units):
    with tf.variable_scope(layer_name):
        fw_lstm_cell = tf.contrib.rnn.LSTMCell(hidden_units)
        bw_lstm_cell = tf.contrib.rnn.LSTMCell(hidden_units)
        (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            fw_lstm_cell, bw_lstm_cell, inputs, dtype=tf.float32
        )
        output = tf.concat((output_fw, output_bw), 2)
        #output_state_c = tf.concat((output_state_fw.c, output_state_bw.c), 1)
        #output_state_h = tf.concat((output_state_fw.h, output_state_bw.h), 1)
        output_state = tf.contrib.rnn.LSTMStateTuple(output_state_fw, output_state_bw)
        return output, output_state
    
def base_cnn(x):
    """The basel convolutional neural network (BCNN) module for low-level visual representation
    Args:
        x, 4D tensor [b, w, h, c], w equal 100 and h equal 100 and channel equal 3
    """
    with tf.name_scope('BCNN') as scope:
        x = _conv(layer_name='conv_1', inputs=x, out_channels=64)
        x = _max_pool(layer_name='max_pool_1', inputs=x, strides=[2, 2], paddings=[0, 0])
        x = _conv(layer_name='conv_2', inputs=x, out_channels=128)
        x = _max_pool(layer_name='max_pool_2', inputs=x, strides=[2, 2], paddings=[1, 1])
        x = _conv(layer_name='conv_3', inputs=x, out_channels=256)
        x = _conv(layer_name='conv_4', inputs=x, out_channels=256)
        return x
    

def rot90(tensor,axes=[1,2],name=None):
    '''
    autor:lizh
    tensor: a tensor 4 or more dimensions
    k: integer, Number of times the array is rotated by 90 degrees.
    axes: (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.
    
    -----
    Returns
    -------
    tensor : tf.tensor
             A rotated view of `tensor`.
    See Also: https://www.tensorflow.org/api_docs/python/tf/image/rot90 
    '''
    axes = tuple(axes)
#    if len(axes) != 2:
#        raise ValueError("len(axes) must be 2.")
        
    tenor_shape = (tensor.get_shape().as_list())
    dim = len(tenor_shape)
    
#    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == dim:
#        raise ValueError("Axes must be different.")
#        
#    if (axes[0] >= dim or axes[0] < -dim 
#        or axes[1] >= dim or axes[1] < -dim):
#        
#        raise ValueError("Axes={} out of range for tensor of ndim={}."
#            .format(axes, dim))
    
    axes_list = np.arange(0, dim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],axes_list[axes[0]]) # 替换
    
    img90=tf.transpose(tf.reverse(tensor,axis=[axes[1]]), perm=axes_list, name=name)
    return img90

    
def _arbitrary_orientation_network(inputs):
    """the arbitrary orientation network (AON) for capturing the horizontal, vertical and character placement features
    Args:
        feature_map, 4D tensor [b, w, h, c]
    """
    
    def get_character_placement_cluse(inputs):
        with tf.variable_scope('placement_cluse'):
            x = _conv(layer_name='conv_1', inputs=inputs, out_channels=512)
            x = _max_pool(layer_name='max_pool_1', inputs=x, strides=[2, 2], paddings=[1, 1])
            x = _conv(layer_name='conv_2', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_2', inputs=x, strides=[2, 2], paddings=[1, 1])
            x = tf.reshape(x, shape=[-1, 64, 512])
            x = tf.transpose(x, perm=[0, 2, 1])
            x = _fc('fc_1', inputs=x, out_nodes=23)
            x = tf.reshape(x, shape=[-1, 512, 23])
            x = tf.transpose(x, perm=[0, 2, 1])
            x = _fc('fc_2', inputs=x, out_nodes=4)
            x = tf.reshape(x, shape=[-1, 23, 4])
            x = tf.nn.softmax(x, axis=2, name='softmax')
            return x

    def get_feature_sequence(inputs, reuse=None):
        with tf.variable_scope('shared_stack_conv', reuse=reuse):
            x = _conv(layer_name='conv_1', inputs=inputs, out_channels=512)
            x = _max_pool(layer_name='max_pool_1', inputs=x, strides=[2, 1], paddings=[1, 0])
            x = _conv(layer_name='conv_2', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_2', inputs=x, strides=[2, 1], paddings=[0, 1])
            x = _conv(layer_name='conv_3', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_3', inputs=x, strides=[2, 1], paddings=[1, 0])
            x = _conv(layer_name='conv_4', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_4', inputs=x, strides=[2, 1], paddings=[0, 0])
            x = _conv(layer_name='conv_5', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_5', inputs=x, strides=[2, 1], paddings=[0, 0])
            x = tf.squeeze(x, axis=1, name='squeeze')
            return x

    with tf.name_scope('AON_core') as scope:
        feature_horizontal = get_feature_sequence(inputs=inputs)
        feature_seq_1, _= _bilstm(layer_name='bilstm_1', inputs=feature_horizontal, hidden_units=256)
        feature_seq_1_reverse = tf.reverse(feature_seq_1, axis=[1])

        featute_vertical = get_feature_sequence(inputs=rot90(inputs), reuse=True)
        feature_seq_2, _= _bilstm(layer_name='bilstm_2', inputs=featute_vertical, hidden_units=256)
        feature_seq_2_reverse = tf.reverse(feature_seq_2, axis=[1])

        character_placement_cluse = get_character_placement_cluse(inputs=inputs)
        
        res_dict = {
            'feature_seq_1': feature_seq_1,
            'feature_seq_1_reverse': feature_seq_1_reverse,
            'feature_seq_2': feature_seq_2,
            'feature_seq_2_reverse': feature_seq_2_reverse,
            'character_placement_cluse': character_placement_cluse,
        }
        return res_dict


def _filter_gate(aon_core_output_dict, single_seq=False):
    """the filter gate (FG) for combing four feature sequences with the character sequence.
    """
    feature_seq_1 = aon_core_output_dict['feature_seq_1']
    # DEBUG
    if single_seq:
        return feature_seq_1

    feature_seq_1_reverse = aon_core_output_dict['feature_seq_1_reverse']
    feature_seq_2 = aon_core_output_dict['feature_seq_2']
    feature_seq_2_reverse = aon_core_output_dict['feature_seq_2_reverse']
    character_placement_cluse = aon_core_output_dict['character_placement_cluse']

    with tf.name_scope('FG') as scope:
        A = feature_seq_1 * tf.tile(tf.reshape(character_placement_cluse[:, :, 0], [-1, 23, 1]), [1, 1, 512])
        B = feature_seq_1_reverse * tf.tile(tf.reshape(character_placement_cluse[:, :, 1], [-1, 23, 1]), [1, 1, 512])
        C = feature_seq_2 * tf.tile(tf.reshape(character_placement_cluse[:, :, 2], [-1, 23, 1]), [1, 1, 512])
        D = feature_seq_2_reverse * tf.tile(tf.reshape(character_placement_cluse[:, :, 3], [-1, 23, 1]), [1, 1, 512])
        res = A + B + C + D
        res = tf.tanh(res)
        return res
       

    
def build_decoder(tgt_vocab_size,
                  decoder_inputs,num_units,
                  encoder_outputs,
                  is_training,
                  beam_width,
                  decoder_lengths,
                  #encoder_state,
                  batch_size,
                  encoder_length
                  ):
    
    one_hot = tf.one_hot([x for x in range(tgt_vocab_size)],tgt_vocab_size)
    decoder_emb_inputs = tf.nn.embedding_lookup(one_hot, decoder_inputs)   
    projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)
     
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units*2)   
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
    
    
    if is_training == False:
    
        attention_states = tf.contrib.seq2seq.tile_batch( attention_states, multiplier=beam_width )
        decoder_lengths_new = tf.contrib.seq2seq.tile_batch( decoder_lengths, multiplier=beam_width)
        #encoder_state = tf.contrib.seq2seq.tile_batch( encoder_state, multiplier=beam_width )
        batch_size_new = batch_size * beam_width        
    else:
        batch_size_new = batch_size
        decoder_lengths_new = decoder_lengths
    

    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units*2,attention_states,
                                                               memory_sequence_length=decoder_lengths_new)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
          decoder_cell, attention_mechanism,
          attention_layer_size=num_units*2)
    
    initial_state = decoder_cell.zero_state(batch_size_new, tf.float32)#.clone(cell_state=encoder_state)
    
    if is_training == True:
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, decoder_lengths_new, time_major=True)
    
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state,
            output_layer=projection_layer)
    
        outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
       
        logits = outputs.rnn_output
        sample_ids = outputs.sample_id
        
    else:
        # We should specify maximum_iterations, it can't stop otherwise.
        source_sequence_length = encoder_length
        maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)

        # Define a beam-search decoder
        inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=one_hot,
                start_tokens=tf.fill([batch_size], tools.sos_id),
                end_token=tools.eos_id,
                initial_state=initial_state,
                beam_width=beam_width,
                output_layer=projection_layer)
        
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder, maximum_iterations=maximum_iterations)
        
        # Dynamic decoding
        logits = outputs.beam_search_decoder_output.scores#outputs.beam_search_decoder_output.scores
        sample_ids = outputs.predicted_ids
        
    return logits, sample_ids


        
     
def build_network(is_training,
                  batch_size,
                  height,
                  width,
                  channels,
                  decoder_length,
                  tgt_vocab_size,
                  num_units,
                  beam_width,
                  encoder_length,
                  max_gradient_norm,
                  embedding_size):

    inputs = tf.placeholder(shape=(batch_size, height, width, channels),name='input_images',dtype=tf.float32)
    base_features = base_cnn(inputs)
    aon_core_output_dict = _arbitrary_orientation_network(base_features)
    encoded_sequence = _filter_gate(aon_core_output_dict, False)
    encoded_sequence, _= _bilstm(layer_name='bilstm_3', inputs=encoded_sequence, hidden_units=256)
    encoded_sequence = tf.transpose(encoded_sequence, [1, 0, 2])
    
    decoder_inputs = tf.placeholder(tf.int32, shape=(decoder_length, batch_size), name="decoder_inputs")
    decoder_lengths = tf.placeholder(tf.int32, shape=(batch_size), name="decoer_length")
    target_labels = tf.placeholder(tf.int32,shape=(batch_size,decoder_length), name="target_label")

    
    logits, sample_ids = build_decoder(tgt_vocab_size,
                                       decoder_inputs,
                                       num_units,
                                       encoded_sequence,
                                       is_training,
                                       beam_width,
                                       decoder_lengths,
                                       #encoder_state,
                                       batch_size,
                                       encoder_length
                                       )
    

    
    
    
    
    
   
              
    if is_training == True:    

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_labels, logits=logits)
        loss = tf.reduce_sum(loss)
        

        

        

        # Train
        
        
        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
        

        optimizer = tf.train.AdadeltaOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params)),# global_step=global_step)
        


    else : 

        
        loss = tf.no_op()
        train_op = tf.no_op()

        
    return train_op, loss, sample_ids, logits, inputs, decoder_inputs,decoder_lengths, target_labels

