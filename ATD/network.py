import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import tools


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def identity_block(X_input,  in_filter, out_filter, stage, block, training):

    with tf.variable_scope('res' + str(stage) + block):
        X_shortcut = X_input

        W_conv1 = weight_variable([1, 1, in_filter, out_filter])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        W_conv2 = weight_variable([3, 3, out_filter, out_filter])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        add = tf.add(X, X_shortcut)
        add_result = tf.nn.relu(add)

    return add_result


def convolutional_block(X_input,  in_filter,
                        out_filter, stage, block, training, stride=[2,2]):
    
    with tf.variable_scope('res' + str(stage) + block):

        x_shortcut = X_input

        W_conv1 = weight_variable([3, 3, in_filter, out_filter])
        X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride[0], stride[1], 1],padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        W_conv2 = weight_variable([3, 3, out_filter, out_filter])
        X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        W_shortcut = weight_variable([3, 3, in_filter, out_filter])
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride[0], stride[1], 1], padding='SAME')


        add = tf.add(x_shortcut, X)
        add_result = tf.nn.relu(add)

    return add_result


def build_cnn(training,batch_size, height, width, channels):

    inputs = tf.placeholder(shape=(batch_size, height, width, channels),name='input_images',dtype=tf.float32)
    with tf.variable_scope('cnn') :

        w_conv1 = weight_variable([3, 3, 1, 32])
        x = tf.nn.conv2d(inputs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)

        x = convolutional_block(x, 32, 32, 2, 'a', training)
        x = identity_block(x,  32, 32, 2, 'b', training)
        x = identity_block(x, 32, 32, 2, 'c', training)
        
        x = convolutional_block(x, 32, 64, 3, 'a', training)
        x = identity_block(x,  64, 64, 3, 'b', training)
        x = identity_block(x, 64, 64, 3, 'c', training)
        
        x = convolutional_block(x, 64, 128, 4, 'a', training, stride=[2,1])
        x = identity_block(x,  128, 128, 4, 'b', training)
        x = identity_block(x, 128, 128, 4, 'c', training)
        
        x = convolutional_block(x, 128, 256, 5, 'a', training, stride=[2,1])
        x = identity_block(x,  256, 256, 5, 'b', training)
        x = identity_block(x, 256, 256, 5, 'c', training)
        x2 = x
        
        x = convolutional_block(x, 256, 512, 6, 'a', training, stride=[2,1])
        x = identity_block(x,  512, 512, 6, 'b', training)
        x = identity_block(x, 512, 512, 6, 'c', training)
        x = tf.squeeze(x, axis=1)
        
        return x, x2, inputs

def build_classifier(training,batch_size,c_learning_rate):
    inputs_f = tf.placeholder(shape=(batch_size,2,25,256),name='input_l',dtype=tf.float32)
    inputs_l =  tf.placeholder(shape=(batch_size,2),dtype=tf.float32)
    with tf.variable_scope('classifier') :
      
        x2 = convolutional_block(inputs_f, 256, 512, 7, 'a', training, stride=[2,1])
        x2 = identity_block(x2,  512, 512, 7, 'b', training)
        x2 = identity_block(x2, 512, 512, 7, 'c', training)
        x2 = tf.layers.flatten(x2)
        x2 = tf.layers.dense(x2, units=6400, activation=tf.nn.relu)
        x2 = tf.layers.dense(x2, units=50, activation=tf.nn.relu)
        x2 = tf.layers.dense(x2, units=2, activation=tf.nn.softmax)
    
        if training == True:    

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=inputs_l,logits=x2)
            loss = tf.reduce_sum(loss)

            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
            

        
            optimizer = tf.train.GradientDescentOptimizer(c_learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(
                        zip(clipped_gradients, params))

        else : 
    
            
            loss = tf.no_op()
            train_op = tf.no_op()
    
        return x2, inputs_f,inputs_l, loss, train_op
    

       
def build_encoder(encoder_inputs,encoder_length,num_units,keep_prob):
    
    with tf.variable_scope('encoder'):
        encoder_inputs = tf.transpose(encoder_inputs, [1, 0, 2])
        
    
        cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units,reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units,reuse=tf.get_variable_scope().reuse)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
        encoder_outputs1 , encoder_state = tf.nn.bidirectional_dynamic_rnn( cell_fw , cell_bw , encoder_inputs , time_major=True,dtype=tf.float32,scope='BLSTM_1')
        encoder_outputs1 = tf.concat(encoder_outputs1, 2)
        
        cell_fw2 = tf.contrib.rnn.BasicLSTMCell(num_units,reuse=tf.get_variable_scope().reuse)
        cell_bw2 = tf.contrib.rnn.BasicLSTMCell(num_units,reuse=tf.get_variable_scope().reuse)
        cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell_fw2, output_keep_prob=keep_prob)
        cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell_bw2, output_keep_prob=keep_prob)
        encoder_outputs , encoder_state = tf.nn.bidirectional_dynamic_rnn( cell_fw2 , cell_bw2 , encoder_outputs1 , time_major=True,dtype=tf.float32,scope='BLSTM_2')
        encoder_outputs = tf.concat(encoder_outputs, 2)
        

 
        c = tf.concat([encoder_state[0][0],encoder_state[1][0]],axis=1)
        h = tf.concat([encoder_state[0][1],encoder_state[1][1]],axis=1)

        encoder_state = tf.contrib.rnn.LSTMStateTuple(c,h)
        return encoder_outputs, encoder_state
    
def build_decoder(tgt_vocab_size,
                  decoder_inputs,num_units,
                  encoder_outputs,
                  is_training,
                  beam_width,
                  decoder_lengths,
                  encoder_state,
                  batch_size,
                  encoder_length,
                  sos
                  ):
    
    one_hot = tf.one_hot([x for x in range(tgt_vocab_size)],tgt_vocab_size)
    decoder_emb_inputs = tf.nn.embedding_lookup(one_hot, decoder_inputs)   
    projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)
     
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units*2)   
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
    
    
    if is_training == False:
    
        attention_states = tf.contrib.seq2seq.tile_batch( attention_states, multiplier=beam_width )
        decoder_lengths_new = tf.contrib.seq2seq.tile_batch( decoder_lengths, multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch( encoder_state, multiplier=beam_width )
        batch_size_new = batch_size * beam_width        
    else:
        batch_size_new = batch_size
        decoder_lengths_new = decoder_lengths
    

    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units*2,attention_states,
                                                               memory_sequence_length=decoder_lengths_new)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
          decoder_cell, attention_mechanism,
          attention_layer_size=num_units*2)
    
    initial_state = decoder_cell.zero_state(batch_size_new, tf.float32).clone(cell_state=encoder_state)
    
    if is_training == True:
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, decoder_lengths_new, time_major=True)
    
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state,
            output_layer=projection_layer)
    
        outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
       
        logits = outputs.rnn_output
        sample_ids = outputs.sample_id
        
    else:

        source_sequence_length = encoder_length
        maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)


        inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=one_hot,
                start_tokens=sos,
                end_token=tools.eos_id,
                initial_state=initial_state,
                beam_width=beam_width,
                output_layer=projection_layer)
        
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder, maximum_iterations=maximum_iterations)
        

        logits = outputs.beam_search_decoder_output.scores
        sample_ids = outputs.predicted_ids
        
    return logits, sample_ids


        
     
def build_network(encoder_inputs,
                  is_training,
                  batch_size,
                  decoder_length,
                  tgt_vocab_size,
                  num_units,
                  beam_width,
                  encoder_length,
                  max_gradient_norm,
                  embedding_size,
                  initial_learning_rate):
    
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    prob = tf.placeholder(shape=(batch_size),dtype=tf.int32, name="label")

    encoder_outputs, encoder_state = build_encoder(encoder_inputs,encoder_length,num_units,keep_prob)
    
    
    decoder_inputs = tf.placeholder(tf.int32, shape=(decoder_length, batch_size), name="decoder_inputs")
    decoder_lengths = tf.placeholder(tf.int32, shape=(batch_size), name="decoer_length")
    target_labels = tf.placeholder(tf.int32,shape=(batch_size,decoder_length), name="target_label")

    
    logits, sample_ids = build_decoder(tgt_vocab_size,
                                       decoder_inputs,
                                       num_units,
                                       encoder_outputs,
                                       is_training,
                                       beam_width,
                                       decoder_lengths,
                                       encoder_state,
                                       batch_size,
                                       encoder_length,
                                       prob
                                       )
    

              
    if is_training == True:    

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_labels, logits=logits)
        loss = tf.reduce_sum(loss)
        

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, max_gradient_norm)

        optimizer = tf.train.GradientDescentOptimizer(initial_learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params))
        

    else : 
       
        loss = tf.no_op()
        train_op = tf.no_op()
    
    return train_op, loss, sample_ids, logits, decoder_inputs,decoder_lengths, target_labels, keep_prob, prob
