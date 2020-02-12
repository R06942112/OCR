import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import tools
from cnn import build_cnn




       
def build_encoder(encoder_inputs,encoder_length,num_units,keep_prob):
    
    with tf.variable_scope('encoder'):
        encoder_inputs = tf.transpose(encoder_inputs, [1, 0, 2])
        
    
        cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units,reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units,reuse=tf.get_variable_scope().reuse)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
        encoder_outputs , encoder_state = tf.nn.bidirectional_dynamic_rnn( cell_fw , cell_bw , encoder_inputs , time_major=True,dtype=tf.float32,scope='BLSTM_1')
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
                  embedding_size,
                  initial_learning_rate):
    
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    encoder_inputs, inputs = build_cnn(is_training, batch_size, height, width, channels)
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
                                       encoder_length
                                       )
    

    
    
    
    
    
   
              
    if is_training == True:    
        # Target labels
        #   As described in doc for sparse_softmax_cross_entropy_with_logits,
        #   labels should be [batch_size, decoder_lengths] instead of [batch_size, decoder_lengths, tgt_vocab_size].
        #   So labels should have indices instead of tgt_vocab_size classes.
        
        # Loss
        #global_step = tf.Variable(0, name='global_step', trainable=False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_labels, logits=logits)
        loss = tf.reduce_sum(loss)
        

        

        
#        loss_l = tf.nn.softmax_cross_entropy_with_logits(labels=inputs_l,logits=o)
#        loss_l = tf.reduce_sum(loss_l)
#        
#        loss = loss_c + 0.1*loss_l
        
        # Train
        
        
        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
        
        #learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=1000000,decay_rate=0.1,staircase=True)
        #learning_rate = tf.train.cosine_decay_restarts(initial_learning_rate, global_step, 400000,
        #                  t_mul=1.0, m_mul=0.5, alpha=0.0, name=None)

        # Optimization
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        #learning_rate = initial_learning_rate
        optimizer = tf.train.GradientDescentOptimizer(initial_learning_rate)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=initial_learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params)),# global_step=global_step)
        
        #optimizer = tf.train.GradientDescentOptimizer(hparams.learning_rate)
        #train_op = optimizer.minimize(loss, global_step=global_step)

    else : 

        
        loss = tf.no_op()
        train_op = tf.no_op()
        #learning_rate = tf.no_op()

        
    return train_op, loss, sample_ids, logits, inputs, decoder_inputs,decoder_lengths, target_labels, keep_prob#, learning_rate
    #return train_op, loss, sample_ids, logits, inputs, decoder_inputs, decoder_lengths, target_labels, keep_prob, learning_rate

