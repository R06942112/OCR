import tensorflow as tf
from seq2seq_model import build_seq2seq
from six.moves import xrange

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
        
    return num_feed, outputs


    
    
        
     
def build_network(encoder_inputs,
                  forward_only,
                  batch_size,
                  decoder_length,
                  tgt_vocab_size,
                  attn_num_hidden,
                  encoder_length,
                  max_gradient_norm
                  ):

    
    
    decoder_inputs = tf.placeholder(tf.int32, shape=(decoder_length, batch_size), name="decoder_inputs")
    decoder_inputs_u = tf.unstack(decoder_inputs)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    target_labels = tf.placeholder(tf.int32,shape=(batch_size,decoder_length), name="target_label")
    
    
    encoder_outputs, encoder_state  = build_encoder(encoder_inputs,encoder_length,attn_num_hidden/2,keep_prob)
    
    sample_ids, logits = build_decoder(tgt_vocab_size,
                                      decoder_inputs_u,
                                      encoder_outputs,
                                      decoder_length,
                                      encoder_length,
                                      attn_num_hidden,
                                      forward_only
                                      )
    logits = tf.stack(logits,axis=0)
    logits = tf.transpose(logits, [1,0,2])
    
    if not forward_only:  # train
    
        learning_rate = tf.placeholder(tf.float32, shape=[])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_labels, logits=logits)
        loss = tf.reduce_sum(loss)
    
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
#        train_op = optimizer.apply_gradients(
#                    zip(clipped_gradients, params)),
#                        
    else:         
        loss = tf.no_op()
        train_op = tf.no_op()
        learning_rate = tf.no_op()
        
    return train_op, loss, sample_ids, logits, decoder_inputs, target_labels, learning_rate,keep_prob
        
        
        
        
        
    

    


        


