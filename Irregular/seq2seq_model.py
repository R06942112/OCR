"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from seq2seq import model_with_buckets
from seq2seq import embedding_attention_decoder



def build_seq2seq(encoder_inputs_tensor,
                 decoder_inputs,
                 target_vocab_size,
                 encoder_length,
                 decoder_length,
                 attn_num_layers,
                 attn_num_hidden,
                 forward_only):
            # Create the internal multi-layer cell for our RNN.

        single_cell = tf.contrib.rnn.BasicLSTMCell(
            attn_num_hidden, forget_bias=0.0, state_is_tuple=False
        )


        cell = single_cell

        if attn_num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell(
                [single_cell] * attn_num_layers, state_is_tuple=False
            )

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(lstm_inputs, decoder_inputs, do_decode):

            batch_size = tf.shape(decoder_inputs[0])[0] 
            encoder_inputs = lstm_inputs
            top_states = [tf.reshape(e, [-1, 1, 258])
                          for e in encoder_inputs]
            attention_states = tf.concat(top_states, 1)
            #print('here',batch_size)
            initial_state = cell.zero_state(batch_size, tf.float32)
            outputs, _, attention_weights_history = embedding_attention_decoder(
                decoder_inputs, initial_state, attention_states, cell,
                num_symbols=target_vocab_size,
                tgt_vocab_size=target_vocab_size,
                num_heads=1,
                output_size=target_vocab_size,
                output_projection=None,
                feed_previous=do_decode,
                initial_state_attention=False,
                attn_num_hidden=attn_num_hidden)
            return outputs, attention_weights_history

        
        encoder_inputs = tf.split(encoder_inputs_tensor, encoder_length, 0)
        encoder_inputs = [tf.squeeze(inp, squeeze_dims=[0]) for inp in encoder_inputs]
        outputs, attention_weights_history = seq2seq_f(encoder_inputs,
                                                    decoder_inputs,forward_only)

#        bucket_outputs, attention_weights_history = seq2seq(encoder_inputs[:int(bucket[0])],
#                                                                decoder_inputs[:int(bucket[1])],
#                                                                int(bucket[0]))
        
        return outputs, attention_weights_history
    
