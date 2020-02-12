import numpy as np
import tensorflow as tf
from dataset import Dataset
import time
from network import build_network
from tools import get_label, load_train_img
from parse_args import parse_args

FLAGS = parse_args()

def train():
    with open(FLAGS.train_txt) as f:
        sample = [line.rstrip() for line in f]
    
    sample = np.array(sample)
    data = Dataset(sample)
    
    tf.reset_default_graph()
    train_graph = tf.Graph()
    
    with train_graph.as_default():
    
        train_op, loss , sample_ids,logits, inputs, decoder_inputs, decoder_lengths, \
        target_labels = build_network(is_training=True,
                                                 batch_size=FLAGS.batch_size,
                                                 height=FLAGS.height,
                                                 width=FLAGS.width,
                                                 channels=FLAGS.channels,
                                                 decoder_length=FLAGS.decoder_length,
                                                 tgt_vocab_size=FLAGS.tgt_vocab_size,
                                                 num_units=FLAGS.num_units,
                                                 beam_width=FLAGS.beam_width,
                                                 encoder_length=FLAGS.encoder_length,
                                                 max_gradient_norm=FLAGS.max_gradient_norm,
                                                 embedding_size=FLAGS.embedding_size)
    
        initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()
    
    train_sess = tf.Session(graph=train_graph)
    train_sess.run(initializer)
    
    start = time.time()
    
    for i in range(FLAGS.iteration):
        batch_train = data.next_batch(FLAGS.batch_size)
        path = []
        texts = []   
        for line in batch_train:
            path.append(line.split(' ')[0])
            texts.append(line.split(' ')[1])
        
    
        images = load_train_img(path,FLAGS.height,FLAGS.width)
    
        training_target_labels = get_label(texts,FLAGS.decoder_length)
        training_decoder_inputs = np.delete(training_target_labels, -1, axis=1)
        training_decoder_inputs = np.c_[ np.zeros(training_decoder_inputs.shape[0]), training_decoder_inputs].T
        feed_dict = {inputs:images[:, :, :, np.newaxis],decoder_inputs:training_decoder_inputs,
                 decoder_lengths:np.ones((FLAGS.batch_size), dtype=int) * FLAGS.decoder_length,
                 target_labels:training_target_labels} 
        _,loss_value = train_sess.run([train_op, loss], feed_dict=feed_dict)
        
        
        step = float(i+1)
        if step % FLAGS.display_step == 0:
    
            now = time.time()
            print(step, now-start, loss_value)
            start = now
            
        if step % FLAGS.save_step == 0:
    
            train_saver.save(train_sess,FLAGS.save_dir)

if __name__ == '__main__':
    train()












































