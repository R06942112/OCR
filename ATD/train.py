from parse_args import parse_args
import numpy as np
import tensorflow as tf
from network import build_network
from network import build_cnn

flags = parse_args()

def train():
    from dataset import Dataset
    from tools import get_label, load_train_img
    import time
    
    with open(flags.train_s_txt) as f:
        sample = [line.rstrip() for line in f]
    sample = np.array(sample)
    data1 = Dataset(sample)
    
    with open(flags.train_u_txt) as f:
        sample =  [line.rstrip() for line in f]  
        
    sample = np.array(sample)
    data2 = Dataset(sample)
    
    
    tf.reset_default_graph()
    train_graph = tf.Graph()
    
    with train_graph.as_default():
        
        encoder_inputs, x2, inputs =  build_cnn(training=True,batch_size=flags.batch_size, height=flags.height, width=flags.width, channels=flags.channels)
    
        train_op, loss , sample_ids,logits, decoder_inputs, decoder_lengths, \
        target_labels, keep_prob, prob = build_network(encoder_inputs,
                                                 is_training=True,
                                                 batch_size=flags.batch_size,
                                                 decoder_length=flags.decoder_length,
                                                 tgt_vocab_size=flags.tgt_vocab_size,
                                                 num_units=flags.num_units,
                                                 beam_width=flags.beam_width,
                                                 encoder_length=flags.encoder_length,
                                                 max_gradient_norm=flags.max_gradient_norm,
                                                 embedding_size=flags.embedding_size,
                                                 initial_learning_rate=flags.learning_rate)
    
        initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()
    
    train_sess = tf.Session(graph=train_graph)
    train_sess.run(initializer)
    
    
    
    start = time.time()
    for i in range(flags.iteration):
        train_batch = int(flags.batch_size/2)
        batch_train1 = data1.next_batch(train_batch)
        batch_train2 = data2.next_batch(train_batch)
        batch_train = np.append(batch_train1,batch_train2)
        np.random.shuffle(batch_train)
        path = []
        texts = []   
        label = []
        for line in batch_train:
            path.append(line.split(' ')[0])
            texts.append(line.split(' ')[1])
            label.append(line.split(' ')[2])
        
        label = np.array(label).astype(np.int32)
        images = load_train_img(path,flags.height,flags.width)
        
        training_target_labels = get_label(texts,flags.decoder_length)
        training_decoder_inputs = np.delete(training_target_labels, -1, axis=1)
        training_decoder_inputs = np.column_stack([label, np.delete(training_target_labels, -1, axis=1)]).T
        feed_dict = {inputs:images[:, :, :, np.newaxis],decoder_inputs:training_decoder_inputs,
                 decoder_lengths:np.ones((flags.batch_size), dtype=int) * flags.decoder_length,
                 target_labels:training_target_labels,keep_prob:0.8,prob:label} 
        _,loss_value = train_sess.run([train_op, loss], feed_dict=feed_dict)
    
        
        step = float(i+1)
        if step % flags.display_step == 0:
            now = time.time()
            print(step, now-start, loss_value)
            start = now
        
        if step % flags.save_step == 0:
            train_saver.save(train_sess,flags.save_dir)


    
if __name__ == '__main__':
    train()

    


    

    







    
    
    
    
