import numpy as np
import tensorflow as tf
from dataset import Dataset
from network import build_network
from tools import get_label, load_img, load_train_img
import tools
from parse_args import parse_args
import time

FLAGS = parse_args()

def train():
    
    with open(FLAGS.train_txt) as f:
        sample = [line.rstrip() for line in f]
     
    sample = np.array(sample)
    iteration = len(sample)//FLAGS.batch_size
    data = Dataset(sample)
    
    tf.reset_default_graph()
    train_graph = tf.Graph()
    infer_graph = tf.Graph()
    
    with train_graph.as_default():
    
        train_op, loss , sample_ids,logits, inputs, decoder_inputs, decoder_lengths, \
        target_labels, keep_prob = build_network(is_training=True,
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
                                                 initial_learning_rate=FLAGS.learning_rate)
    
        initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()
    
    train_sess = tf.Session(graph=train_graph)
    train_sess.run(initializer)
    
    with infer_graph.as_default():
        _, _,  pred_ids,pred_logits , inputs_t, decoder_inputs_t, decoder_lengths_t, \
        _,keep_prob_t = build_network(is_training=False,
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
                                                     initial_learning_rate=None)
        infer_saver = tf.train.Saver()
    infer_sess = tf.Session(graph=infer_graph)
    
    start = time.time()
    acc_log = 0
    count = 0
    lr = FLAGS.learning_rate
    for h in range(FLAGS.epoch):
        for i in range(iteration):
            batch_train = data.next_batch(FLAGS.batch_size)
            np.random.shuffle(batch_train)
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
                     target_labels:training_target_labels,keep_prob:0.8} 
            _,loss_value = train_sess.run([train_op, loss], feed_dict=feed_dict)
        
            
            step = float(i)
            if step % FLAGS.display_step == 0:
                now = time.time()
                print(step, now-start, loss_value)
                start = now
        
                
            if step % FLAGS.eval_step == 0:
                train_saver.save(train_sess,FLAGS.save_dir) 
                model_file=tf.train.latest_checkpoint(FLAGS.save_dir.rsplit('/',1)[0])
                infer_saver.restore(infer_sess, model_file)
                 
                gt = []
                predict = []
        
        
        
                images = load_img(path,FLAGS.height,FLAGS.width)
                
                feed_dict_t = {inputs_t:images[:, :, :, np.newaxis],
                               decoder_lengths_t:np.ones((FLAGS.batch_size), \
                               dtype=int) * FLAGS.decoder_length,
                               keep_prob_t:1}
                q= infer_sess.run( pred_ids,feed_dict=feed_dict_t)
        
                       
                for j in range(len(texts)):
                    gt.append(texts[j])
                    ans = q[j].T[0]
            
                    pd = []
                    for c in ans:
                        if c != -1:
                            character = tools.idx_to_word[c]
                            if character != '<EOS>':
                                pd.append(character)
                    predict.append(''.join(pd))
        

                
                correct = float(0)  
                cnt = 0
                acc_s = 0
                
                for l in range(len(gt)):
                    cnt =cnt + 1
                    if gt[l] == predict[l]:
                        correct = correct + 1 
                
                        
                acc_s = correct / cnt
                if acc_s > acc_log:
                    acc_log = acc_s
                    count = 0
                if count == (iteration // FLAGS.eval_step):
                    lr = lr / 5
        
if __name__ == '__main__':
    train()
