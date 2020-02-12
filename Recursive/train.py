from parse_args import parse_args
import numpy as np
import tensorflow as tf
from dataset import Dataset
import time
from cnn import build_cnn
from network import build_network
from tools import get_label, load_img
import tools

flags = parse_args()
def train():
    with open(flags.train_txt) as f:
        sample = [line.rstrip() for line in f]
    sample = np.array(sample)
    iteration = len(sample)//flags.batch_size
    data = Dataset(sample)
     
    tf.reset_default_graph()
    train_graph = tf.Graph()
    infer_graph = tf.Graph()
    
    with train_graph.as_default():     
        encoder_outputs,inputs = build_cnn(True,flags.batch_size,flags.height, flags.width, flags.channels)  
        train_op, loss, sample_ids, logits, decoder_inputs, \
        target_labels, learning_rate,keep_prob = build_network(encoder_outputs,
                                                     False,                                     
                                                     flags.batch_size,
                                                     flags.decoder_length,
                                                     flags.tgt_vocab_size,
                                                     flags.attn_num_hidden,
                                                     flags.encoder_length,
                                                     flags.max_gradient_norm
                                                     )
        initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()
    
    train_sess = tf.Session(graph=train_graph)
    train_sess.run(initializer)
    
    with infer_graph.as_default():
        encoder_outputs_t,inputs_t = build_cnn(False,flags.batch_size,flags.height, flags.width, flags.channels)
        _, _, pred_ids, logits_t, decoder_inputs_t, \
        _, _ ,keep_prob_t= build_network(encoder_outputs_t,
                             True,                                     
                             flags.batch_size,
                             flags.decoder_length,
                             flags.tgt_vocab_size,
                             flags.attn_num_hidden,
                             flags.encoder_length,
                             flags.max_gradient_norm
                             )
        infer_saver = tf.train.Saver()
    infer_sess = tf.Session(graph=infer_graph)
    
    # Training
    
    start = time.time()    
    acc_log = 0
    count = 0
    lr = flags.learning_rate
    for h in range(flags.epoch):
        for i in range(iteration):
            batch_train = data.next_batch(flags.batch_size)
            path = []
            texts = []   
            for line in batch_train:
                path.append(line.split(' ')[0])
                texts.append(line.split(' ')[1])
        
            images = load_img(path,flags.height,flags.width)
            
            training_target_labels = get_label(texts,flags.decoder_length)
            training_decoder_inputs = np.delete(training_target_labels, -1, axis=1)
            training_decoder_inputs = np.c_[ np.zeros(training_decoder_inputs.shape[0]), training_decoder_inputs].T
            
            
            feed_dict = {inputs:images[:, :, :, np.newaxis],decoder_inputs:training_decoder_inputs,
                     target_labels:training_target_labels,learning_rate:lr,keep_prob:0.5} 
            _,loss_value = train_sess.run([train_op, loss], feed_dict=feed_dict)
            
            step = float(i+1)
            if step % flags.display_step == 0:
        
                now = time.time()
                print(step, now-start, loss_value)
                start = now
            
            
            if step % flags.eval_step == 0:
                train_saver.save(train_sess,flags.save_dir) 
                model_file=tf.train.latest_checkpoint(flags.save_dir.rsplit('/',1)[0])
                infer_saver.restore(infer_sess, model_file)
            
                gt = []
                predict = []
                    
                images = load_img(path,flags.height,flags.width)
        
                testing_decoder_inputs = np.zeros((flags.decoder_length,flags.batch_size), dtype=float)
                feed_dict_t = {inputs_t:images[:, :, :, np.newaxis],
                               decoder_inputs_t:testing_decoder_inputs,keep_prob_t:1}
                q= infer_sess.run( pred_ids,feed_dict=feed_dict_t)
        
                for j in range(flags.batch_size):
                    gt.append(texts[j])
                    ans = np.array(q).T[j]
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
                count = count + 1
                acc_s = correct / cnt
                if acc_s > acc_log:
                    acc_log = acc_s
                    count = 0
                if count == (iteration // flags.eval_step):
                    lr = lr / 5
                
if __name__ == '__main__':
    train()

