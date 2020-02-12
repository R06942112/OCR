import numpy as np
import tensorflow as tf
from dataset import Dataset
import os
import time
from network import build_network, build_cnn, build_deconv
from tools import get_label, load_img, load_img_label
import tools
from parse_args import parse_args

flags = parse_args()

def train():
    f_size = int(flags.img_size/8)
    encoder_length = f_size * f_size
    
    with open(flags.train_txt) as f:
        sample = [line.rstrip() for line in f]
    sample = np.array(sample)
    iteration = len(sample)//flags.batch_size
    data = Dataset(sample)
    
    tf.reset_default_graph()
    train_graph = tf.Graph()
    infer_graph = tf.Graph()
    start = time.time()
    with train_graph.as_default():
        
        c, inputs =  build_cnn(is_training=True,batch_size=flags.batch_size,img_size=flags.img_size, channels=flags.channels)
        deconv_outputs = build_deconv(True,c,flags.batch_size)
        x = np.linspace(-0.5,0.5,f_size)
        x = np.tile(x,(f_size,1))
        y = np.transpose(x)
        x = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=2)
        m = np.concatenate((x,y),axis=2)
        m = np.expand_dims(m, axis=0)
        m = np.repeat(m, flags.batch_size, axis=0)
        m = tf.convert_to_tensor(m, np.float32)
        encoder_outputs = tf.concat([c, m],-1)   
        encoder_outputs = tf.reshape(encoder_outputs, shape=(-1, f_size*f_size, 258))
        encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
    
        train_op, loss , sample_ids,logits, decoder_inputs,  \
        target_labels, learning_rate,attention_weights_history,att_label,lamda,att_mask,input_seg= build_network(encoder_outputs,
                                                                                          False,
                                                                                          flags.batch_size,
                                                                                          flags.decoder_length,
                                                                                          flags.tgt_vocab_size,
                                                                                          flags.attn_num_hidden,
                                                                                          encoder_length,
                                                                                          flags.max_gradient_norm,
                                                                                          f_size,
                                                                                          flags.att_loss,
                                                                                          flags.img_size,
                                                                                          deconv_outputs
                                                                                          )
        initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()
    
    train_sess = tf.Session(graph=train_graph)
    train_sess.run(initializer)
    
    with infer_graph.as_default():
        c_t, inputs_t =  build_cnn(is_training=False,batch_size=flags.batch_size,img_size=flags.img_size, channels=flags.channels)
        deconv_outputs_t = build_deconv(False,c_t,flags.batch_size)
        
        x_t = np.linspace(-0.5,0.5,f_size)
        x_t = np.tile(x_t,(f_size,1))
        y_t = np.transpose(x_t)
        x_t = np.expand_dims(x_t, axis=2)
        y_t = np.expand_dims(y_t, axis=2)
        m_t = np.concatenate((x_t,y_t),axis=2)
        m_t = np.expand_dims(m_t, axis=0)
        m_t = np.repeat(m_t, flags.batch_size, axis=0)
        m_t = tf.convert_to_tensor(m_t, np.float32)
        encoder_outputs_t = tf.concat([c_t, m_t],-1)   
        encoder_outputs_t = tf.reshape(encoder_outputs_t, shape=(-1, f_size*f_size, 258))
        encoder_outputs_t = tf.transpose(encoder_outputs_t, [1, 0, 2])
    
        _, _ , pred_ids,logits_t, decoder_inputs_t,  \
            _, _,_,_,_,_,_= build_network(encoder_outputs_t,
                                      True,
                                      flags.batch_size,
                                      flags.decoder_length,
                                      flags.tgt_vocab_size,
                                      flags.attn_num_hidden,
                                      encoder_length,
                                      flags.max_gradient_norm,
                                      f_size,
                                      flags.att_loss,
                                      flags.img_size,
                                      deconv_outputs_t
                                      )
        infer_saver = tf.train.Saver()
    infer_sess = tf.Session(graph=infer_graph)
    
    # Training

    la = 10
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
                
            if flags.att_loss:
                images,npy,mask,seg = load_img_label(path,flags.img_size,flags.decoder_length)
            else:
                images = load_img(path,flags.img_size)
            
            training_target_labels = get_label(texts,flags.decoder_length)
            training_decoder_inputs = np.delete(training_target_labels, -1, axis=1)
            training_decoder_inputs = np.c_[ np.zeros(training_decoder_inputs.shape[0]), training_decoder_inputs].T
            feed_dict = {inputs:images,decoder_inputs:training_decoder_inputs,
                     target_labels:training_target_labels,learning_rate:lr} 
            if flags.att_loss:
                feed_dict[att_label] = npy
                feed_dict[att_mask] = mask
                feed_dict[input_seg] = seg[:, :, :, np.newaxis]
                feed_dict[lamda] = la
            _,loss_value,att = train_sess.run([train_op, loss,attention_weights_history], feed_dict=feed_dict)
   
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
                
                images = load_img(path,flags.img_size)
    
                testing_decoder_inputs = np.zeros((flags.decoder_length,flags.batch_size), dtype=float)
                feed_dict_t = {inputs_t:images,
                               decoder_inputs_t:testing_decoder_inputs}
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
        
    


