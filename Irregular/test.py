import numpy as np
import tensorflow as tf
from network import build_network, build_cnn, build_deconv
from tools import load_img
import tools
import editdistance
from parse_args import parse_args

flags = parse_args()





class Dataset:

    def __init__(self,x):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._x = x
  
        self._num_examples = len(x)
        pass
    
    @property
    def x(self):
        return self._x
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes

            self._x = self.x[idx]  # get list of `num` random samples
    
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            x_rest_part = self._x[start:self._num_examples]
            
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes

            self._x = self.x[idx0]  # get list of `num` random samples
    
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            x_new_part =  self._x[start:end]  

            return np.concatenate((x_rest_part, x_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._x[start:end]
        
def test():
    f_size = int(flags.img_size/8)
    encoder_length = f_size * f_size

    tf.reset_default_graph()
    infer_graph = tf.Graph()
    
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
    
    model_file=tf.train.latest_checkpoint(flags.load_dir)
    infer_saver.restore(infer_sess, model_file)
    
    with open(flags.test_txt) as f:
        test = [line.rstrip() for line in f]
    test_len = len(test)
    test = np.array(test)
    data_test = Dataset(test)
    
    if flags.lex_txt !=None:
        with open(flags.lex_txt) as f:
            lex = [line.rstrip().lower() for line in f]
    
    
    ti = int(test_len / flags.batch_size)
    rest = test_len % flags.batch_size
    
    gt = []
    predict = []
    
    for t in range(ti):
        batch_test = data_test.next_batch(flags.batch_size)
        path = []
        texts = []   
        for line in batch_test:
            path.append(line.split(' ')[0])
            texts.append(line.split(' ')[1])
    
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
            
    batch_test = data_test.next_batch(flags.batch_size)
    path = []
    texts = []   
    for line in batch_test:
        path.append(line.split(' ',1)[0])
        texts.append(line.split(' ',1)[1])
    images = load_img(path,flags.img_size)
    
    
    feed_dict_t = {inputs_t:images,
                       decoder_inputs_t:testing_decoder_inputs}
    q = infer_sess.run( pred_ids,feed_dict=feed_dict_t)
        
        
    for k in range(rest):
        gt.append(texts[k])
        ans =  np.array(q).T[k]
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
    
    for l in range(len(gt)):
        cnt =cnt + 1
        if gt[l] == predict[l]:
            correct = correct + 1 
                  
    acc_s = correct / cnt
    
    
    if flags.lex_txt !=None:        
        correct_l = float(0) 
        cnt = 0
        for l in range(len(gt)):
            cnt =cnt + 1
            lexicon = lex[l].split(',')
            dt = editdistance.eval(predict[l], lexicon[0])
            pl = lexicon[0]
            for ll in lexicon[1:]:
                dt_temp = editdistance.eval(predict[l], ll)
                
                if dt_temp < dt:
                    dt = dt_temp
                    pl = ll
            if pl == gt[l]:
                correct_l = correct_l + 1
               
        acc_l = correct_l / cnt  
    
      
    print('accuracy: ', acc_s)
    if flags.lex_txt !=None:
        print('accuracy with lexicon: ', acc_l)
        
if __name__ == '__main__':
    test()