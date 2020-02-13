import numpy as np
import tensorflow as tf
from network import build_network
from tools import load_img
import tools
import editdistance
from parse_args import parse_args

FLAGS = parse_args()


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
    tf.reset_default_graph()
    infer_graph = tf.Graph()
    
    
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
                                                     embedding_size=FLAGS.embedding_size,
                                                     initial_learning_rate=None)
        infer_saver = tf.train.Saver()
    infer_sess = tf.Session(graph=infer_graph)
    
    model_file=tf.train.latest_checkpoint(FLAGS.load_dir)
    infer_saver.restore(infer_sess, model_file)
    
    with open(FLAGS.test_txt) as f:
        test = [line.rstrip() for line in f]
    test_len = len(test)
    
    test = np.array(test)
    data_test = Dataset(test)
    
    
    if FLAGS.lex_txt != None:
        with open(FLAGS.lex_txt) as f:
            lex = [line.rstrip().lower() for line in f]
    

    
    
    ti = int(test_len / FLAGS.batch_size)
    rest = test_len % FLAGS.batch_size
      
    gt = []
    predict = []
    path_log = []
    
    for t in range(ti):
        batch_test = data_test.next_batch(FLAGS.batch_size)
        path = []
        texts = []   
        for line in batch_test:
            path.append(line.split(' ',1)[0])
            path_log.append(line.split(' ',1)[0])
            texts.append(line.split(' ',1)[1])
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
    
            
    batch_test = data_test.next_batch(FLAGS.batch_size)
    path = []
    texts = []   
    for line in batch_test:
        path.append(line.split(' ',1)[0])
        texts.append(line.split(' ',1)[1])
    images = load_img(path,FLAGS.height,FLAGS.width)
    
    feed_dict_t = {inputs_t:images[:, :, :, np.newaxis],
                   decoder_lengths_t:np.ones((FLAGS.batch_size), \
                   dtype=int) * FLAGS.decoder_length,
                   keep_prob_t:1}
    q = infer_sess.run( pred_ids,feed_dict=feed_dict_t)
     
    for k in range(rest):
        gt.append(texts[k])
        path_log.append(path[k])
    
        ans = q[k].T[0]
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
    
    
    if FLAGS.lex_txt != None:            
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
    if FLAGS.lex_txt != None:
        print('accuracy with lexicon: ', acc_l)
     
if __name__ == '__main__':
    test()
    
    


    
    
