import numpy as np
import tensorflow as tf
from dataset import Dataset_test
from network import build_network
from tools import load_img
import tools
import distance
from parse_args import parse_args

FLAGS = parse_args()

def test():

    tf.reset_default_graph()
    infer_graph = tf.Graph()
    
    
    with infer_graph.as_default():
        _, _,  pred_ids,pred_logits , inputs_t, decoder_inputs_t, decoder_lengths_t, \
        _ = build_network(is_training=False,
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
        infer_saver = tf.train.Saver()
    infer_sess = tf.Session(graph=infer_graph)
    

    with open(FLAGS.test_txt) as f:
        test = [line.rstrip() for line in f]
    test_len = len(test)
    
    test = np.array(test)
    data_test = Dataset_test(test)
    
    
    if FLAGS.lex_txt != None:
        with open(FLAGS.lex_txt) as f:
            lex = [line.rstrip().lower() for line in f]
                
    ti = int(test_len / FLAGS.batch_size)
    rest = test_len % FLAGS.batch_size
    gt = []
    predict = []
    model_file=tf.train.latest_checkpoint(FLAGS.load_dir)
    infer_saver.restore(infer_sess, model_file)
    for t in range(ti):
        batch_test = data_test.next_batch(FLAGS.batch_size)
        path = []
        texts = []   
        for line in batch_test:
            path.append(line.split(' ',1)[0])
            
            texts.append(line.split(' ',1)[1])
        images = load_img(path,FLAGS.height,FLAGS.width)
        
        feed_dict_t = {inputs_t:images[:, :, :, np.newaxis],
                       decoder_lengths_t:np.ones((FLAGS.batch_size), \
                       dtype=int) * FLAGS.decoder_length
                       }
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
                   dtype=int) * FLAGS.decoder_length
                   }
    q = infer_sess.run( pred_ids,feed_dict=feed_dict_t)
                       
    for k in range(rest):
        gt.append(texts[k])
    
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
            dt = distance.levenshtein(predict[l], lexicon[0])
            pl = lexicon[0]
            for ll in lexicon[1:]:
                dt_temp = distance.levenshtein(predict[l], ll)
                if dt_temp < dt:
                    dt = dt_temp
                    pl = ll
            if pl == gt[l]:
                correct_l = correct_l + 1
        acc_l = correct_l / cnt   
        
    print('acc_s: ', acc_s)
    if FLAGS.lex_txt != None:
        print('acc_l: ', acc_l)
    
if __name__ == '__main__':
    test()