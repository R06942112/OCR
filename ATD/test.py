from parse_args import parse_args
import numpy as np
import tensorflow as tf
from network import build_network
from network import build_classifier, build_cnn
import tools

flags = parse_args()

def test():
    from dataset import Dataset_test
    from tools import load_img
    import editdistance

    tf.reset_default_graph()
    infer_graph = tf.Graph()

    with infer_graph.as_default():
        encoder_inputs_t, x2_t, inputs_t =  build_cnn(training=False,batch_size=flags.batch_size, height=flags.height, width=flags.width, channels=flags.channels)
        _, _,  pred_ids,pred_logits , decoder_inputs_t, decoder_lengths_t, \
        _,keep_prob_t,prob_t = build_network(encoder_inputs_t,
                                                     is_training=False,
                                                     batch_size=flags.batch_size,
                                                     decoder_length=flags.decoder_length,
                                                     tgt_vocab_size=flags.tgt_vocab_size,
                                                     num_units=flags.num_units,
                                                     beam_width=flags.beam_width,
                                                     encoder_length=flags.encoder_length,
                                                     max_gradient_norm=None,
                                                     embedding_size=flags.embedding_size,
                                                     initial_learning_rate=None)

        infer_saver = tf.train.Saver()
    infer_sess = tf.Session(graph=infer_graph)
    model_file=tf.train.latest_checkpoint(flags.r_path)
    print(flags.r_path)
    infer_saver.restore(infer_sess, model_file)

    class_graph = tf.Graph()
    with class_graph.as_default():
    
        prob,inputs_f,inputs_l,loss,train_op = build_classifier(training=False,
                                                     batch_size=flags.batch_size,
                                                     c_learning_rate=None)

        class_saver = tf.train.Saver()

    class_sess = tf.Session(graph=class_graph)
    model_file=tf.train.latest_checkpoint(flags.c_path)
    class_saver.restore(class_sess, model_file)

    with open(flags.test_txt) as f:
        test = [line.rstrip() for line in f]
    test_len = len(test)
    test = np.array(test)
    data_test = Dataset_test(test)
    
    if flags.lex_txt != None:
        with open(flags.lex_txt) as f:
            lex = [line.rstrip().lower() for line in f]
    
    steps = int(test_len / flags.batch_size)
    rest = test_len % flags.batch_size

    predict_c = []
    path_log = []
    labelc = []
    
    for t in range(steps):
        batch_test = data_test.next_batch(flags.batch_size)
        path = [] 
        label = np.tile([1,0],(256,1))
        for line in batch_test:
            path.append(line.split(' ',1)[0])
            path_log.append(line.split(' ',1)[0])
        images = load_img(path,flags.height,flags.width)       
        feed_dict_t = {inputs_t:images[:, :, :, np.newaxis],
                       decoder_lengths_t:np.ones((flags.batch_size), \
                       dtype=int) * flags.decoder_length,
                       keep_prob_t:1,prob_t:np.zeros(flags.batch_size)}
        feature = infer_sess.run(x2_t, feed_dict=feed_dict_t)
        
        feed_dict = {inputs_f:feature,inputs_l:label}
        o = class_sess.run(prob, feed_dict=feed_dict)


        for j in range(len(label)):
            predict_c.append(o[j])
            labelc.append(np.argmax(o[j]))
    
            
    batch_test = data_test.next_batch(flags.batch_size)
    path = [] 
    label = np.tile([1,0],(256,1))
    for line in batch_test:
        path.append(line.split(' ',1)[0])
    images = load_img(path,flags.height,flags.width)
  
    feed_dict_t = {inputs_t:images[:, :, :, np.newaxis],
                   decoder_lengths_t:np.ones((flags.batch_size), \
                   dtype=int) * flags.decoder_length,
                   keep_prob_t:1,prob_t:np.zeros(flags.batch_size)}
    feature = infer_sess.run(x2_t, feed_dict=feed_dict_t)
    
    feed_dict = {inputs_f:feature,inputs_l:label}
    o = class_sess.run(prob, feed_dict=feed_dict)
    for j in range(len(label)):
        labelc.append(np.argmax(o[j]))
           
    for k in range(rest):
        predict_c.append(o[k])
    
    correct = float(0)  
    cnt = 0
    acc_c = 0
    
    for l in range(len(predict_c)):
        cnt =cnt + 1  
        if 0 == np.argmax(predict_c[l]):       
            correct = correct + 1 
    
    acc_c = correct / cnt
    #print('acc_c:', acc_c)
    
    with open(flags.test_txt) as f:
        test = [line.rstrip() for line in f]
    test_len = len(test)
    test = np.array(test)
    data_test = Dataset_test(test)
    
    with open(flags.lex_txt) as f:
        lex = [line.rstrip().lower() for line in f]
    
    steps = int(test_len /flags.batch_size)
    rest = test_len % flags.batch_size
     
    gt = []
    predict = []

    for t in range(steps):
        batch_test = data_test.next_batch(flags.batch_size)
        path = []
        texts = []   
        label = labelc[256*(t):256*(t+1)]
        for line in batch_test:
            path.append(line.split(' ',1)[0])
            texts.append(line.split(' ',1)[1])
    
        images = load_img(path,flags.height,flags.width)
        
        feed_dict_t = {inputs_t:images[:, :, :, np.newaxis],
                       decoder_lengths_t:np.ones((flags.batch_size), \
                       dtype=int) * flags.decoder_length,
                       keep_prob_t:1,prob_t:label}
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
                    
    batch_test = data_test.next_batch(flags.batch_size)
    path = []
    texts = []   
    label = labelc[-256:]
    for line in batch_test:
        path.append(line.split(' ',1)[0])
        texts.append(line.split(' ',1)[1])
    
    images = load_img(path,flags.height,flags.width)
    
    feed_dict_t = {inputs_t:images[:, :, :, np.newaxis],
                   decoder_lengths_t:np.ones((flags.batch_size), \
                   dtype=int) * flags.decoder_length,
                   keep_prob_t:1,prob_t:label}
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
    if flags.lex_txt != None:     
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
    if flags.lex_txt != None:
        print('accuracy with lexicon: ', acc_l)
    
if __name__ == '__main__':
    test()
    
    


    

    







    
    
    
    
