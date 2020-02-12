import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import editdistance
import numpy as np
import argparse

import models.crnn as crnn

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
        

def load_img(path):
    
    transformer = dataset.resizeNormalize((100, 32))

    
    result = []   
    for p in path: 
        image = Image.open(p).convert('L')
        image = transformer(image)
        result.append(image)
        
    return torch.stack(result)

def test(test_txt,lex_txt):
    batch_size = 256
    model_path = './data/crnn.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))
    
    with open(test_txt) as f:
        test = ['/'+line.rstrip() for line in f]
    test_len = len(test)
    test = np.array(test)
    data_test = Dataset(test)
    
    if lex_txt != None:
        with open(lex_txt) as f:
            lex = [line.rstrip().lower() for line in f]
    
    
    
    
    ti = int(test_len / batch_size)
    rest = test_len % batch_size
       
    gt = []
    predict = []
    
    converter = utils.strLabelConverter(alphabet)
    for t in range(ti):
        batch_test = data_test.next_batch(batch_size)
        path = []
        texts = []   
        for line in batch_test:
            path.append(line.split(' ',1)[0])
            texts.append(line.split(' ',1)[1])
            
        images = load_img(path)
        if torch.cuda.is_available():
            images = images.cuda()
    
        images = Variable(images)
        
        model.eval()
        preds = model(images)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)   
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        predict = predict + sim_preds
        gt = gt + texts
    
    batch_test = data_test.next_batch(batch_size)
    path = []
    texts = []   
    for line in batch_test:
        path.append(line.split(' ',1)[0])
        texts.append(line.split(' ',1)[1])
        
    images = load_img(path)
    if torch.cuda.is_available():
        images = images.cuda()
    
    images = Variable(images)
    
    model.eval()
    preds = model(images)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)    
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    predict = predict + sim_preds[:rest]
    gt = gt + texts[:rest]
        
    
            
    correct = float(0)  
    cnt = 0
    acc_s = 0
    
    for l in range(len(gt)):
        cnt =cnt + 1
        if gt[l] == predict[l]:
            correct = correct + 1 
       
    acc_s = correct / cnt        
    
    if lex_txt != None:
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
    
      
    print('acc_s: ', acc_s)
    if lex_txt != None:
        print('acc_l: ', acc_l)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_txt',type=str,default='../dataset/iiit5k.txt',help='txt file of testing images')
    parser.add_argument('--lex_txt',type=str,default='../dataset/iiit5k_lex.txt',help='txt file of testing lexicon')
    args = parser.parse_args()
    test(args.test_txt,args.lex_txt)
        
        
