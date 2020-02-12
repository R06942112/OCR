import cv2
import numpy as np
import math
import time
import random


sos_id = 0
eos_id = 1


#with open('dic_cht.txt','r') as f:
chars = '0123456789abcdefghijklmnopqrstuvwxyz'
word_to_idx = {chars[i]:i+2 for i in range(len(chars))}
word_to_idx['<SOS>'] = 0
word_to_idx['<EOS>'] = 1
#size= len(word_to_idx)
#for i in range(len(chars)):
#    word_to_idx['<'+chars[i]+'>'] = i + size
idx_to_word = {0: "<SOS>", 1: "<EOS>"}


for i in range(len(chars)):
    idx_to_word[i+2] = chars[i]
    #idx_to_word[i+size] = '<'+chars[i]+'>'
    
def get_label(texts,decoder_length):
    text_list = []    
    for text in texts :
        char_list = []
        for c in text:
            char_list.append(word_to_idx[c])
        char_list += [eos_id] * (decoder_length - len(char_list))
        text_list.append(char_list)
    return np.array(text_list)

def get_r_label(texts,decoder_length):
    text_list = []    
    for text in texts :
        char_list = []
        for c in reversed(text):
            char_list.append(word_to_idx[c])
        char_list += [eos_id] * (decoder_length - len(char_list))
        text_list.append(char_list)
    return np.array(text_list)

def load_train_img(path,height,width):
    result = []   
    for p in path: 
        image = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        try:
            dims = image.shape
        except:
            print(p)
            
        w = dims[1]
        if w <= width:
            resized = cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA)
        #resized = cv2.equalizeHist(resized)    
        normalized = ((2.0 / 255.0) * resized - 1.0).astype(np.float32)

        
        if random.choice([True, False]):
            normalized = np.rot90(np.rot90(normalized))
            
        result.append(normalized)
    return np.array(result)

def load_img(path,height,width):
    result = []   
    for p in path: 
        image = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        
        dims = image.shape
        w = dims[1]
        if w <= width:
            resized = cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA)
        #resized = cv2.equalizeHist(resized)    
        normalized = ((2.0 / 255.0) * resized - 1.0).astype(np.float32)
        result.append(normalized)
        
    return np.array(result)


        

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
