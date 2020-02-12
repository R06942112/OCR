import numpy as np

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
            np.random.shuffle(idx)  # shuffle indexe
            self._x = self.x[idx]  # get list of `num` random samples
    
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            x_rest_part = self._x[start:self._num_examples]
            
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
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
        
class Dataset_test:

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
        