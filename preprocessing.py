import h5py
import matplotlib.pyplot as plt

class preprocessing:

    def __init__(self):
        # fetch dataset
        self.train_x = h5py.File('./train_signs.h5', 'r')['train_set_x']
        self.train_y = h5py.File('./train_signs.h5', 'r')['train_set_y']
        self.test_x = h5py.File('./test_signs.h5', 'r')['test_set_x']
        self.test_y = h5py.File('./test_signs.h5', 'r')['test_set_y']
        # some info
        self.train_size = len(self.train_y)
        self.test_size = len(self.test_y)
        self.sample_shape = self.train_x[0].shape
    
    def fetch_datasets(self):
        return self.train_x,self.train_y,self.test_x,self.test_y
           
    # print some info
    def print_info(self):
        print("some info about the dataset :-")
        print("train data set size",self.train_size,sep=" : ")
        print("test data set size",self.test_size,sep=" : ")
        print("sample shape",self.sample_shape,sep=" : ")
    
    # plot sample by index
    def plot_sample(self,sample_index):
        if(sample_index < 0 or sample_index >= self.train_size): # checking
            print("invalid sample index !")
            return 
        print("sample",sample_index,"is ploted",sep=" ")
        plt.imshow(self.train_x[sample_index])
        plt.title(self.train_y[sample_index])
        plt.axis('off')
        
        
        
