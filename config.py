import os
import numpy as np
import torch

class Config():

    def __init__(self):
    
        self.root = '/mnt/db/Dataset/' 
        print('Dataset : ', self.root)

        self.state = 'Train' #'Train' or 'Inference'
        print('State : ', self.state)

        self.validation = True
        print('Validation: ', self.validation)

        self.pretrain = False
        print('pretrain : ', self.pretrain )

        self.num_epochs = 20
        print('Epochs : ', self.num_epochs)
   
        self.batch_size = 8
        print('Batch size : ', self.batch_size )

        self.debug = False
        print('Debug: ', self.debug)

        if (self.state == 'Train') and self.pretrain:
            pass   

        elif self.state == 'Inference':
            pass

        if(self.debug):
            self.debug_num_samples = self.batch_size*16
        
        self.model_shape = 'Multiple'   #'Single'  # 'Multiple'
        print('Model Shape : ', self.model_shape)

        self.loss = 'Rec&Tem' # 'Rec', 'Rec&Tem'
        print('Loss Type: ', self.loss)

        self.fileName = "hdrvideonet_dataset.pkl"

        self.root_model_path = "./models"

        if not os.path.exists(self.root_model_path):
            os.mkdir(self.root_model_path)    
        print('Saving Model in: ', self.root_model_path)

        self.in_channels = 3
        self.out_channels = 3

        self.frame_size = (360, 640) #(Height, Width)
        print('frame size: ', self.frame_size)

        self.data_split_ratio = 0.75
        print('data split ratio: ', self.data_split_ratio)

        self.num_workers = 4

        self.temporal_num = 5

        self.alpha = 0.8

        self.hdr_format = "tiff"

        self.down_samp = 10
        print("Down Sample Rate : {}".format(self.down_samp))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device: ', self.device)

        self.eps = 1/(2**16 -1)

        print('########################################################################################################')
        








        
