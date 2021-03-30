### python libraries
import os
import numpy as np
import warnings
warnings.simplefilter('ignore')
### pytorch libraries
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter


### custom libraries
from model import Model, Model_single
from config import Config
from dataset import get_data, inference_dataset_prepare
from train import train_epoch
from validation import validation_epoch
from inference import inference
from utils import current_time_str
from video import ground_truth_video, hdr_video_make, ldr_video_make

if __name__ == '__main__':
  
    config = Config()
    
    state = config.state

    if(state == 'Train'):
        writer = SummaryWriter()

        if (config.pretrain):
            
            model = torch.load(os.path.join(config.root_model_path, config.model_name))

        else:
            if(config.model_shape=="Multiple"):
                model = Model(config)
            else:
                model = Model_single(config)
                

        model.to(config.device)

        criterion = nn.L1Loss()
        
        criterion = criterion.to(config.device)


        training_data =  get_data(config, mode='train')
        if(config.validation):
            validation_data =  get_data(config, mode='validation')


        train_loader = torch.utils.data.DataLoader(training_data, config.batch_size, shuffle=True, pin_memory=True,
                            num_workers=config.num_workers)
        if(config.validation):
            validation_loader = torch.utils.data.DataLoader( validation_data, config.batch_size//2 , shuffle=True,
                                pin_memory=True,num_workers=config.num_workers)


        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

        for epoch in range(1, config.num_epochs+1):

            
            print('=======================================Epoch No : {}/{}========================================='.format(epoch, config.num_epochs))
            
            train_loss      = train_epoch(model, config, train_loader, criterion, optimizer)

            if(config.validation):
                validation_loss = validation_epoch(model, config, validation_loader, criterion)
            
            writer.add_scalar("Loss_reconstruction/Train", train_loss[0], epoch)
            if(config.loss == 'Rec&Tem'):
                writer.add_scalar("Loss_temporal/Train", train_loss[1], epoch)
                writer.add_scalar("Loss_total/Train", train_loss[2], epoch)

            if(config.validation):
                writer.add_scalar("Loss_reconstruction/Validation",validation_loss[0], epoch)
                if(config.loss == 'Rec&Tem'):
                    writer.add_scalar("Loss_temporal/Validation", validation_loss[1], epoch)
                    writer.add_scalar("Loss_total/Validation", validation_loss[2], epoch)
            

            print('Train Loss : {}'.format(train_loss))
            if(config.validation):
                print('Validation Loss : {}'.format(validation_loss))


            model_name = os.path.join(config.root_model_path, 'model-{}-Epoch#{}.pth'.format(current_time_str(), epoch))
            torch.save(model, model_name)

            
        # Closing the SummaryWriter
        writer.close()

        model_name = os.path.join(config.root_model_path, 'model_{}-{}.pth'.format(config.model_shape, current_time_str()))
        torch.save(model, model_name)
    
    elif(state == 'Inference'):
        

        model = torch.load(os.path.join(config.root_model_path, config.model_name))
        model.to(config.device)
                
        criterion = nn.L1Loss()
        criterion = criterion.to(config.device)

        #preparing inference dataset
        inference_dataset_prepare(config)

        test_data =  get_data(config, mode='inference')
        test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, shuffle=False, pin_memory=True,
                            num_workers=config.num_workers)
        
        print('Video frames are loaded!')
        inference(config, model, test_loader, criterion)

        print('Conversion is done!')

        ground_truth_video(config)

        hdr_video_make(config)
        print('HDR Video is ready!')

        ldr_video_make(config)


             
    else:
        print('State is Wrong!')


    

