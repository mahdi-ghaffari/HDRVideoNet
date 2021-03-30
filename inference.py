import torch
import os
from tqdm import tqdm
from utils import warp
import cv2
import numpy as np


def inference(config, model, test_loader, criterion):

    with torch.no_grad():

        model.eval()

        loss_rec_sum = 0.0
        loss_tem_sum = 0.0
        loss_sum = 0.0
        loss_cnt = 0

        for _ , (inputs, masks, output_name, target) in enumerate(tqdm(test_loader)):
            
            inputs = inputs.to(config.device)
            mask  = masks[:, :, 2, :,:].to(config.device)
            batch_size_mod = inputs.size()[0]

            target = target.to(config.device)
            
            if(config.model_shape == 'Multiple'):
                output_net = model(inputs)
                output     = torch.add(output_net, inputs[:,:,2,:,:])
            else :# (config.model_shape == 'Single'):
                output_net = model(inputs[:,:,2,:,:])
                output     = torch.add(output_net, inputs[:,:,2,:,:])

            scale = torch.tensor(1.5)
            scale = scale.to(config.device)
            output_ill = torch.mul(output, mask)
            target_ill = torch.mul(scale*target, mask)

            loss_rec = criterion(output_ill, target_ill)
            loss_rec_sum += loss_rec.item()
            loss_cnt     += 1

            output = output.permute(0, 2, 3, 1)
            mask   = mask.permute(0, 2, 3, 1)
            input  = inputs[:,:,2,:,:].permute(0, 2, 3, 1)

            input_lin = torch.pow(torch.div(0.6*input, torch.clamp(1.6-input, min=config.eps)), 1.0/0.9)

            mask_inv = torch.sub(1.0, mask)
            output = torch.add(torch.mul(mask, output), torch.mul(mask_inv, input_lin))
            # output = torch.add(mask, torch.mul(mask_inv, input_lin)) #without net output


            output = output.cpu().detach().numpy()

            output[output < 0.0]= 0.0
           
            max_val = np.max(output)

            output = output / max_val

            output_mod = (output*(2**16 - 1)).astype('uint16')


            for b in range(batch_size_mod):
                output_file_name = os.path.join(output_name[b])
                print(output_file_name)
                cv2.imwrite(output_file_name, output_mod[b, :, :, :])
                       
        print('Loss :',loss_rec_sum/loss_cnt)

