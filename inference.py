import torch
import os
from tqdm import tqdm
from utils import warp
import cv2
import numpy as np
from tools.VirtualCamera.virtual_camera import VirtualCamera
from tools.img_io import writeEXR
from torch.utils.tensorboard.writer import SummaryWriter

def inference(config, model, test_loader, criterion):

    with torch.no_grad():

        model.eval()

        loss_rec_sum = 0.0
        loss_tem_sum = 0.0
        loss_sum = 0.0
        loss_cnt = 0

        writer = SummaryWriter()

        for batch , (inputs, target, mask_rec, flows, inputs_pre, output_file_name) in enumerate(tqdm(test_loader)):
                     
            if(config.model_shape == 'Multiple'):
                inputs = inputs.to(config.device)
                output_net = model(inputs)
                inputs     = inputs.to('cpu')
                input      = inputs[:,:,2,:,:].to(config.device)
                output     = torch.add(output_net, input)
    
            else:
                input = inputs[:,:,2,:,:].to(config.device)
                output_net = model(input)
                output     = torch.add(output_net, input)
                
            target = target.to(config.device)
         
            mask_rec = mask_rec.to(config.device)
            # mask_rec = gauss_conv(mask_rec)

            output[output < 0] = 0.0
            target[target < 0] = 0.0

            output_ill = torch.mul(output, mask_rec)
            target_ill = torch.mul(target, mask_rec)

            if(config.loss == 'Rec&Tem'):
                flow_fw = flows[0]
                flow_fw = flow_fw.to(config.device)
                pass

            if(config.loss == 'Rec&Tem'):

                temp = inputs[:,:,1,:,:].to(config.device)
                temp =  warp(temp, flow_fw)
                mask_tem = torch.exp(-50*torch.norm((input -
                                temp), dim=1, keepdim=True))
                mask_tem = torch.cat((mask_tem, mask_tem, mask_tem), dim = 1)

                if(config.model_shape == 'Multiple'):
                    inputs_pre = inputs_pre.to(config.device)
                    output_pre_net = model(inputs_pre)
                    output_pre = torch.add(output_pre_net, inputs_pre[:,:,2,:,:])
                    output_pre_warped = warp(output_pre, flow_fw)
                else:
                    input_pre = inputs_pre[:,:,2,:,:].to(config.device)
                    output_pre_net = model(input_pre)
                    output_pre = torch.add(output_pre_net, input_pre)
                    output_pre_warped = warp(output_pre, flow_fw)

                output_pre_warped[output_pre_warped < 0] = 0.0
                output_pre_warped_ill = torch.mul(output_pre_warped, mask_rec)
                loss_tem = criterion(torch.mul(mask_tem,output_ill), torch.mul(mask_tem,output_pre_warped_ill))

            loss_rec = criterion(output_ill, target_ill)
            loss = loss_rec
            if(config.loss == 'Rec&Tem'):
                loss = torch.add(loss , config.alpha*loss_tem)
                pass

            loss_rec_sum += loss_rec.item()
            if(config.loss == 'Rec&Tem'):
                loss_tem_sum += loss_tem.item()
            loss_sum     += loss.item()
            loss_cnt     += 1

            output = output.permute(0, 2, 3, 1)
            mask_rec   = mask_rec.permute(0, 2, 3, 1)
            input  = input.permute(0, 2, 3, 1)

            input_lin = torch.pow(torch.div(0.6*input, torch.clamp(1.6-input, min=config.eps)), 1.0/0.9)

            mask_inv = torch.sub(1.0, mask_rec)
            output = torch.add(torch.mul(mask_rec, output), torch.mul(mask_inv, input_lin))
            # output = torch.add(mask, torch.mul(mask_inv, input_lin)) #without net output

            output = output.cpu().detach().numpy()

            output[output < 0.0]= 0.0
           
            # max_val = np.max(output)
            # output = output / max_val
            # output = (output*(2**16 - 1)).astype('uint16')

            batch_size_mod = inputs.size()[0]
            for b in range(batch_size_mod):
                # cv2.imwrite(output_file_name[b][:-4], output[b, :, :, :])
                writeEXR(output[b, :, :, :], output_file_name[b][:-4] + 'exr')

            
            writer.add_scalar("Loss_reconstruction/Inference", loss_rec_sum/loss_cnt, batch)
            writer.add_scalar("Loss_temporal/Inference", loss_tem_sum/loss_cnt, batch)
            writer.add_scalar("Loss_total/Inference", loss_sum/loss_cnt, batch)


def inference_dataset_prepare(config):

    path = os.path.join(config.root, 'Inference')

    videos_name = os.listdir(path)

    virtual_camera = VirtualCamera(config, action='scene_based')

    for _, video_name in enumerate(videos_name): 

        video_frames = sorted(os.listdir(os.path.join(path, video_name, 'frames')))

        input_path = os.path.join(path, video_name, 'input')
        if os.path.exists(input_path):
            os.removedirs(input_path)
        os.makedirs(input_path)

        mask_path = os.path.join(path, video_name, 'mask')
        if os.path.exists(mask_path):
            os.removedirs(mask_path)
        os.makedirs(mask_path) 

        output_path = os.path.join(path, video_name, 'output')
        if os.path.exists(output_path):
            os.removedirs(output_path)
        os.makedirs(output_path)       

        scene_num = int(video_frames[0].split('_')[0][1:])
        perv_idx = 0

        video_path = os.path.join(path, video_name)
        
        for idx, video_frame in enumerate(video_frames):

            if ((not (int(video_frame.split("_")[0][1:]) == scene_num)) or (idx == (len(video_frames) - 1))):
                scene_num += 1 
                scene_frames = [os.path.join(video_path, 'frames', frame_name) for frame_name in video_frames[perv_idx:idx]]
                virtual_camera(scene_frames)
                perv_idx = idx

