import torch
from torch.autograd import Variable
import sys
from utils import warp 
from tqdm import tqdm
from utils import warp, detect_occlusion, gauss_conv_gen

def validation_epoch(epoch, model, config, validation_loader, criterion):

    model.eval()

    loss_rec_sum = 0.0
    loss_tem_sum = 0.0
    loss_sum = 0.0
    loss_cnt = 0

    loss_msg = [0, 0]
    message = 'loss rec : {}, loss tem : {}'.format(loss_msg[0], loss_msg[1])
    t = tqdm(validation_loader, desc=message)

    # gauss_conv = gauss_conv_gen()
    # gauss_conv = gauss_conv.to(config.device)

    for _ , (inputs, target, temp, mask_rec, flows) in enumerate(t):
        

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
        mask_rec = torch.cat((mask_rec, mask_rec, mask_rec), dim=1)    
       
        output[output < 0] = 0.0
        target[target < 0] = 0.0

        output_ill = torch.mul(output, mask_rec)
        target_ill = torch.mul(target, mask_rec)


        if(config.loss == 'Rec&Tem'):
            flow_fw = flows[0]
            flow_fw = flow_fw.to(config.device)
            pass


        if(config.loss == 'Rec&Tem'):
            temp = temp.to(config.device)
            temp_warped = warp(temp, flow_fw)
            input_pre = inputs[:,:,1,:,:].to(config.device)
            input_pre_wrp =  warp(input_pre, flow_fw)
            mask = torch.exp(-50*torch.norm((input -
                            input_pre_wrp), dim=1, keepdim=True))
            mask = torch.cat((mask, mask, mask), dim = 1)
            temp_warped[temp_warped < 0] = 0.0
            temp_warped_ill = torch.mul(temp_warped, mask_rec)
            loss_tem = criterion(mask*output_ill, mask*temp_warped_ill)
            # loss_tem = criterion(mask*output_ill, mask*temp_warped_ill)


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


        loss_msg[0] =  loss_rec.item() 
        loss_msg[1] =  0
        if(config.loss == 'Rec&Tem'):
            loss_msg[1] = loss_tem.item()

        message = 'loss rec : {:.4f}, loss tem : {:.4f} '.format(loss_msg[0], loss_msg[1])

        t.set_description(message)            
    
    validation_loss = [loss_rec_sum/loss_cnt]
    if(config.loss == 'Rec&Tem'):
        validation_loss.append(loss_tem_sum/loss_cnt)
        validation_loss.append(loss_sum/loss_cnt)

    return validation_loss
    