### python lib
from tqdm import tqdm

### pytorch lib
import torch

### custom lib
from utils import warp

def validation_epoch(model, config, validation_loader, criterion):

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

    for _ , (inputs, target, mask_rec, flows, inputs_pre) in enumerate(t):
        

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

            temp = inputs[:,:,1,:,:].to(config.device)
            temp =  warp(temp, flow_fw)
            mask = torch.exp(-50*torch.norm((input -
                            temp), dim=1, keepdim=True))
            mask_tem = torch.cat((mask, mask, mask), dim = 1)

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

        loss_msg[0] =  loss_rec.item() 
        loss_msg[1] =  0
        if(config.loss == 'Rec&Tem'):
            loss_msg[1] = loss_tem.item()

    
        loss_rec_sum += loss_rec.item()
        if(config.loss == 'Rec&Tem'):
            loss_tem_sum += loss_tem.item()
        loss_sum     += loss.item()
        loss_cnt     += 1

        message = 'loss rec : {:.4f}, loss tem : {:.4f} '.format(loss_msg[0], loss_msg[1])

        t.set_description(message)
           
    
    validation_loss = [loss_rec_sum/loss_cnt]
    if(config.loss == 'Rec&Tem'):
        validation_loss.append(loss_tem_sum/loss_cnt)
        validation_loss.append(loss_sum/loss_cnt)

    return validation_loss    