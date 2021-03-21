### python lib
import os, sys, random, math, cv2, pickle, subprocess, PIL
import numpy as np
# TODO-parsa :checkout PIL_SIMD currently it uses opencv
from PIL import Image
from datetime import datetime
import scipy.stats as st

### torch lib
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import torch.nn as nn

### custom library
from tools.resample2d_package.resample2d import Resample2d
from config import Config


####################### Time Utils #######################
def current_time_str():
    return datetime.now().strftime('%Y-%m-%d_%H-%M')

####################### Image Utility #######################

def img2tensor(img):

    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t

def tensor2img(img_t):

    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


def frame_loader_full_path(config, path):
    if (path.split('.')[1]== 'tiff'):
        frame = cv2.imread(path, cv2.IMREAD_ANYCOLOR| cv2.IMREAD_ANYDEPTH)
        frame = cv2.resize(frame, (config.frame_size[1], config.frame_size[0]))/(65535.0)
        frame = frame.astype(np.float32)
    elif(path.split('.')[1]=='png'):
        frame = cv2.imread(path)/(255.0)
        frame = cv2.resize(frame, (config.frame_size[1], config.frame_size[0]))
        frame = frame.astype(np.float32)

    return frame 


def sequence_loader_full_path(config, paths):
    sequence = []
    for path in paths:
        sequence.append(frame_loader_full_path(config, path))
    return sequence

def to_tensor(frame):
    frame = frame.transpose((2, 0, 1))
    tensor = torch.from_numpy(frame)
    return tensor

def seq_to_tensor(sequence):
    sequence_tensor = [torch.zeros(3, 360, 640)]* 6     #TODO: One Extra Frame for Temp
    for i, seq in enumerate(sequence):
        if(seq == []):
            continue
        seq = seq.transpose((2, 0, 1))
        seq_tensor = torch.from_numpy(seq)
        sequence_tensor[i] = seq_tensor
    sequence_tensor = torch.stack(sequence_tensor, 0).permute(1, 0, 2, 3)

    return sequence_tensor


####################### Flow Utility #######################

# https://github.com/phoenix104104/fast_blind_video_consistency/blob/master/utils.py

def flow_loader(config, path):
    
    
    with open(path, 'rb') as f:
        
        _ = np.fromfile(f, np.float32, count=1)
        
        w = int(np.fromfile(f, np.int32, count=1))
        h = int(np.fromfile(f, np.int32, count=1))
        #print 'Reading %d x %d flo file' % (w, h)
            
        data = np.fromfile(f, np.float32, count=2*w*h)

        # Reshape data into 3D array (columns, rows, bands)
        flow = np.resize(data, (2, h+40, w))

    return flow

def flow_sequence_loader(config, paths):
    flow_seq = list()
    for path in paths:
        flow_seq.append(flow_loader(config, path))
    return flow_seq


def compute_flow_magnitude(flow):

    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag

def compute_flow_gradients(flow):

    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))
    
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv

def warp(x, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    flow = flow.float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = torch.add(grid, flow)
    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask


def detect_occlusion(fw_flow, bw_flow):
    
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1

    
    with torch.no_grad():

        ## load on cuda
        fw_flow_c = fw_flow.to('cuda')
        bw_flow_c = bw_flow.to('cuda')

        ## warp fw-flow to img2
        flow_warping = Resample2d().cuda()
        fw_flow_w = flow_warping(fw_flow_c, bw_flow_c)
    
        ## convert to numpy array
        fw_flow_im = tensor2img(fw_flow_w)
        bw_flow_im = tensor2img(bw_flow_c)


    ## occlusion
    fb_flow_sum = fw_flow_im + bw_flow_im
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_im_mag = compute_flow_magnitude(fw_flow_im)
    bw_flow_mag = compute_flow_magnitude(bw_flow_im)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_im_mag + bw_flow_mag) + 0.5
    
    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow_im)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask    #TODO-mahdi: should be cheched with mask is better and then tune the combination mask
    mask = np.logical_or(mask1, mask2)
    occlusion = np.zeros((fw_flow_im.shape[0], fw_flow_im.shape[1]))
    occlusion[mask == 1] = 1

    return np.float32(occlusion)


def gaussian_kernel(length=11, nsig=3):
    x = np.linspace(-nsig, nsig, length+1)
    k1 = np.diff(st.norm.cdf(x))
    k2 = np.outer(k1, k1)
    norm = k2.sum()
    kernel = k2/norm
    kernel = kernel[np.newaxis, np.newaxis, ...]
    return np.float32(kernel)

def gauss_conv_gen():
    kernel = gaussian_kernel()
    conv = torch.nn.Conv2d(1, 1, 11, 1, 5, 1)   
    conv.weight = torch.nn.Parameter(torch.from_numpy(kernel))
    return conv


def count_parameters(model):
   return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":

    kernel = gaussian_kernel()

    # print(kernel)

    conv = torch.nn.Conv2d(1, 1, 11, 1, 5, 1)
    
    conv.weight = torch.nn.Parameter(torch.from_numpy(kernel))

    mask = torch.randn(1, 1, 360, 640)

    c = conv(mask)

    # print(c.size())

    gauss_conv = gauss_conv_gen()

    print(1 - gauss_conv.weight)