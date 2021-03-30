import numpy as np 
import cv2
from torchvision import transforms
import random
from utils import gaussian_kernel, sequence_loader_full_path, frame_loader_full_path
from config import Config


class VirtualCamera(object):

    def __init__(self, config, action):
        
        clip = [0.05, 0.15] # Min and max percentage of intensities to clip

        self.n_mean, self.n_std             =  0.9, 0.1    # Sigmoidal parameter 'n' mean and std
        self.sigma_mean, self.sigma_std     =  0.6, 0.1    # Sigmoidal parameter 'sigma' mean and std                    
        self.scale_step = 0.01
        self.cl = np.random.uniform(low=clip[0], high=clip[1])
        self.scale = 1
        self.config = config
        self.action = action
    
    ## find scale parameter
    def _scale_find_single(self, img):

        scale = 1.0
        sz = img.shape
        num_pix = sz[0]*sz[1]*sz[2]

        sat0 =  np.sum(img > 1.0)/ num_pix
        sat_diff = 0

        while(sat_diff < self.cl):
            scale += self.scale_step
            img_sat = scale * img

            if (scale >= 2):
                break

            sat_diff =  np.sum(img_sat > 1.0)/ num_pix - sat0
        
        self.scale = scale

    
    def _scale_find_multiple(self, imgs):


        num_frames = len(imgs)
        
        sz = imgs[0].shape
        num_pix = sz[0]*sz[1]*sz[2]


        scale_acc = 0
        
        for idx in range(num_frames):

            sat0 =  np.sum(imgs[idx] > 1.0)/ num_pix

            scale = 1.0
            sat_diff = 0
            # while(sat_diff < self.cl):
            while(sat_diff < 0.05):
                scale += self.scale_step
                img_sat = scale * imgs[idx]

                if (scale >= 2.0):
                    break

                sat_diff =  np.sum(img_sat > 1.0)/ num_pix - sat0
            
            scale_acc += scale
        
        self.scale = scale_acc / num_frames


    # hdr to ldr conversion
    def __call__(self, hdr_frames):


        n       =  self.n_mean + self.n_std * np.random.randn(1)
        sigma   =  self.sigma_mean + self.sigma_std * np.random.randn(1)
        

        if(self.action == 'frame_based'):

            ldr_frames = []

            i = hdr_frames[3]
            
            self._scale_find_single(i)

            target_frame = None

            for id, frame in enumerate(hdr_frames):

                if(frame == []):
                    ldr_frames.append([])
                    continue

                frame *= self.scale 

                if(id == 3):
                    target_frame = frame
                    mask_sat = (np.max(frame, axis=2, keepdims=True) >= (1.0 - self.config.eps))
                    mask_sat = mask_sat.astype(np.float32)
                  

                frame[frame > 1.0] = 1.0     # extreme value saturation                

                frame_n = frame**n
                frame = (1.0 + sigma) * np.multiply(frame_n, 1.0/(frame_n + sigma))   # Apply camera curve
            
                ldr_frames.append(frame.astype(np.float32))
            
            return ldr_frames, target_frame, mask_sat

        elif (self.action == 'scene_based'):

                scene_len = len(hdr_frames)

                selected_indices = random.sample(range(scene_len), scene_len//30 + 1)

                selected_frames = sequence_loader_full_path(self.config , [hdr_frames[sel] for sel in selected_indices])

                self._scale_find_multiple(selected_frames)

                for frame_name in hdr_frames:

                    frame = frame_loader_full_path(self.config, frame_name)
                                        
                    frame = frame * self.scale    # scale values

                    mask = (np.max(frame, axis=2, keepdims=True) >= (1.0 - self.config.eps))
                    mask = mask.astype(np.float32)
                    mask = np.tile(mask, (1, 1, 3))

                    frame[frame > 1.0] = 1.0     # extreme value saturation                

                    frame_n = frame**n
                    frame = (1.0 + sigma) * np.multiply(frame_n, 1.0/(frame_n + sigma))   # Apply camera curve
                    
                    frame_mod = (frame*(2**8 - 1)).astype('uint8')                  
                    mask_mod  = (mask*(2**8 - 1)).astype('uint8') 

                    
                    cv2.imwrite(frame_name.split('.')[0].replace('frames', 'input') + '.png', frame_mod)
                    cv2.imwrite(frame_name.split('.')[0].replace('frames', 'mask') + '.png', mask_mod)
                    print(frame_name)
                    