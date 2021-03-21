
### python lib
import os, sys, random, math, cv2, pickle, subprocess, json, time
import numpy as np

### pytorch lib
import torch
import torch.utils.data as data
from torchvision import transforms

### utils lib
from utils import gaussian_kernel, sequence_loader_full_path, frame_loader_full_path, flow_loader, seq_to_tensor, to_tensor
from tools.VirtualCamera.virtual_camera import VirtualCamera

def get_data(config, mode, options=None):
    data = HDRVideoDataset(config, mode, options)
    return data


def gen_dataset(config, mode, dump_data=True, read_when_data_exist = False):
    """[summary]

    Args:
        config ([Config]): [description]
        mode: select one of 'train', 'validation', 'test' mode.
        dump_data (bool, optional): [Dump dataset list via pickle]. Defaults to True.
        read_when_data_exist (bool, optional): [load the data from already dumped one instead of regenerate it, the data set name should be hdrvideonet_dataset.pkl ]. Defaults to False.

    Returns:
        [type]: [description]
    """

    if config.debug:
        data = list(range(config.debug_num_samples))
        return data

    state = config.state

    if state=='Train':
        if mode=='train':
            path = os.path.join(config.root, 'train')
        elif mode=='validation':
            path = os.path.join(config.root, 'validation')
    else:
        path = os.path.join(config.root, 'inference')
 
    if read_when_data_exist:
        # print("Loading dataset from file")
        try:
            with open(os.path.join(config.root , config.fileName), 'rb') as file:
                data = pickle.load(file)
                return data
        except:
            # print("Something went wrong during loading file from file exiting the program ...")
            sys.exit(1)

    temporal_num = config.temporal_num 
    temp_mem = temporal_num //2 

    margin_mode = config.margin_mode 
    
    # This will hold the dictionary of dataset(containing full path relative to the current working directory)
    data = []
    videos_name = os.listdir(path)
    num_of_videos = len(videos_name)
    
    
    
    if(state == 'Train'):

        for idx_v, video_name in enumerate(videos_name): 
            # print("[{}/{}]Processing {} File".format(idx_v+1, num_of_videos, video_name))
            # Check if the corresponding data in HDR and Flow exist
            if not os.path.exists(os.path.join(path, video_name, 'frames')) and os.path.exists(os.path.join(path, video_name, 'flows', 'foreward')) : 
                # print("{} data are not Consistent, Moving to the Next Video".format(video_name))
                continue

            video_frames = sorted(os.listdir(os.path.join(path, video_name, 'frames')))
            video_flows_fw = []
            video_flows_bw = []
            if(config.loss == 'Rec&Tem'):
                video_flows_fw = sorted(os.listdir(os.path.join(path, video_name, 'flows', 'foreward')))
                video_flows_bw = sorted(os.listdir(os.path.join(path, video_name, 'flows', 'backward')))

            scene_num = int(video_frames[0].split('_')[0][1:])
            perv_idx = 0

            for idx, video_frame in enumerate(video_frames):  


                if (idx % config.down_samp != config.down_samp -1 ):
                    continue
                
                if not (int(video_frame.split("_")[0][1:]) == scene_num):   
                    scene_num += 1 
                    video_path = os.path.join(path, video_name) 
                    scene_len = idx// config.down_samp - perv_idx

                    
                    if margin_mode == "NONE":
                        if scene_len < (2*temp_mem+1 + 1): #TODO: One Extra Frame for Temp
                            # print("The scene {} doesn't have the sufficient frames in this padding mode".format(margin_mode))
                            continue
                        
                        for _, middle_frame in enumerate(range(temp_mem + 1 + perv_idx, scene_len-temp_mem +  perv_idx)): #TODO: One Extra Frame for Temp
                            # Fullpath of input frames
                            
                            input_frames = [os.path.join(video_path, 'frames', frame_name) for frame_name in video_frames[(config.down_samp*middle_frame-temp_mem - 1):(config.down_samp*middle_frame+temp_mem+1)]] #TODO: One Extra Frame for Temp
                            target_frame = os.path.join(video_path, 'frames' , video_frames[config.down_samp*middle_frame]) 
                            temp_frame = []
                            flows = []

                            if(config.loss == 'Rec&Tem'):
                                temp_frame   = os.path.join(video_path, 'frames' , video_frames[config.down_samp*middle_frame-1]) 
                                
                                # this include both forward and backward flows indexis should be done modulo 2
                                flow_fw = os.path.join(path, video_name, 'flows', 'foreward', video_flows_fw[config.down_samp*middle_frame-1])
                                flow_bw = os.path.join(path, video_name, 'flows', 'backward', video_flows_bw[config.down_samp*middle_frame])
                                
                                flows = [flow_fw, flow_bw]

                            data.append({'input_frames': input_frames,
                                        'target_frame': target_frame,
                                        'temp_frame':   temp_frame,
                                        'flows'       : flows
                                        })
                            
                    perv_idx = idx // config.down_samp
    
    elif(state == 'Inference'):
    
        for idx, video_name in enumerate(videos_name): 
            # print("[{}/{}]Processing {} File".format(idx+1, num_of_videos, video_name))
            # Check if the corresponding data in HDR and Flow exist
            if not os.path.exists(os.path.join(path, video_name, 'frames')) : 
                # print("{} data are not Consistent, Moving to the Next Video".format(video_name))
                continue

            video_frames = sorted(os.listdir(os.path.join(path, video_name, 'frames')))
            video_frames_input = sorted(os.listdir(os.path.join(path, video_name, 'input')))
            
            scene_num = int(video_frames[0].split('_')[0][1:])
            perv_idx = 0

            for idx, video_frame in enumerate(video_frames):  
                
                if not (int(video_frame.split("_")[0][1:]) == scene_num):   
                    scene_num += 1 
                    video_path = os.path.join(path, video_name) 
                    scene_len = idx- perv_idx

                    
                    if margin_mode == "NONE":
                        if scene_len < (2*temp_mem+1):
                            # print("The scene {} doesn't have the sufficient frames in this padding mode".format(scene_num))
                            continue
                        
                        for _, middle_frame in enumerate(range(temp_mem + perv_idx, scene_len-temp_mem +  perv_idx)):
                            # Fullpath of input frames
                            input_frames = [os.path.join(video_path, 'input', frame_name) for frame_name in video_frames_input[(middle_frame-temp_mem):(middle_frame+temp_mem+1)]]
                            mask_frames = [os.path.join(video_path, 'mask', frame_name) for frame_name in video_frames_input[(middle_frame-temp_mem):(middle_frame+temp_mem+1)]]
                            output_name = os.path.join(path, video_name, video_frames_input[middle_frame].split('.')[0] + '.tiff')
                            target_frame = os.path.join(video_path, 'frames', video_frames[middle_frame])
                            data.append({'input_frames': input_frames,
                                            'mask_frames': mask_frames,
                                            'output_name' : output_name,
                                            'target_frame': target_frame})
                            
                    perv_idx = idx
                
                elif(int(video_frames[0].split('_')[0][1:]) == int(video_frames[-1].split('_')[0][1:])):
                                            
                    scene_num += 1 
                    video_path = os.path.join(path, video_name) 
                    scene_len = len(video_frames)

                    
                    if margin_mode == "NONE":
                        if scene_len < (2*temp_mem+1):
                            print("The scene {} doesn't have the sufficient frames in this padding mode".format(scene_num))
                            continue
                        
                        for _, middle_frame in enumerate(range(temp_mem + perv_idx, scene_len-temp_mem +  perv_idx)):
                            # Fullpath of input frames
                            input_frames = [os.path.join(video_path, 'input', frame_name) for frame_name in video_frames_input[(middle_frame-temp_mem):(middle_frame+temp_mem+1)]]
                            mask_frames = [os.path.join(video_path, 'mask', frame_name) for frame_name in video_frames_input[(middle_frame-temp_mem):(middle_frame+temp_mem+1)]]
                            output_name = os.path.join(path, video_name, 'output', video_frames_input[middle_frame].split('.')[0] + '.tiff')
                            target_frame = os.path.join(video_path, 'frames', video_frames[middle_frame])
                            data.append({'input_frames': input_frames,
                                            'mask_frames': mask_frames,
                                            'output_name' : output_name,
                                            'target_frame': target_frame})
                            
                    perv_idx = idx

                    break
                    

           
    else:
        print('Error!')
        sys.exit(1)

    if dump_data:
        # print("Dumping the dataset to {}".format(config.fileName))
        with open(os.path.join(config.root, config.fileName), 'wb') as file:
            pickle.dump(data, file)
    
    return data

class HDRVideoDataset(data.Dataset):

    def __init__(self,config, mode, options):
        self.config = config
        self.data = gen_dataset(config, mode)
        self.transform_hdr2ldr = transforms.Compose([VirtualCamera(config, action='frame_based')])

    def __getitem__(self, index):
        state = self.config.state

        if state == 'Train':

            if not self.config.debug:
                input_frame_paths = self.data[index]['input_frames'] 
                # target_frame_path = self.data[index]['target_frame']
                
                # temp_frame_path = []
                flows_paths = []
                if(self.config.loss == 'Rec&Tem'):
                    # temp_frame_path = self.data[index]['temp_frame']
                    flows_paths = self.data[index]['flows']

                input_frames = [[]]*self.config.temporal_num + 1  #TODO: One Extra Frame for Temp
                if(self.config.model_shape == 'Single'):
                    input_frames[2] = frame_loader_full_path(self.config , input_frame_paths[3])  #TODO: One Extra Frame for Temp
                    if(self.config.loss == 'Rec&Tem'):
                        input_frames[1] = frame_loader_full_path(self.config , input_frame_paths[2])  #TODO: One Extra Frame for Temp

                else:
                    input_frames_pluse_one = sequence_loader_full_path(self.config , input_frame_paths)  #TODO: One Extra Frame for Temp

                flows_fw = []
                flows_bw = []
                if(self.config.loss == 'Rec&Tem'): 
                    flows_fw = flow_loader(self.config, flows_paths[0])
                    # flows_bw = flow_loader(self.config, flows_paths[1])

                # Convert hdr2ldr using VirtualCamera.
                input_frames_pluse_one, target_frame, temp_frame, mask = self.transform_hdr2ldr(input_frames_pluse_one)  #TODO: One Extra Frame for Temp

                input_frames = input_frames_pluse_one[1:6]
                if(self.config.loss == 'Rec&Tem'):   #TODO: One Extra Frame for Temp
                    input_frames_pre = input_frames_pluse_one[0:5]

                input_frames = seq_to_tensor(input_frames)
                
                input_frames_pre = []
                if(self.config.loss == 'Rec&Tem'):   #TODO: One Extra Frame for Temp
                    input_frames_pre = seq_to_tensor(input_frames_pre)

                target_frame = to_tensor(target_frame)

                # temp_frame = []
                flows = []
                if(self.config.loss == 'Rec&Tem'):
                    temp_frame = to_tensor(temp_frame)

                    flows = [flows_fw, flows_bw]

                mask = to_tensor(mask)

                return input_frames,  target_frame, temp_frame, mask, flows, input_frames_pre

            else:
                
                input_frames = torch.randn(self.config.in_channels, self.config.temporal_num, self.config.frame_size[0], self.config.frame_size[1])
                input_frames_pre = torch.randn(self.config.in_channels, self.config.temporal_num, self.config.frame_size[0], self.config.frame_size[1])
                target_frame = torch.randn(self.config.in_channels, self.config.frame_size[0], self.config.frame_size[1])
                temp_frame = torch.randn(self.config.in_channels, self.config.frame_size[0], self.config.frame_size[1])
                
                flow_fw  = torch.randn(2, self.config.frame_size[0], self.config.frame_size[1])
                flow_bw  = torch.randn(2, self.config.frame_size[0], self.config.frame_size[1])
                flows = [flow_fw, flow_bw] 

                mask = torch.randn(1, self.config.frame_size[0], self.config.frame_size[1])


            return input_frames, target_frame, temp_frame, mask, flows, input_frames_pre
        
        elif state == 'Inference':
            
            if not self.config.debug:
                input_frame_paths = self.data[index]['input_frames']
                mask_frame_paths = self.data[index]['mask_frames']
                target_frame_pathes = self.data[index]['target_frame']
                input_frames = sequence_loader_full_path(self.config , input_frame_paths)   
                mask_frames = sequence_loader_full_path(self.config , mask_frame_paths)              
                input_frames = seq_to_tensor(input_frames)
                mask_frames = seq_to_tensor(mask_frames)
                
                output_name = self.data[index]['output_name']

                target_frame = frame_loader_full_path(self.config , target_frame_pathes)
                target_frame = to_tensor(target_frame)
            else:
                input_frames = torch.randn(self.config.in_channels, self.config.temporal_num, self.config.frame_size[0], self.config.frame_size[1])
                mask_frames = torch.randn(self.config.in_channels, self.config.temporal_num, self.config.frame_size[0], self.config.frame_size[1])
                output_name  = os.path.join(self.config.root, 'inference', 'Video1', 'output', 'DEBUG.tiff')
                target_frame = torch.randn(self.config.in_channels, self.config.temporal_num, self.config.frame_size[0], self.config.frame_size[1])
            return input_frames, mask_frames, output_name, target_frame
            
            
        else:
            print('Error!')
            sys.exit(1)
        
    
    def __len__(self):
        return len(self.data)



def inference_dataset_prepare(config):

    path = os.path.join(config.root, 'inference')

    videos_name = os.listdir(path)
    num_of_videos = len(videos_name)

    virtual_camera = VirtualCamera(config, action='scene_based')

    for idx_v, video_name in enumerate(videos_name): 
        # print("[{}/{}]Processing {} File".format(idx_v+1, num_of_videos, video_name))
        if not os.path.exists(os.path.join(path, video_name, 'frames')): 
            # print("{} data are not Consistent, Moving to the Next Video".format(video_name))
            continue

        video_frames = sorted(os.listdir(os.path.join(path, video_name, 'frames')))

        if not os.path.exists(os.path.join(path, video_name, 'input')):
            os.makedirs(os.path.join(path, video_name, 'input'))

        if not os.path.exists(os.path.join(path, video_name, 'mask')):
            os.makedirs(os.path.join(path, video_name, 'mask'))
        
        if not os.path.exists(os.path.join(path, video_name, 'output')):
            os.makedirs(os.path.join(path, video_name, 'output'))

        scene_num = int(video_frames[0].split('_')[0][1:])
        perv_idx = 0
        
        scale = []
        
        for idx, video_frame in enumerate(video_frames):  

            if not (int(video_frame.split("_")[0][1:]) == scene_num):   
                scene_num += 1 
                video_path = os.path.join(path, video_name) 
                scene_len = idx - perv_idx
  
                scene_frames = [os.path.join(video_path, 'frames', frame_name) for frame_name in video_frames[perv_idx:scene_len+perv_idx]]

                virtual_camera(scene_frames)
                        
                perv_idx = idx

            
            elif(int(video_frames[0].split('_')[0][1:]) == int(video_frames[-1].split('_')[0][1:])):

                video_path = os.path.join(path, video_name) 
                scene_len = len(video_frames)
                scene_frames = [os.path.join(video_path, 'frames', frame_name) for frame_name in video_frames[perv_idx:scene_len+perv_idx]]

                virtual_camera(scene_frames)


                break
                
        







