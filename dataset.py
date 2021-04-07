### python lib
import os, sys, pickle

### pytorch lib
import torch
import torch.utils.data as data
from torchvision import transforms

### utils lib
from utils import sequence_loader_full_path, frame_loader_full_path, flow_loader, seq_to_tensor, to_tensor
from tools.VirtualCamera.virtual_camera import VirtualCamera

def get_data(config, mode, options=None):
    data = HDRVideoDataset(config, mode, options)
    return data

def load_dataset(config, mode, dump_data=True, read_when_data_exist=False):

    if config.debug:
        return [[]] * config.debug_num_samples

    state = config.state

    if state =='Train': 
        path = os.path.join(config.root, 'Train')
    else: #state == 'Inference'
        path = os.path.join(config.root, 'Inference')
 
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

    
    # This will hold the dictionary of dataset(containing full path relative to the current working directory)
    data = []
    videos_name_all = os.listdir(path)   
    num_of_videos  = len(videos_name_all)
    
    
    if(state == 'Train'):
        if(mode=='train'):        
            videos_name = videos_name_all[0:int(num_of_videos*(config.data_split_ratio))]    
        elif(mode=='validation'):                      
            videos_name = videos_name_all[int(num_of_videos*(config.data_split_ratio)):]

        for _, video_name in enumerate(videos_name):        

            video_frames = sorted(os.listdir(os.path.join(path, video_name, 'frames')))
            
            video_flows_fw = []
            video_flows_bw = []
            if(config.loss == 'Rec&Tem'):
                video_flows_fw = sorted(os.listdir(os.path.join(path, video_name, 'flows', 'foreward')))
                # video_flows_bw = sorted(os.listdir(os.path.join(path, video_name, 'flows', 'backward')))

            scene_num = int(video_frames[0].split('_')[0][1:])
            perv_idx = 0

            for idx, video_frame in enumerate(video_frames):  

                if ((not(int(video_frame.split("_")[0][1:]) == scene_num)) or (idx == (len(video_frames)-1))):   
                    scene_num += 1 
                    video_path = os.path.join(path, video_name) 
                    scene_len = (idx - perv_idx)                    
                                                
                    if scene_len < config.down_samp:
                        continue
                    
                    for _, middle_frame in enumerate(range(perv_idx + (config.down_samp//2) , idx, config.down_samp)): 
                        # Fullpath of input frames
                        input_frames = [os.path.join(video_path, 'frames', frame_name) for frame_name in video_frames[(middle_frame - (temp_mem + 1) ):(middle_frame  + (temp_mem + 1) )]]                               
                        flows = []

                        if(config.loss == 'Rec&Tem'):
                            # this include both forward and backward flows indexis should be done modulo 2
                            flow_fw = os.path.join(path, video_name, 'flows', 'foreward', video_flows_fw[middle_frame - 1])
                            flow_bw = []
                            # flow_bw = os.path.join(path, video_name, 'flows', 'backward', video_flows_bw[config.down_samp*middle_frame])
                            
                            flows = [flow_fw, flow_bw]

                        data.append({'input_frames': input_frames,
                                    'flows'       : flows
                                    })
                            
                    perv_idx = idx 
    
    elif(state == 'Inference'):

        for idx, video_name in enumerate(videos_name_all): 

            video_frames = sorted(os.listdir(os.path.join(path, video_name, 'frames')))
            video_frames_input = sorted(os.listdir(os.path.join(path, video_name, 'input')))
            
            video_flows_fw = []
            video_flows_bw = []
            if(config.loss == 'Rec&Tem'):
                video_flows_fw = sorted(os.listdir(os.path.join(path, video_name, 'flows', 'foreward')))
                # video_flows_bw = sorted(os.listdir(os.path.join(path, video_name, 'flows', 'backward')))

            scene_num = int(video_frames[0].split('_')[0][1:])
            perv_idx = 0
            
            video_path = os.path.join(path, video_name) 

            for idx, video_frame in enumerate(video_frames):  
                
                if ((not(int(video_frame.split("_")[0][1:]) == scene_num)) or (idx == (len(video_frames)-1))):   
                    scene_num += 1 
                    scene_len = idx- perv_idx                    
                   
                    if scene_len < (temporal_num + 1):
                        continue
                    
                    for _, middle_frame in enumerate(range(perv_idx + temp_mem + 1, idx-temp_mem)):
                        # Fullpath of input frames
                        input_frames = [os.path.join(video_path, 'input', frame_name) for frame_name in video_frames_input[(middle_frame-temp_mem-1):(middle_frame+temp_mem+1)]]
                        mask = os.path.join(video_path, 'mask', video_frames_input[middle_frame])
                        output_file_name  = os.path.join(path, video_name, 'output', video_frames_input[middle_frame].split('.')[0] + '.tiff')
                        target_frame = os.path.join(video_path, 'frames', video_frames[middle_frame])

                        flows = []
                        if(config.loss == 'Rec&Tem'):
                            # this include both forward and backward flows indexis should be done modulo 2
                            flow_fw = os.path.join(path, video_name, 'flows', 'foreward', video_flows_fw[middle_frame - 1])
                            flow_bw = []
                            # flow_bw = os.path.join(path, video_name, 'flows', 'backward', video_flows_bw[config.down_samp*middle_frame])
                            flows = [flow_fw, flow_bw]

                        data.append({'input_frames': input_frames,
                                     'mask'  : mask,
                                     'output_file_name' : output_file_name,
                                     'target_frame': target_frame,
                                     'flows'       : flows})
                            
                    perv_idx = idx
                  
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
        self.data = load_dataset(config, mode)
        self.transform_hdr2ldr = transforms.Compose([VirtualCamera(config, action='frame_based')])

    def __getitem__(self, index):
        state = self.config.state

        if state == 'Train':

            if not self.config.debug:
                input_frame_paths = self.data[index]['input_frames'] 
                
                flows_paths = []
                if(self.config.loss == 'Rec&Tem'):
                    flows_paths = self.data[index]['flows']

                input_frames_pluse_one = [[]]*(self.config.temporal_num + 1)  
                if(self.config.model_shape == 'Single'):
                    input_frames_pluse_one[3] = frame_loader_full_path(self.config , input_frame_paths[3])  
                    if(self.config.loss == 'Rec&Tem'):
                        input_frames_pluse_one[2] = frame_loader_full_path(self.config , input_frame_paths[2])  
                else:
                    input_frames_pluse_one = sequence_loader_full_path(self.config , input_frame_paths)  

                flows_fw = []
                flows_bw = []
                if(self.config.loss == 'Rec&Tem'): 
                    flows_fw = flow_loader(self.config, flows_paths[0])
                    # flows_bw = flow_loader(self.config, flows_paths[1])

                # Convert hdr2ldr using VirtualCamera.
                input_frames_pluse_one, target_frame, mask = self.transform_hdr2ldr(input_frames_pluse_one)

                input_frames = input_frames_pluse_one[1:6]
                input_frames = seq_to_tensor(input_frames)

                input_frames_pre = []
                if(self.config.loss == 'Rec&Tem'):   
                    input_frames_pre = input_frames_pluse_one[0:5]
                    input_frames_pre = seq_to_tensor(input_frames_pre)                    

                target_frame = to_tensor(target_frame)

                flows = []
                if(self.config.loss == 'Rec&Tem'):
                    flows = [flows_fw, flows_bw]

                mask = to_tensor(mask)

            else:
                
                input_frames = torch.randn(self.config.in_channels, self.config.temporal_num, self.config.frame_size[0], self.config.frame_size[1])
                target_frame = torch.randn(self.config.in_channels, self.config.frame_size[0], self.config.frame_size[1])
                
                mask = torch.randn(1, self.config.frame_size[0], self.config.frame_size[1])

                flow_fw  = torch.randn(2, self.config.frame_size[0], self.config.frame_size[1])
                flow_bw  = []
                flows = [flow_fw, flow_bw] 

                input_frames_pre = torch.randn(self.config.in_channels, self.config.temporal_num, self.config.frame_size[0], self.config.frame_size[1])                

            return input_frames, target_frame, mask, flows, input_frames_pre
        
        elif state == 'Inference':
            
            if not self.config.debug:
                input_frame_paths = self.data[index]['input_frames']
                mask_paths = self.data[index]['mask']
                target_frame_paths = self.data[index]['target_frame']
                flows_paths = self.data[index]['flows']
                
                input_frames_pluse_one = [[]]*(self.config.temporal_num + 1)
                if(self.config.model_shape == 'Single'):
                    input_frames_pluse_one[3] = frame_loader_full_path(self.config , input_frame_paths[3]) 
                    if(self.config.loss == 'Rec&Tem'):
                        input_frames_pluse_one[2] = frame_loader_full_path(self.config , input_frame_paths[2])
                else:
                    input_frames_pluse_one = sequence_loader_full_path(self.config , input_frame_paths)  

                input_frames = input_frames_pluse_one[1:6]
                input_frames = seq_to_tensor(input_frames)

                input_frames_pre = []
                if(self.config.loss == 'Rec&Tem'):   
                    input_frames_pre = input_frames_pluse_one[0:5]
                    input_frames_pre = seq_to_tensor(input_frames_pre)                    

                mask = frame_loader_full_path(self.config , mask_paths)
                mask = to_tensor(mask)
                
                output_file_name = self.data[index]['output_file_name']

                target_frame = frame_loader_full_path(self.config , target_frame_paths)
                target_frame = to_tensor(target_frame)

                flows = []
                if(self.config.loss == 'Rec&Tem'):
                    flows_fw = flow_loader(self.config, flows_paths[0])
                    # flows_bw = flow_loader(self.config, flows_paths[1])
                    flows = [flows_fw, []]

            else:
                input_frames = torch.randn(self.config.in_channels, self.config.temporal_num , self.config.frame_size[0], self.config.frame_size[1])
                input_frames_pre = torch.randn(self.config.in_channels, self.config.temporal_num , self.config.frame_size[0], self.config.frame_size[1])
                mask = torch.randn(self.config.in_channels , self.config.frame_size[0], self.config.frame_size[1])
                output_file_name  = os.path.join(self.config.root, 'Inference', 'Video1', 'output', 'DEBUG.tiff')
                target_frame = torch.randn(self.config.in_channels, self.config.temporal_num, self.config.frame_size[0], self.config.frame_size[1])
                
                flow_fw  = torch.randn(2, self.config.frame_size[0], self.config.frame_size[1])
                flow_bw  = []
                flows = [flow_fw, flow_bw]

            return input_frames, target_frame, mask, flows, input_frames_pre, output_file_name
            
        else:
            print('Error!')
            sys.exit(1)
    def __len__(self):
        return len(self.data)