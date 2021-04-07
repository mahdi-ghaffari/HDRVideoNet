import os 


def hdr_video_make(config):

    video_names = os.listdir(os.path.join(config.root, 'Inference'))

    frame_rename(config, 'output')

    for video in video_names:

        frames_path = os.path.join(config.root, 'Inference', video, 'output', 'frame%05d.exr')
        video_out_path = os.path.join(config.root, 'Inference', video, video + '.mkv')
        # os.system('ffmpeg  -i  {}  -c:v libx265 -vf "format=yuv420p10le" -vf "eq=saturation=1.5" {}'.format(frames_path, video_out_path)) 
        os.system('ffmpeg  -i  {}  -c:v libx265 -vf "format=yuv420p10le" {}'.format(frames_path, video_out_path)) 


def ldr_video_make(config):

    video_names = os.listdir(os.path.join(config.root, 'Inference'))
    
    frame_rename(config, 'input')
    frame_rename(config, 'mask')

    for video in video_names:

        frames_path = os.path.join(config.root, 'Inference', video, 'input', 'frame%05d.png')
        video_out_path = os.path.join(config.root, 'Inference', video, video + '.mp4')
        os.system('ffmpeg  -i  {}  -c:v libx264  {}'.format(frames_path, video_out_path)) 


def ground_truth_video(config):

    video_names = os.listdir(os.path.join(config.root, 'Inference'))

    frame_rename(config, 'frames')

    for video in video_names:

        frames_path = os.path.join(config.root, 'Inference', video, 'frames', 'S009_%04d.tiff')
        video_out_path = os.path.join(config.root, 'Inference', video, video + '-gt.mkv')
        os.system('ffmpeg  -i  {}  -c:v libx265 -vf "format=yuv420p10le" -vf "eq=saturation=1.5" {}'.format(frames_path, video_out_path)) 


def frame_rename(config, mode='input'):

    video_names = os.listdir(os.path.join(config.root, 'Inference'))

    if mode == 'output':

        for video in video_names:
   
            frames_names = sorted(os.listdir(os.path.join(config.root, 'Inference', video, 'output')))
            frame_num = 0      
            
            for frame in frames_names: 
                frame_num += 1
                os.rename(r'{}'.format(os.path.join(config.root, 'Inference', video, 'output', frame)), 
                        r'{}'.format(os.path.join(config.root, 'Inference', video, 'output', 'frame%05d.exr'%frame_num)))
    
    elif mode == 'input':

         for video in video_names:

            frames_names = sorted(os.listdir(os.path.join(config.root, 'Inference', video, 'input')))

            frame_num = 0
            
            for frame in frames_names: 
                frame_num += 1
                os.rename(r'{}'.format(os.path.join(config.root, 'Inference', video, 'input', frame)), 
                        r'{}'.format(os.path.join(config.root, 'Inference', video, 'input', 'frame%05d.png'%frame_num)))
    
    elif mode == 'mask':
                 
        for video in video_names:

            frames_names = sorted(os.listdir(os.path.join(config.root, 'Inference', video, 'mask')))

            frame_num = 0
                
            for frame in frames_names: 
                frame_num += 1
                os.rename(r'{}'.format(os.path.join(config.root, 'Inference', video, 'mask', frame)), 
                        r'{}'.format(os.path.join(config.root, 'Inference', video, 'mask', 'frame%05d.png'%frame_num)))

        