#!/usr/bin/env python3.7
import argparse
import os 
import sys
import numpy as np 
import tqdm 
import subprocess

if __name__ == '__main__':
    extParse = argparse.ArgumentParser(description='This Script will be used for Video Frame Extraction to be used for HDRVideoNet')

    extParse.add_argument('--vid_path', 
                            type=str, 
                            help="Root Video Path", 
                            default="./")
    extParse.add_argument('--dst_img_path', 
                            type=str, 
                            help="This path will be used for the root of extracted images", 
                            default="./")
    extParse.add_argument('--frame_sample_rate', 
                            default=3,
                            type=int, 
                            help='Frame Sampling'
                            )

    extParse.add_argument("--out_format",
                            type=str,
                            help="The output format of the extracted images like png, tiff, jpg, ...",
                            default='png')

    extParse.add_argument("--out_width",
                            type=int,
                            help="Width of the extracted images",
                            default = 640
                            )
    extParse.add_argument("--out_height",
                            type=int,
                            help="Height of the extracted images",
                            default= 360)
    extParse.add_argument("-v", "--verbosity", action="count")

    extParse.add_argument("--pix_fmt",
                            type=str,
                            help="The output pixel format, run following command to see the supported formats:\nffmpeg -pix_fmts",
                            default='rgb48le')
    
    extParse.add_argument("--redo_scene_meta",
                            type=bool,
                            help="Redo the data extraction even if the files currently exist, note that the correspondance is only by name no hashing will be done...",
                            default=False)

    accept_ext = ['ts', 'mkv', 'mp4', 'TS']
    #### Don't Change this 
    root_meta_path=  "./"

    args = extParse.parse_args()    

    if not os.path.exists(args.vid_path): 
        print("Sorry The Path you entered is not valid")
        sys.exit(1)
    
    if not os.path.exists(args.dst_img_path):
        print("Making Destination Directory")
        os.mkdir(args.dst_img_path)

    src_dir_files = os.listdir(args.vid_path)
    acc_file_num = 0 
    for file in src_dir_files:
        os.rename(os.path.join(args.vid_path, file), os.path.join(args.vid_path, file.replace(" ", "_")))
        file = file.replace(" ", "_")
        csv_file = file + '.csv'

        if file.split('.')[-1] in accept_ext: 
            acc_file_num += 1 
            print("\n\n\n############ Processing File #{} ############\n\n\n".format(acc_file_num))
            file_full_path = os.path.join(args.vid_path, file)
            
            cur_scene_file_name = file.split('.')[0] + '-Scenes.csv'

            if (not os.path.exists(cur_scene_file_name)) or args.redo_scene_meta : 
                out_scene_detect = subprocess.check_output(["scenedetect", 
                                                "-i",
                                                os.path.join(args.vid_path, file) ,
                                                "-s",
                                                os.path.join(args.vid_path, csv_file),
                                                "detect-content",
                                                "list-scenes"]
                                                )
            else :
                print("\nEscapping Scene Detection for File {}\n".format(file))
            
            
            scene_data = np.genfromtxt( cur_scene_file_name, delimiter=',', skip_header=2)

            output_path = os.path.join(args.dst_img_path, file.split('.')[0])
            if not os.path.exists(output_path): 
                os.mkdir(output_path)
            
            for idx, scene in enumerate(scene_data): 
                print("\n\n #### Processing Scene#{} ####\n\n".format(idx+1))
                name_prefix = 'S' + '%03d'%(idx+1) + '_%04d'
                path_output_images = os.path.join(output_path, name_prefix)
                print(path_output_images)

                beg = int(scene_data[idx][1]) 
                end = int(scene_data[idx][4])
                print("Selecting From {} to {} For Scene Number {}".format(beg, end, idx+1))

                # Note the Chaining 
                p = subprocess.run(["ffmpeg",
				                "-loglevel",
			                	"warning",
                                "-i",
                                file_full_path,
                                "-vf",
                                "select='between(n \, %d \, %d)', select='not(mod(n \, %d))', scale=%d:%d, format=%s"%(int(beg), int(end), int(args.frame_sample_rate), int(args.out_width), int(args.out_height), args.pix_fmt),
                                "-vsync",
                                "0",
                                path_output_images + (args.out_format if args.out_format[0] == "." else "." + args.out_format) 
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

                print(str(p.stdout))
                print(str(p.stderr))
