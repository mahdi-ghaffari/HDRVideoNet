# HDR Extractor
Image Extractor for all videos of the given path according to some resize factors and sampling rate, also equipped with scene distinguishing using the scenedetect library.
Checkout the requirements.txt for perquisites

# Sample 
`python3.7 ./ext.py --vid_path /mnt/db/HDR_Videos/test --dst_img_path /mnt/db/HDR_Videos/test_images --out_format tiff --frame_sample_rate 5 2>&1 | tee /tmp/Extractor.log`

# Notes 
In case your using the ssh, in order the code continue to run even though your ssh session closed, append the above command with `nohup` and add `& exit` at the end. note that the the `Extractor.log` file is meant to store the `stderr` and `stdout`. 


# Sample Script for Flow
Note the pix_fmt and out_format options.

`nohup python3.7 ./ext.py --vid_path /mnt/db/HDR_Videos/test --dst_img_path /mnt/db/HDR_Videos/flow_images --out_format png --out_width 640 --out_height 360 --frame_sample_rate 1 --pix_fmt rgb24 2>&1 | tee /tmp/Extractor_flow_images.log &`

