import os
from reconstruction import digest_openpose_output

extensions_video = {".mov", ".mp4", ".avi", ".MOV", ".MP4"}
video_dir = './data/'
output_dir = './output/'
openpose_dir = '~/workspace/openpose/'

if not os.path.exists(output_dir+"jsons/"):
    os.mkdir(output_dir+"jsons/")
if not os.path.exists(output_dir+"images/"):
    os.mkdir(output_dir+"images/")

for filename in os.listdir(video_dir):
    for ext in extensions_video:
        if filename.endswith(ext):
            movement = os.path.splitext(filename)[0]
            print('processing the movement {}'.format(movement))
            filepath = video_dir+filename
            output_json_dir = output_dir + "jsons/" + movement + "/"
            output_image_dir = output_dir + "images/" + movement + "/"
            if os.path.exists(output_json_dir):
                print("jason output alread exists!")
                os.system('rm -rf {}'.format(output_json_dir+"/*"))
            else:
                os.mkdir(output_json_dir)
            if os.path.exists(output_image_dir):
                print("image output already exists!")
                os.system('rm -rf {}'.format(output_image_dir+"/*"))
            else:
                os.mkdir(output_image_dir)

            cmd_command = '~/workspace/openpose/build/examples/openpose/openpose.bin --video {0} --write_json {1} --write_images {2} --write_images_format jpg --model_folder ~/workspace/openpose/models --display 0'.format(filepath, output_json_dir, output_image_dir)
            print(cmd_command)

            run = os.system(cmd_command)
            print('reading the openpose output')
            digest_openpose_output(output_json_dir, filepath, movement)
            print('finish preparing the openpose data')
            print('############################')
