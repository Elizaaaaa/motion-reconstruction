import os

extensions_video = {".mov", ".mp4", ".avi"}
video_dir = './data/'
output_dir = './output/'

for filename in os.listdir(video_dir):
    for ext in extensions_video:
        if filename.endswith(ext):
            filepath = video_dir+filename
            cmd_command = '/Users/eliza/Documents/openpose/build/examples/openpose/openpose.bin --video {0} --write_json {1} --write_images {2} --write_images_format jpg --model_folder /Users/eliza/Documents/openpose/models'.format(filepath, output_dir, output_dir)
            print(cmd_command)