#get data from openpose, then do the reconstruction

import os
from glob import glob
import json

import numpy as np

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    keypoints = []
    for people in data['people']:
        keypoint = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        keypoints.append(keypoint)
    return keypoints

def get_bbox(keypoint):
    return 0

def clean_data(all_keypoints):
    persons = {}
    start_frame, end_frame = -1, 1
    for i, keypoints in all_keypoints:
        if len(keypoints) == 0:
            continue
        valid_keypoints = []
        for keypoint in keypoints:
            print(keypoint[:, 2])
            #TODO: get the bbox and assign the people to bbox

#read and store keypoints from json output
def digest_openpose_output(json_path):
    all_json_paths = sorted(glob(os.path.join(json_path, "*.json")))
    all_keypoints = []
    for i, j in all_json_paths:
        keypoints = read_json(j)
        all_keypoints.append(keypoints)

video_dir = './data/*.mp4'
output_dir = './output'

cmd_command = '/Users/eliza/Documents/openpose/build/examples/openpose/openpose.bin --video ./data/vault.mp4 --model_folder /Users/eliza/Documents/openpose/models'
run = os.system(cmd_command)
#print(run)
