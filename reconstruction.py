#get data from openpose, then do the reconstruction

import os
from glob import glob
import json

import numpy as np

VISIBLE_THRESH = 0.1
NUM_VISIBLE_THRESH = 5
BOX_SIZE = 224
RADIUS = BOX_SIZE / 2.
NMS_THRESH = 0.5
OCCL_THRESH = 30
IOU_THRESH = 0.005
FREQ_THRESH = .1
SIZE_THRESH = .23
SCORE_THRSH = .4

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    keypoints = []
    for people in data['people']:
        keypoint = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        keypoints.append(keypoint)
    return keypoints

def get_bbox(keypoint):
    visible = keypoint[:, 2] > VISIBLE_THRESH
    visible_keypoint = keypoint[visible, :2]
    min_pt = np.min(visible_keypoint, axis=0)
    max_pt = np.max(visible_keypoint, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad input')
    center = (min_pt+max_pt) / 2.
    scale = 150. / person_height

    score = np.sum(keypoint[visible, 2]) / np.sum(visible)

    radius = RADIUS * (1/scale)
    top_corner= center - radius
    bbox = np.hstack([top_corner, radius*2, radius*2])

    return np.hstack([center, scale, score, bbox]), keypoint

def select_bbox(bboxes, valid_keypoints):
    if len(bboxes) == 0:
        print('invalid input!')
        return [], []
    if bboxes.shape[0] == 1:
        #only one bbox in the frame
        return bboxes, valid_keypoints

    pick = []
    scores = bboxes[:, 3]
    bboxes_shape = bboxes[:, 4:]
    x1 = bboxes_shape[:, 0]
    y1 = bboxes_shape[:, 1]
    x2 = x1 + bboxes_shape[:, 2] - 1
    y2 = y1 + bboxes_shape[:, 3] - 1
    area = bboxes_shape[:, 2] * bboxes_shape[:, 3]

    idxs = np.argsort(scores)
    print(idxs)

def compute_iou(bbox, bboxes):

    def iou(boxA, boxB):
        A_area = boxA[2] * boxA[3]
        B_area = boxB[2] * boxB[3]
        min_x = max(boxA[0], boxB[0])
        min_y = max(boxA[1], boxB[1])
        endA = boxA[:2] + boxA[2:]
        endB = boxB[:2] + boxB[2:]
        max_x = min(endA[0], endB[0])
        max_y = min(endA[1], endB[1])
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        inter_area = float(w * h)
        iou = max(0, inter_area / (A_area+B_area-inter_area))
        return iou

    return [iou(bbox[-4:], b[-4:]) for b in bboxes]

def clean_data(all_keypoints):
    persons = {}
    start_frame, end_frame = -1, 1
    for i, keypoints in enumerate(all_keypoints):
        if len(keypoints) == 0:
            continue
        valid_keypoints = []
        bboxes = []
        for keypoint in keypoints:
            bbox, keypoint_using = get_bbox(keypoint)
            if bbox is not None:
                bboxes.append(bbox)
                valid_keypoints.append(keypoint_using)

        if len(bboxes) == 0:
            print('Not found qualified bbox!')
            continue
        bboxes = np.vstack(bboxes)
        valid_keypoints = np.stack(valid_keypoints)
        bboxes, valid_keypoints = select_bbox(bboxes, valid_keypoints)

        #adding person
        if len(persons.keys()) == 0:
            start_frame = i
            for j, (bbox, valid_keypoint) in enumerate(zip(bboxes, valid_keypoints)):
                persons[j] = [(i, bbox, valid_keypoint)]
        else:
            end_frame = i
            iou_scores = []
            for pid, pbboxes in persons.items():
                last_time, last_bbox, last_keypoint = pbboxes[-1]
                if (i-last_time) > OCCL_THRESH:
                    ious = -np.ones(len(bboxes))
                else:
                    ious = compute_iou(last_bbox, bboxes)
                iou_scores.append(ious)

#read and store keypoints from json output
def digest_openpose_output(json_path):

    #TODO: read all movements in output file
    #hardcode to vault for now

    json_path = json_path+"vault/"
    print(json_path)
    all_json_paths = sorted(glob(os.path.join(json_path, "*.json")))
    all_keypoints = []
    for i, j in enumerate(all_json_paths):
        keypoints = read_json(j)
        all_keypoints.append(keypoints)
    clean_data(all_keypoints)

video_dir = './data/*.mp4'
output_dir = './output/'

cmd_command = '/Users/eliza/Documents/openpose/build/examples/openpose/openpose.bin --video ./data/vault.mp4 --model_folder /Users/eliza/Documents/openpose/models'
#only run once to get the output json and bbox.h5
#run = os.system(cmd_command)
print('reading the openpose output')
digest_openpose_output(output_dir)

#TODO: read the openpose data, remove the irrelevant detections, only keep the main one
