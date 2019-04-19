#get data from openpose, then do the reconstruction

import os
from glob import glob
import json

import numpy as np
import cv2

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

def read_frames(path, max_num=None):
    video = cv2.VideoCapture(path)

    images = []
    count = 0
    success = True
    while success:
        success, img = video.read()
        if success:
            # Make BGR->RGB!!!!
            images.append(img[:, :, ::-1])
            count += 1
            if max_num is not None and count >= max_num:
                break

    video.release()

    return images

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

def clean_data(all_keypoints, video_path):
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
            iou_scores = np.vstack(iou_scores)
            num_bboxes = len(bboxes)
            num_persons = len(persons.keys())
            box_is_matched = np.zeros(num_bboxes)
            box_is_visited = np.zeros(num_bboxes)
            pid_is_matched = np.zeros(num_persons)
            count = 0

            iou_scores_copy = np.copy(iou_scores)

            while not np.all(pid_is_matched) and not np.all(box_is_visited) and not np.all(iou_scores == -1):
                row, col = np.unravel_index(iou_scores.argmax(), (num_persons, num_bboxes))
                box_is_visited[col] = True

                if (iou_scores[row, col] > IOU_THRESH
                    and not pid_is_matched[row] and not box_is_matched[col]):
                    pid_is_matched[row] = True
                    box_is_matched[col] = True

                    iou_scores[row, :] = -1
                    count += 1
                    if count > 100:
                        print('infinite loop here')

            unmatched_boxes = bboxes[np.logical_not(box_is_matched)]
            unmatched_keypoints = valid_keypoints[np.logical_not(box_is_matched)]
            for tmp, (bbox, keypoint_using) in enumerate(zip(unmatched_boxes, unmatched_keypoints)):
                persons[num_persons + tmp] = [(i, bbox, keypoint_using)]

    #start cleaning
    frames = read_frames(video_path, 1)
    img_area = frames[0].shape[0]*frames[0].shape[1]
    duration = float(end_frame-start_frame)
    for personid in persons.keys():
        med_score = np.median([bbox[3] for (_, bbox, _) in persons[personid]])
        frequency = len(persons[personid])/duration
        med_bbox_area = np.median([bbox[6]*bbox[7] for (_, bbox, _) in persons[personid]]) / float(img_area)
        if frequency < FREQ_THRESH:
            print('frequency too low')
            del persons[personid]
            continue
        if med_score < SCORE_THRSH:
            print('score too low')
            del persons[personid]
            continue
    print('Total survived: {}'.format(len(persons.keys())))

    if len(persons.keys()) == 0:
        print('nothing survived')
        return {}

    per_frame_smooth = smooth_detections(persons)

#read and store keypoints from json output
def digest_openpose_output(json_path, video_path):

    #TODO: read all movements in output file
    #hardcode to vault for now

    json_path = json_path+"vault/"
    print(json_path)
    all_json_paths = sorted(glob(os.path.join(json_path, "*.json")))
    all_keypoints = []
    for i, j in enumerate(all_json_paths):
        keypoints = read_json(j)
        all_keypoints.append(keypoints)
    clean_data(all_keypoints, video_path)

#hardcode everything to vault first
video_dir = './data/vault.mp4'
output_dir = './output/'

cmd_command = '/Users/eliza/Documents/openpose/build/examples/openpose/openpose.bin --video ./data/vault.mp4 --model_folder /Users/eliza/Documents/openpose/models'
#only run once to get the output json and bbox.h5
#run = os.system(cmd_command)
print('reading the openpose output')
digest_openpose_output(output_dir, video_dir)

#TODO: read the openpose data, remove the irrelevant detections, only keep the main one
