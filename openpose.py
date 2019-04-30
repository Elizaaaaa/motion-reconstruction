import os
from glob import glob
import deepdish as dd
import json
import numpy as np
import cv2

import scipy.signal as signal
import scipy.ndimage as ndimage


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
END_BOX_CONFIG = 0.1

KMaxLength = 1000
KVisThr = 0.2



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
    top_corner = center - radius
    bbox = np.hstack([top_corner, radius*2, radius*2])

    return np.hstack([center, scale, score, bbox]), keypoint

def select_bbox(bboxes, valid_keypoints):
    if len(bboxes) == 0:
        print('invalid input!')
        return [], []
    if bboxes.shape[0] == 1:
        #print('only one bbox in the frame')
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
    while len(idxs) > 0:
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)
        print('have picked: {}'.format(pick))
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        overlap = (w*h)/area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap>NMS_THRESH)[0])))

    return bboxes[pick], valid_keypoints[pick]

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

def fill_in_bboxes(bboxes, start_frame, end_frame):
    bboxes_filled = []
    boxid = 0
    for i in range(start_frame, end_frame):
        if bboxes[boxid][0] == i:
            bboxes_filled.append(bboxes[boxid][1:])
            boxid = boxid+1
        else:
            copybox = np.copy(bboxes_filled[-1])
            copybox[1][:, 2] = 0.
            bboxes_filled.append(copybox)

    return bboxes_filled


def params_to_bboxes(cx, cy, scale):
    center = [cx, cy]
    radius = RADIUS*(1/scale)
    top_corner = center-radius
    bbox = np.hstack([top_corner, radius*2, radius*2])

    return bbox


def smooth_detections(persons):
    per_frame = {}
    for personid in persons.keys():
        bboxes = persons[personid]
        start_frame = bboxes[0][0]
        end_frame = bboxes[-1][0]
        if len(bboxes) != (end_frame-start_frame):
            bboxeskeypoints_filled = fill_in_bboxes(bboxes, start_frame, end_frame)
        else:
            bboxeskeypoints_filled = [bbox[1:] for bbox in bboxes]
        #print(bboxeskeypoints_filled)
        bboxes_filled, keypoints_filled = [], []
        for bbox, keypoint in bboxeskeypoints_filled:
            bboxes_filled.append(bbox)
            keypoints_filled.append(keypoint)

        times = np.arange(start_frame, end_frame)
        if len(bboxes_filled) == 0:
            print('lack of bboxes')
            continue

        bboxes_filled = np.vstack(bboxes_filled)
        keypoints_filled = np.stack(keypoints_filled)
        bbox_params = bboxes_filled[:, :3]
        bbox_scores = bboxes_filled[:, 3]
        smoothed = np.array([signal.medfilt(param, 11) for param in bbox_params.T]).T
        smoothed_gaussian = np.array([ndimage.gaussian_filter(traj, 3) for traj in smoothed.T]).T

        smoothed_bboxes = np.vstack([params_to_bboxes(cx, cy, scale) for (cx,cy,scale) in smoothed_gaussian])
        last_index = len(bbox_scores)-1
        while bbox_scores[last_index] < END_BOX_CONFIG:
            if last_index <= 0:
                break
            last_index = last_index - 1

        final_bboxes = np.hstack([smoothed_gaussian[:last_index], bbox_scores.reshape(-1, 1)[:last_index], smoothed_bboxes[:last_index]])
        final_keypoints = keypoints_filled[:last_index]

        for time, bbox, keypoints in zip(times, final_bboxes, final_keypoints):
            if time in per_frame.keys():
                per_frame[time].append((personid, bbox, keypoints))
            else:
                per_frame[time] = [(personid, bbox, keypoints)]

    return per_frame


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



def clean_data(all_keypoints, video_path):
    persons = {}
    start_frame, end_frame = -1, 1
    for i, keypoints in enumerate(all_keypoints):
        if len(keypoints) == 0:
            print('{0} not having keypoints'.format(i))
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
            print('adding the first person')
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

            while not np.all(pid_is_matched) and not np.all(box_is_visited) and not np.all(iou_scores == -1):
                row, col = np.unravel_index(iou_scores.argmax(), (num_persons, num_bboxes))
                box_is_visited[col] = True

                if (iou_scores[row, col] > IOU_THRESH and not pid_is_matched[row] and not box_is_matched[col]):
                    persons[row].append((i, bboxes[col], valid_keypoints[col]))
                    pid_is_matched[row] = True
                    box_is_matched[col] = True

                iou_scores[row, :] = -1
                count += 1
                if count > 100:
                    print('infinite loop here')
                    break

            unmatched_boxes = bboxes[np.logical_not(box_is_matched)]
            unmatched_keypoints = valid_keypoints[np.logical_not(box_is_matched)]
            for tmp, (bbox, keypoint_using) in enumerate(zip(unmatched_boxes, unmatched_keypoints)):
                persons[num_persons + tmp] = [(i, bbox, keypoint_using)]

    #start cleaning
    print('before cleaning has: {}'.format(len(persons.keys())))
    frames = read_frames(video_path, 1)
    img_area = frames[0].shape[0]*frames[0].shape[1]
    duration = float(end_frame-start_frame)
    for personid in persons.keys():
        med_score = np.median([bbox[3] for (_, bbox, _) in persons[personid]])
        frequency = len(persons[personid])/duration
        print('{0} person frequency is {1}, length is {2} and duration is {3}'.format(personid, frequency, len(persons[personid]), duration))
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

    return per_frame_smooth


#read and store keypoints from json output
def digest_openpose_output(json_path, video_path, movement):

    #TODO: read all movements in output
    print('loading json files from {}'.format(json_path))
    all_json_paths = sorted(glob(os.path.join(json_path, "*.json")))
    all_keypoints = []
    for i, j in enumerate(all_json_paths):
        keypoints = read_json(j)
        all_keypoints.append(keypoints)
    per_frame_people = clean_data(all_keypoints, video_path)
    dd.io.save('./output/'+movement+"_bboxes.h5", per_frame_people)


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

            cmd_command = '~/workspace/openpose/build/examples/openpose/openpose.bin --net_resolution "1056x640" --scale_number 4 --scale_gap 0.25 --video {0} --write_json {1} --write_images {2} --write_images_format jpg --model_folder ~/workspace/openpose/models --display 0'.format(filepath, output_json_dir, output_image_dir)
            print(cmd_command)

            run = os.system(cmd_command)
            print('reading the openpose output')
            digest_openpose_output(output_json_dir, filepath, movement)
            print('finish preparing the openpose data')
            print('############################')
