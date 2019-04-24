#get data from openpose, then do the reconstruction

import os
from glob import glob
import json

import numpy as np
import cv2
import scipy.signal as signal
import scipy.ndimage as ndimage
import deepdish as dd
import tensorflow as tf

from src.config import get_config
from src.util.video import read_data, collect_frames
from src.mimicus_refiner import Refiner

from smpl_webuser.serialization import load_model

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
        bboxes_filled, keypoints_filled = [], []
        for bbox, keypoint in bboxeskeypoints_filled:
            bboxes_filled.append(bbox)
            keypoints_filled.append(keypoint)

        times = np.arange(start_frame, end_frame)
        if len(bboxes_filled) == 0:
            print('lack of bboxes')
            continue

        bboxes_filled = np.vstack(bboxes_filled)
        keypoints_filled = np.vstack(keypoints_filled)
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

        print('hows valid keypoints? {}'.format(valid_keypoints))

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

                if (iou_scores[row, col] > IOU_THRESH
                    and not pid_is_matched[row] and not box_is_matched[col]):
                    persons[row].append((i, bboxes[col], valid_keypoints[col]))
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
    per_frame = {}
    for personid in persons.keys():
        for time, bbox, keypoint_using in persons[personid]:
            if time in per_frame.keys():
                per_frame[time].append((personid, bbox, keypoint_using))
            else:
                per_frame[time] = [(personid, bbox, keypoint_using)]

    return per_frame

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
    per_frame_people = clean_data(all_keypoints, video_path)
    #hardcode to vault
    dd.io.save('./output/vault_bboxes.h5', per_frame_people)

def get_pred_pred_prefix(load_path):
    checkpt_name = os.path.basename(load_path)
    model_name = os.path.basename(os.path.dirname(config.load_path))
    print("ckpt is {0}, modelname is {1}".format(checkpt_name, model_name))

    prefix = []

    if config.refine_inpose:
        prefix += ['OptPose']

    prefix += ['kpw%.2f' % config.e_loss_weight]
    prefix += ['shapew%.2f' % config.shape_loss_weight]
    prefix += ['jointw%.2f' % config.joint_smooth_weight]
    if config.use_weighted_init_pose:
        prefix += ['init-posew%.2f-weighted' % config.init_pose_loss_weight]
    else:
        prefix += ['init-posew%.2f' % config.init_pose_loss_weight]
    if config.camera_smooth_weight > 0:
        prefix += ['camw%.2f' % config.camera_smooth_weight]

    prefix += ['numitr%d' % config.num_refine]

    prefix = '_'.join(prefix)
    if 'Feb12_2100' not in model_name:
        pred_dir = os.path.join(config.out_dir, model_name + '-' + checkpt_name, prefix)
    else:
        if prefix == '':
            save_prefix = checkpt_name
        else:
            save_prefix = prefix + '_' + checkpt_name

        pred_dir = os.path.join(config.out_dir, save_prefix)

    print('\n***\nsaving output in %s\n***\n' % pred_dir)

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    return pred_dir


def run_video(frames, per_frame_people, config, output_path):
    print('run video with smpl')

    proc_images, proc_keypoints, proc_params, start_frame, end_frame = collect_frames(
        frames, per_frame_people, config.img_size, KVisThr)

    num_frames = len(proc_images)
    print('after run video has {} images'.format(num_frames))
    proc_images = np.vstack(proc_images)
    result_path = output_path.replace('.mp4', '.h5')
    if not os.path.exists(result_path):
        tf.reset_default_graph()
        #TODO: replace the refiner with our version, now try to only abstract joint information
        model = Refiner(config, num_frames)

        scale_factors = [np.mean(pp['scale']) for pp in proc_params]
        offsets = np.vstack([pp['start_pt'] for pp in proc_params])
        results = model.predict(proc_images, proc_keypoints, scale_factors, offsets)
        results['proc_params'] = proc_params

        result_dict = {}
        used_frames = frames[start_frame:end_frame + 1]
        for i, (frame, proc_param) in enumerate(zip(used_frames, proc_params)):
            op_kp = proc_param['op_kp']

            theta = results['theta'][i]
            pose = theta[3:3+72]
            shape = theta[3+72:]
            smpl.trans[:] = 0.
            smpl.betas[:] = shape
            smpl.pose[:] = pose

            result = {
                'theta': np.expand_dims(theta, 0),
                'joints': np.expand_dims(results['joints'][i], 0),
                'cams': results['cams'][i],
                'joints3d': results['joints3d'][i],
                'op_kp': op_kp,
                'proc_param': proc_param
            }
            result_dict[i] = [result]

        print('writing into {}'.format(result_path))
        dd.io.save('temp.h5', result_dict)
        write_to_csv(result_dict)



#hardcode everything to vault first
video_dir = './data/vault.mp4'
output_dir = './output/'

#TODO: output path is not correct
cmd_command = '/Users/eliza/Documents/openpose/build/examples/openpose/openpose.bin --video ./data/vault.mp4 --write_json ./data/output/ --write_images ./data/output/ --write_images_format jpg --model_folder /Users/eliza/Documents/openpose/models'
#only run once to get the output json
#run = os.system(cmd_command)
print('reading the openpose output')
digest_openpose_output(output_dir, video_dir)
print('finish preparing the openpose data')
print('############################')

print('reading config')
config = get_config()
if 'model.ckpt' not in config.load_path:
    raise Exception('Must specify a model checkpoint!')

print('loading smpl model')
smpl = load_model(config.smpl_model_path)

np.random.seed(5)
video_paths = sorted(glob(os.path.join(config.video_dir, "*.mp4")))
pred_dir = get_pred_pred_prefix(config.load_path)

for i, video_path in enumerate(video_paths):
    print('working on {}'.format(video_path))

    output_path = os.path.join(pred_dir, os.path.basename(video_path).replace('.mp4', '.h5'))
    #TODO: run it even exists
    if os.path.exists(output_path):
        print('output file exists!')
        os.remove(output_path)

    print('working on {}'.format(os.path.basename(video_path)))
    frames, per_frame_people, valid = read_data(video_path, './output/', max_length=KMaxLength)
    if valid:
        run_video(frames, per_frame_people, config, output_path)
    else:
        print('nothing valid')


