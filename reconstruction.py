#get data from openpose, then do the reconstruction

import os

import numpy as np
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


def run_video(frames, per_frame_people, config, output_path, ext, movement):
    print('run video with smpl and the output_path is {}'.format(output_path))

    proc_images, proc_keypoints, proc_params, start_frame, end_frame = collect_frames(
        frames, per_frame_people, config.img_size, KVisThr)

    num_frames = len(proc_images)
    print('after run video has {} images'.format(num_frames))
    proc_images = np.vstack(proc_images)
    
    result_path = './output/refined/'+movement+".h5"
    if not os.path.exists('./output/refined/'):
        os.mkdir('./output/refined/')

    #result_path = output_path.replace(ext, '.h5')
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
        dd.io.save(result_path, result_dict)


video_dir = './data/'
output_dir = './output/'

print('reading config')
config = get_config()
if 'model.ckpt' not in config.load_path:
    raise Exception('Must specify a model checkpoint!')

print('loading smpl model')
smpl = load_model(config.smpl_model_path)

np.random.seed(5)
video_paths = sorted(os.listdir(video_dir))
pred_dir = get_pred_pred_prefix(config.load_path)

for i, video_path in enumerate(video_paths):
    ext = os.path.splitext(video_path)[1]
    movement = os.path.splitext(video_path)[0]
    output_path = os.path.join(pred_dir, os.path.basename(video_path).replace(ext, '.h5'))

    if os.path.exists(output_path):
        print('output file exists!')
        os.remove(output_path)

    print('working on {}'.format(os.path.basename(video_dir+video_path)))
    frames, per_frame_people, valid = read_data(video_dir+video_path, './output/', max_length=KMaxLength)
    if valid:
        run_video(frames, per_frame_people, config, output_path, ext, movement)
    else:
        print('nothing valid')


