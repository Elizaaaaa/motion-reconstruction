import deepdish as dd
import numpy as np
import pandas as pd

joints_names = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
                   'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
                   'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                   'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
                   'Knee.L_x', 'Knee.L_y', 'Knee.L_z',
                   'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
                   'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z',
                   'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z',
                   'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z',
                   'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
                   'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
                   'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z',
                   'Neck_x', 'Neck_y', 'Neck_z',
                   'Head_x', 'Head_y', 'Head_z',
                   'Nose_x', 'Nose_y', 'Nose_z',
                   'Eye.L_x', 'Eye.L_y', 'Eye.L_z',
                   'Eye.R_x', 'Eye.R_y', 'Eye.R_z',
                   'Ear.L_x', 'Ear.L_y', 'Ear.L_z',
                   'Ear.R_x', 'Ear.R_y', 'Ear.R_z']
# This is what we want.
joint_names = [
    'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
    'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
    'Head', 'Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear'
]
# Order of open pose
op_names = [
    'Nose',
    'Neck',
    'R Shoulder',
    'R Elbow',
    'R Wrist',
    'L Shoulder',
    'L Elbow',
    'L Wrist',
    'Mid Hip',
    'R Hip',
    'R Knee',
    'R Ankle',
    'L Hip',
    'L Knee',
    'L Ankle',
    'R Eye',
    'L Eye',
    'R Ear',
    'L Ear',
    'L BigToe',
    'L SmallToe',
    'L Heel',
    'R BigToe',
    'R SmallToe',
    'R Heel',
    'Background',
    'Head',
]
permute_order = np.array([op_names.index(name) for name in joint_names])
print(permute_order)

keypoints = []

all_frames = dd.io.load('/Users/eliza/Documents/motion-reconstruction/output/vault_bboxes.h5')
for frame in all_frames:
    for i, bbox, joints in all_frames[frame]:
        print(len(joints))
        joints = np.vstack((joints, np.zeros((1, 3))))
        print(len(joints))
        new_joints = joints[permute_order, :]
        keypoints.append(new_joints)

for keypoint in keypoints:
    joints_export = pd.DataFrame(keypoint.reshape(1, 57), columns=joints_names)
    joints_export.index.name = 'frame'
