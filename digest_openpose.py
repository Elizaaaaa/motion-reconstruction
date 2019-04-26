import deepdish as dd

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
    'Head',
]


all_frames = dd.io.load('/Users/eliza/Documents/motion-reconstruction/output/vault_bboxes.h5')
for frame in all_frames:
    for i, bbox, joints in all_frames[frame]:
        print('placeholder')

