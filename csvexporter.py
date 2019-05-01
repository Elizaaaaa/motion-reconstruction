import deepdish as dd
import pandas as pd
import os
import glob

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

path = './output/csv/'

if not os.path.exists(path):
    os.mkdir(path)

for filename in os.listdir('./output/refined/'):
    movement = os.path.basename(filename)
    movement = os.path.splitext(movement)[0]
    this_path = path+movement+"/"
    h5_path = './output/refined/'+filename
    print('loading h5 file {}'.format(h5_path))
    if not os.path.exists(this_path):
        os.mkdir(this_path)
    print('writing csv into {}'.format(this_path))
    all_frames = dd.io.load(h5_path)

    for frame in all_frames.keys():
        for item in all_frames[frame]:
            joints3d = item['joints3d']
            joints_export = pd.DataFrame(joints3d.reshape(1, 57), columns=joints_names)
            joints_export.index.name = 'frame'

            joints_export.iloc[:, 1::3] = joints_export.iloc[:, 1::3] * -1
            joints_export.iloc[:, 2::3] = joints_export.iloc[:, 2::3] * -1

            hipCenter = joints_export.loc[:][['Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                                              'Hip.L_x', 'Hip.L_y', 'Hip.L_z']]

            joints_export['hip.Center_x'] = hipCenter.iloc[0][::3].sum() / 2
            joints_export['hip.Center_y'] = hipCenter.iloc[0][1::3].sum() / 2
            joints_export['hip.Center_z'] = hipCenter.iloc[0][2::3].sum() / 2

            csv_id = str(frame)
            while csv_id.__len__() < 4:
                csv_id = "0"+csv_id

            joints_export.to_csv(this_path + csv_id + ".csv")

    all_files = glob.glob(os.path.join(this_path, "*.csv"))

    df_from_each_file = (pd.read_csv(f) for f in sorted(all_files))
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    concatenated_df['frame'] = concatenated_df.index + 1
    concatenated_df.to_csv(this_path + "csv_joined.csv", index=False)


