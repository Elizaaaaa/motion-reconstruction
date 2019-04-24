import csv
import os
import bpy
import sys


path = './output/csv/'
move_types = os.listdir(path)
for move in move_types:
    objects = bpy.context.scene.objects

    empties = []

    for object in objects:
        if object.type == 'EMPTY':
            empties.append(object)

    print(empties)

    filename = 'csv_joined.csv'
    directory = r'./output/csv/'

    fullpath = os.path.join(directory, filename)
    total_frame = 0

    with open(fullpath, 'r', newline='') as csvfile:
        ofile = csv.reader(csvfile, delimiter=',')
        next(ofile)  # <-- skip the x,y,z header

        for line in ofile:
            f, *pts = line

            # these things are still strings (that's how they get stored in the file)
            # here we recast them to integer and floats
            frame_num = int(f)
            fpts = [float(p) for p in pts]
            print(fpts)
            coordinates = [fpts[0:3], fpts[3:6], fpts[6:9], fpts[9:12],
                           fpts[12:15], fpts[15:18], fpts[18:21], fpts[21:24],
                           fpts[24:27], fpts[27:30], fpts[30:33], fpts[33:36],
                           fpts[36:39], fpts[39:42], fpts[42:45], fpts[45:48],
                           fpts[48:51], fpts[51:54], fpts[54:57], fpts[57:60]]
            total_frame = frame_num
            bpy.context.scene.frame_set(frame_num)
            for ob, position in zip(empties, coordinates):
                ob.location = position
                ob.keyframe_insert(data_path="location", index=-1)

    bpy.data.objects['rig'].select = True
    target_file = './output/bvh_animation/' + 'output.bvh'

    bpy.ops.export_anim.bvh(filepath=target_file, frame_start=1, frame_end=total_frame)

