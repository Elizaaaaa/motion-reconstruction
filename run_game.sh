rm ./data/*
gsutil cp gs://mimicus-videos/user-uploads/* ./data/

rm -rf ./output/*
python openpose.py
python reconstruction.py
python csvexporter.py
blender --background csv_to_bvh.blend -noaudio -P csv_to_bvh.py
