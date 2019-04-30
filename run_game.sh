rm ./data/*
gsutil cp gs://mimicus-videos/user-uploads/* ./data/

rm -rf ./output/*
python reconstruction.py
python csvexporter.py
blender blender --background csv_to_bvh.blend -noaudio -P csv_to_bvh.py
