rm ./data/*
#gsutil cp gs://mimicus-videos/user-uploads/* ./data/
gdrive download --recursive 15uXxRZAiomTdDPp1T933nETZM_oBB7kC
sudo mv demo/videos/* ./data/
rm -rf demo
echo "Downloads finished."

rm -rf ./output/*
python openpose.py
python reconstruction.py
python csvexporter.py
blender --background csv_to_bvh.blend -noaudio -P csv_to_bvh.py
