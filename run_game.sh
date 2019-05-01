rm ./data/*
#gsutil cp gs://mimicus-videos/user-uploads/* ./data/
gdrive download --recursive 15uXxRZAiomTdDPp1T933nETZM_oBB7kC
sudo mv demo/videos/* ./data/
rm -rf demo
echo "Downloads finished."

echo "Resize the input videos"
for f in output/*; do
    video_name=$(basename "$f")
    no_ext="${video_name%.*}"
    output_name="${no_ext}.mp4"
    ffmpeg -i $video_name -filter:v scale=720:-1 -c:a copy $output_name
done

rm -rf ./output/*
python openpose.py
python reconstruction.py
python csvexporter.py
blender --background csv_to_bvh.blend -noaudio -P csv_to_bvh.py
