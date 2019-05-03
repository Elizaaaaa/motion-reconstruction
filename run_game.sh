rm ./data/*
rm ./original_videos/*

#gsutil cp gs://mimicus-videos/user-uploads/* ./data/
#gdrive download --recursive 15uXxRZAiomTdDPp1T933nETZM_oBB7kC
gsutil cp gs://mimicus-videos/debug/45degree/* ./original_videos/
#sudo mv demo/videos/* ./original_videos/
#rm -rf demo
echo "Downloads finished."

echo "Resize the input videos"
for f in original_videos/*; do
    video_name=$(basename "$f")
    no_ext="${video_name%.*}"
    output_name="data/${no_ext}.mp4"
    echo "resize the $video_name and exprot to $output_name"
    ffmpeg -i $f -filter:v scale=720:-1 -c:a copy $output_name
done

rm -rf ./output/*
python openpose.py
python reconstruction.py
python csvexporter.py
blender --background csv_to_bvh.blend -noaudio -P csv_to_bvh.py
