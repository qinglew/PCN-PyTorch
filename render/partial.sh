#!/bin/bash
echo "Begin to generate exr files"

for ((i=1; i<=21; i++)); do
    blender -b -P render_depth.py "/media/rico/BACKUP/Dataset/ShapeNetForPCN" "../dataset/car_split/split${i}.list" "/home/rico/Workspace/Dataset/partials/partial${i}" 8
done

echo "Done"
