#!/bin/bash
# Example script to run BoxMOT tracking with Detectron2

# Set paths
VIDEO="/home/vuhai/Rehab-Tung/videos/GH010354_5_378_3750.avi"
CONFIG="/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
MODEL="/home/vuhai/Rehab-Tung/Detectron2DeepSortPlus/model_0004999.pth"
OUTPUT_DIR="./outputs"

# Create output directory
mkdir -p $OUTPUT_DIR

# Example 1: Single video with ByteTrack
echo "Running ByteTrack with Detectron2..."
python boxmot_tracking_detectron2.py \
    --input $VIDEO \
    --config-file $CONFIG \
    --model-weights $MODEL \
    --tracker bytetrack \
    --out_vid $OUTPUT_DIR/result_bytetrack.avi \
    --out_txt $OUTPUT_DIR/result_bytetrack.txt \
    --region_based 1 \
    --confidence-threshold 0.5

# Example 2: Test all trackers on the same video (organized by tracker)
echo "Testing all trackers with Detectron2 (organized by tracker)..."
python test_all_trackers.py \
    --input $VIDEO \
    --output-dir $OUTPUT_DIR/test_all \
    --config-file $CONFIG \
    --model-weights $MODEL \
    --trackers bytetrack ocsort botsort strongsort hybridsort sort deepsort

# Example 3: Batch process all videos in a directory with all trackers
# All results (txt and avi) in the same directory with naming: {video_name}_{tracker}
echo ""
echo "Batch processing all videos with all trackers..."
VIDEO_DIR="/home/vuhai/Rehab-Tung/videos"
python batch_all_trackers.py \
    --input-dir $VIDEO_DIR \
    --output-dir $OUTPUT_DIR/batch_all \
    --config-file $CONFIG \
    --model-weights $MODEL \
    --region_based 1 \
    --confidence-threshold 0.5

echo "Done!"
