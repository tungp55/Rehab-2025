#!/bin/bash

# Script chạy batch processing cho DeepSort và SORT trackers
# Input: /home/vuhai/Rehab-Tung/test_input1
# Output: 
#   - /home/vuhai/Rehab-Tung/test_output/deepsort
#   - /home/vuhai/Rehab-Tung/test_output/sort

SCRIPT_DIR="/home/vuhai/Rehab-Tung/BoxMOT_tracking_detectron2"
INPUT_DIR="/home/vuhai/Rehab-Tung/test_input1"

# Output directories
OUTPUT_DEEPSORT="/home/vuhai/Rehab-Tung/test_output/deepsort"
OUTPUT_SORT="/home/vuhai/Rehab-Tung/test_output/sort"

echo "=========================================="
echo "🚀 Starting batch processing for 2 trackers"
echo "=========================================="
echo "📁 Input directory: ${INPUT_DIR}"
echo "📁 Output DeepSort: ${OUTPUT_DEEPSORT}"
echo "📁 Output SORT: ${OUTPUT_SORT}"
echo ""

# Process với DeepSort
echo "=========================================="
echo "1️⃣  Processing with DeepSort tracker..."
echo "=========================================="
python3 "${SCRIPT_DIR}/batch_process.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DEEPSORT}" \
    --tracker "deepsort"

if [ $? -eq 0 ]; then
    echo "✅ DeepSort processing completed!"
else
    echo "❌ DeepSort processing failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "2️⃣  Processing with SORT tracker..."
echo "=========================================="

# Process với SORT
python3 "${SCRIPT_DIR}/batch_process.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_SORT}" \
    --tracker "sort"

if [ $? -eq 0 ]; then
    echo "✅ SORT processing completed!"
else
    echo "❌ SORT processing failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ All processing completed!"
echo "=========================================="
echo "📁 DeepSort results: ${OUTPUT_DEEPSORT}"
echo "   - Videos: ${OUTPUT_DEEPSORT}/videos"
echo "   - TXT files: ${OUTPUT_DEEPSORT}/txt"
echo ""
echo "📁 SORT results: ${OUTPUT_SORT}"
echo "   - Videos: ${OUTPUT_SORT}/videos"
echo "   - TXT files: ${OUTPUT_SORT}/txt"
