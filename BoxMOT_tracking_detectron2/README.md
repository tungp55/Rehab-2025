# BoxMOT Tracking với Detectron2 Segmentation Models

Hệ thống tracking sử dụng **Detectron2 segmentation models** kết hợp với **BoxMOT trackers** - các thuật toán tracking tiên tiến: ByteTrack, OcSort, BotSort, StrongSort, HybridSort, SORT, và DeepSort.

## 📋 Tính năng

- ✅ Sử dụng **Detectron2 Mask R-CNN** segmentation models
- ✅ Hỗ trợ nhiều thuật toán tracking từ BoxMOT
- ✅ Tích hợp tương tự `dt2ds.py` nhưng với BoxMOT trackers
- ✅ Xử lý single video hoặc batch processing (thư mục)
- ✅ Export video tracking và file txt (MOT format)
- ✅ Region-based tracking (chỉ track trên vùng segmented)
- ✅ Test tất cả trackers trên cùng video để so sánh

## 🚀 Cài đặt

```bash
cd /home/vuhai/Rehab-Tung/BoxMOT_tracking_detectron2
pip install -r requirements.txt
```

## 📖 Sử dụng

### 1. Single Video Tracking

```bash
python boxmot_tracking_detectron2.py \
    --input /path/to/video.avi \
    --config-file /path/to/detectron2_config.yaml \
    --model-weights /path/to/model_weights.pth \
    --tracker bytetrack \
    --out_vid output_video.avi \
    --out_txt output_tracking.txt \
    --region_based 1
```

### 2. Batch Processing (Thư mục)

```bash
python batch_process.py \
    --input-dir /path/to/video/folder \
    --output-dir /path/to/output/folder \
    --config-file /path/to/config.yaml \
    --model-weights /path/to/model.pth \
    --tracker bytetrack
```

### 3. Test Tất cả Trackers (1 video, organized by tracker)

```bash
python test_all_trackers.py \
    --input /path/to/video.avi \
    --output-dir ./test_results \
    --config-file /path/to/config.yaml \
    --model-weights /path/to/model.pth \
    --trackers bytetrack ocsort botsort strongsort hybridsort sort deepsort
```

### 4. Batch Process Tất cả Videos với Tất cả Trackers

**Mỗi tracker có thư mục riêng với txt/ và videos/ subdirectories**

```bash
python batch_all_trackers.py \
    --input-dir /path/to/video/folder \
    --output-dir /path/to/output/folder \
    --config-file /path/to/config.yaml \
    --model-weights /path/to/model.pth
```

**Output structure:**
```
output_dir/
├── bytetrack/
│   ├── txt/
│   │   ├── video1_bytetrack.txt
│   │   ├── video2_bytetrack.txt
│   │   └── ...
│   └── videos/
│       ├── video1_bytetrack.avi
│       ├── video2_bytetrack.avi
│       └── ...
├── ocsort/
│   ├── txt/
│   │   ├── video1_ocsort.txt
│   │   └── ...
│   └── videos/
│       ├── video1_ocsort.avi
│       └── ...
├── botsort/
│   ├── txt/
│   └── videos/
├── strongsort/
│   ├── txt/
│   └── videos/
├── hybridsort/
│   ├── txt/
│   └── videos/
├── sort/
│   ├── txt/
│   └── videos/
└── deepsort/
    ├── txt/
    └── videos/
```

## 🎯 Các Thuật toán Tracking

| Tracker | Mô tả | BoxMOT |
|---------|-------|--------|
| **ByteTrack** | Multi-object tracking với association strategy | ✅ |
| **OcSort** | Occlusion-aware tracking | ✅ |
| **BotSort** | Boosting tracking với appearance features | ✅ |
| **StrongSort** | Strong association tracking | ✅ |
| **HybridSort** | Hybrid tracking approach | ✅ |
| **SORT** | Simple Online and Realtime Tracking | ⚠️ Fallback |
| **DeepSort** | Deep learning based tracking | ⚠️ Fallback |

## 📝 Arguments

### boxmot_tracking_detectron2.py

- `--input`: Đường dẫn đến video input (required)
- `--config-file`: Đường dẫn đến Detectron2 config file (.yaml)
- `--model-weights`: Đường dẫn đến Detectron2 model weights (.pth)
- `--num-classes`: Số lượng classes (default: 2 cho hand tracking)
- `--tracker`: Loại tracker (`bytetrack`, `ocsort`, `botsort`, `strongsort`, `hybridsort`, `sort`, `deepsort`)
- `--confidence-threshold`: Ngưỡng confidence (default: 0.5)
- `--region_based`: 1 để track trên vùng segmented, 0 cho full image
- `--out_vid`: Đường dẫn output video
- `--out_txt`: Đường dẫn output text file (MOT format)
- `--device`: Device (`cuda` hoặc `cpu`)
- `--fps`: FPS của output video

### batch_process.py

- `--input-dir`: Thư mục chứa videos
- `--output-dir`: Thư mục output
- `--config-file`: Detectron2 config file
- `--model-weights`: Detectron2 model weights
- Các arguments khác tương tự `boxmot_tracking_detectron2.py`

### test_all_trackers.py

- `--input`: Video để test
- `--output-dir`: Thư mục output (organized by tracker)
- `--config-file`: Detectron2 config file
- `--model-weights`: Detectron2 model weights
- `--trackers`: Danh sách trackers để test (space-separated)
- Các arguments khác tương tự `boxmot_tracking_detectron2.py`

### batch_all_trackers.py

- `--input-dir`: Thư mục chứa videos
- `--output-dir`: Thư mục output (mỗi tracker có thư mục riêng: {tracker}/txt/ và {tracker}/videos/)
- `--config-file`: Detectron2 config file
- `--model-weights`: Detectron2 model weights
- `--reid-weights`: (Optional) ReID weights cho strongsort/botsort/hybridsort. Nếu không có sẽ auto-detect từ ./reID_weight/
- `--verbose`: Hiển thị chi tiết lỗi
- Các arguments khác tương tự `boxmot_tracking_detectron2.py`

## 📊 Output Format

### Video Output
- Format: AVI (MJPG codec)
- Hiển thị: Bounding boxes, track IDs, trajectories, masks (nếu có)

### Text Output (MOT Format)
```
frame_id,track_id,x,y,width,height,conf,x,y,z
1,1,100,200,50,80,1,1,1
1,2,300,150,60,90,1,1,1
2,1,105,205,50,80,1,1,1
...
```

## 🔧 Default Paths

- **Config file**: `/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`
- **Model weights**: `/home/vuhai/Rehab-Tung/Detectron2DeepSortPlus/model_0004999.pth`
- **DeepSort checkpoint**: `/home/vuhai/Rehab-Tung/Detectron2DeepSortPlus/deep_sort/deep/checkpoint/ckpt.t7`

## 🔧 Troubleshooting

### BoxMOT không cài đặt được

Nếu BoxMOT không cài đặt được, hệ thống sẽ tự động fallback về SORT/DeepSort từ thư mục `Detectron2DeepSortPlus`.

### Detectron2 không tìm thấy

Đảm bảo Detectron2 được cài đặt và đường dẫn đúng:
```bash
export PYTHONPATH=/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2:$PYTHONPATH
```

### GPU không hoạt động

Đảm bảo CUDA được cài đặt và set `--device cuda`. Nếu không có GPU, sử dụng `--device cpu`.

## 📁 Cấu trúc Thư mục

```
BoxMOT_tracking_detectron2/
├── boxmot_tracking_detectron2.py  # Main tracking script
├── detector_detectron2.py         # Detectron2 detector wrapper
├── batch_process.py               # Batch processing (single tracker)
├── batch_all_trackers.py          # Batch processing với tất cả trackers
├── test_all_trackers.py           # Test all trackers (1 video, organized by tracker)
├── convert_reid_weights.py        # Convert ReID weights .pth to .pt
├── reID_weight/                   # ReID weights directory
│   ├── osnet_x1_0_imagenet.pt
│   ├── mobilenetv2_1.4-bc1cc36b.pt
│   └── mobilenetv2_1.0-0f96a698.pt
├── requirements.txt               # Dependencies
├── example_run.sh                 # Example scripts
└── README.md                      # Documentation
```

## 🔗 So sánh với dt2ds.py

| Tính năng | dt2ds.py | boxmot_tracking_detectron2.py |
|-----------|----------|-------------------------------|
| Detectron2 Model | ✅ | ✅ |
| SORT Tracker | ✅ | ✅ |
| DeepSort Tracker | ✅ | ✅ |
| ByteTrack | ❌ | ✅ |
| OcSort | ❌ | ✅ |
| BotSort | ❌ | ✅ |
| StrongSort | ❌ | ✅ |
| HybridSort | ❌ | ✅ |

## 📝 Notes

- Detectron2 segmentation models yêu cầu config file và model weights
- BoxMOT trackers yêu cầu detections format: `[x1, y1, x2, y2, conf]`
- Output format tương thích với MOT16/MOT17 evaluation
- Region-based tracking chỉ hoạt động với mask models (config file có `mask` trong tên)

## 🔗 Liên kết

- [BoxMOT GitHub](https://github.com/mikel-brostrom/boxmot)
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)
