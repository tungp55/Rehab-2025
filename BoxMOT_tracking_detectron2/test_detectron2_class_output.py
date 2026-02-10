"""
Test script để kiểm tra xem Detectron2 model có output class_id không
"""
import sys
import cv2
import numpy as np

# Add Detectron2 path
sys.path.append('/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2')

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def test_detectron2_class_output(config_file, model_weights, video_path, num_classes=2, num_frames=5):
    """
    Test xem Detectron2 có output class_id không
    
    Args:
        config_file: Path to Detectron2 config file
        model_weights: Path to model weights
        video_path: Path to test video (hoặc None để dùng ảnh test)
        num_classes: Number of classes
        num_frames: Số frame để test
    """
    print("=" * 60)
    print("🧪 TEST: Kiểm tra Detectron2 class_id output")
    print("=" * 60)
    
    # Load model
    print(f"\n📥 Loading Detectron2 model...")
    print(f"   Config: {config_file}")
    print(f"   Weights: {model_weights}")
    print(f"   Num classes: {num_classes}")
    
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_weights
    
    predictor = DefaultPredictor(cfg)
    print("✅ Model loaded!\n")
    
    # Load video hoặc tạo test image
    if video_path:
        print(f"📹 Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
    else:
        print("📷 Creating test image (640x480)...")
        cap = None
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gray background
    
    print("\n" + "=" * 60)
    print("🔍 Checking predictions structure...")
    print("=" * 60)
    
    frame_count = 0
    found_classes = False
    
    for i in range(num_frames):
        if cap:
            ret, im = cap.read()
            if not ret:
                print(f"⚠️  Cannot read frame {i+1}")
                break
        else:
            im = test_image.copy()
        
        # Run inference
        predictions = predictor(im)
        instances = predictions["instances"]
        
        print(f"\n📊 Frame {i+1}:")
        print(f"   Number of detections: {len(instances)}")
        
        if len(instances) > 0:
            # Check available fields
            print(f"\n   Available fields in instances:")
            for field in instances.get_fields():
                print(f"      - {field}")
            
            # Check for pred_classes
            if instances.has("pred_classes"):
                classes = instances.pred_classes.cpu().numpy()
                print(f"\n   ✅ FOUND pred_classes!")
                print(f"   Class IDs: {classes}")
                print(f"   Class IDs type: {type(classes)}")
                print(f"   Class IDs shape: {classes.shape}")
                print(f"   Unique classes: {np.unique(classes)}")
                found_classes = True
            else:
                print(f"\n   ❌ NO pred_classes field found!")
            
            # Check other important fields
            if instances.has("pred_boxes"):
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                print(f"   Boxes shape: {boxes.shape}")
            
            if instances.has("scores"):
                scores = instances.scores.cpu().numpy()
                print(f"   Scores: {scores[:3] if len(scores) >= 3 else scores}")  # Show first 3
            
            # Show full structure for first frame
            if i == 0:
                print(f"\n   Full instances structure:")
                print(f"      {instances}")
        
        frame_count += 1
    
    if cap:
        cap.release()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    if found_classes:
        print("✅ SUCCESS: Detectron2 model DOES output class_id (pred_classes)")
        print("   → Có thể sử dụng class_id từ predictions['instances'].pred_classes")
    else:
        print("❌ FAILED: Detectron2 model does NOT output class_id")
        print("   → Cần kiểm tra lại model hoặc config")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Detectron2 class_id output")
    
    parser.add_argument(
        "--config-file",
        type=str,
        default="/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        help="Path to Detectron2 config file",
    )
    
    parser.add_argument(
        "--model-weights",
        type=str,
        default="/home/vuhai/Rehab-Tung/Detectron2DeepSortPlus/model_0004999.pth",
        help="Path to Detectron2 model weights",
    )
    
    parser.add_argument(
        "--video",
        type=str,
        default="/home/vuhai/Rehab-Tung/videos/GH010354_5_378_3750.avi",
        help="Path to test video (or None to use test image)",
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes",
    )
    
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of frames to test",
    )
    
    args = parser.parse_args()
    
    test_detectron2_class_output(
        args.config_file,
        args.model_weights,
        args.video,
        args.num_classes,
        args.num_frames
    )
