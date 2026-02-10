"""
Test script để kiểm tra class_id được lưu vào file txt sau khi tracking
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add paths
sys.path.append('/home/vuhai/Rehab-Tung/BoxMOT_tracking_detectron2')
from detector_detectron2 import Detectron2SegmentationDetector
from boxmot_tracking_detectron2 import process_frame_boxmot, create_boxmot_tracker
import argparse


def test_class_id_integration(config_file, model_weights, video_path, num_frames=10):
    """
    Test xem class_id có được lưu vào file txt không
    """
    print("=" * 60)
    print("🧪 TEST: Kiểm tra class_id trong tracking output")
    print("=" * 60)
    
    # Load detector
    print(f"\n📥 Loading Detectron2 model...")
    detector = Detectron2SegmentationDetector(
        config_file=config_file,
        model_weights=model_weights,
        conf_threshold=0.5,
        num_classes=2
    )
    
    # Load tracker
    print(f"\n📥 Creating ByteTrack tracker...")
    args = argparse.Namespace(
        reid_weights=None,
        tracker_config=None,
        device='cuda'
    )
    tracker = create_boxmot_tracker('bytetrack', args)
    
    # Load video
    print(f"\n📹 Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Test output file
    output_txt = Path("./test_class_id_output.txt")
    output_txt.unlink(missing_ok=True)  # Remove if exists
    
    print(f"\n🔍 Processing {num_frames} frames...")
    print("=" * 60)
    
    total_counter = [0] * 1000000
    class_ids_found = []
    
    with open(output_txt, 'w') as f:
        for frameID in range(num_frames):
            ret, im = cap.read()
            if not ret:
                break
            
            # Detect
            dets, masks, region = detector.detect(im)
            
            # Check detections format
            if len(dets) > 0:
                dets_array = np.array(dets)
                print(f"\n📊 Frame {frameID + 1}:")
                print(f"   Detections shape: {dets_array.shape}")
                print(f"   First detection: {dets_array[0]}")
                if dets_array.shape[1] == 6:
                    print(f"   ✅ Detections có class_id: {dets_array[0, 5]}")
                else:
                    print(f"   ❌ Detections không có class_id (shape={dets_array.shape[1]})")
            
            # Track
            outputs = process_frame_boxmot(tracker, dets, im, frameID)
            
            # Check outputs format
            if len(outputs) > 0:
                print(f"   Outputs shape: {outputs.shape}")
                print(f"   First output: {outputs[0]}")
                if outputs.shape[1] >= 6:
                    print(f"   ✅ Outputs có class_id: {outputs[0, 5]}")
                    class_ids_found.append(int(outputs[0, 5]))
                else:
                    print(f"   ❌ Outputs không có class_id (shape={outputs.shape[1]})")
                
                # Process and write to file
                tlbr_boxes = outputs[:, :4]
                if outputs.shape[1] >= 6:
                    identities = outputs[:, 4].astype(int)
                    class_ids_output = outputs[:, 5].astype(int)
                else:
                    identities = outputs[:, -1].astype(int)
                    class_ids_output = np.ones(len(outputs), dtype=int)
                
                ordered_identities = []
                for identity in identities:
                    if identity >= len(total_counter):
                        print(f'⚠️  Out of size {len(total_counter)}/{identity}')
                    if not total_counter[identity]:
                        total_counter[identity] = max(total_counter) + 1 if max(total_counter) > 0 else 1
                    ordered_identities.append(total_counter[identity])
                
                # Write to file
                for i in range(len(ordered_identities)):
                    tlbr = tlbr_boxes[i]
                    class_id = int(class_ids_output[i]) if i < len(class_ids_output) else 1
                    line = [
                        frameID + 1,
                        ordered_identities[i],
                        tlbr[0],
                        tlbr[1],
                        tlbr[2] - tlbr[0],
                        tlbr[3] - tlbr[1],
                        1.0,
                        class_id,
                        1
                    ]
                    f.write(",".join(str(item) for item in line) + "\n")
    
    cap.release()
    
    # Check output file
    print("\n" + "=" * 60)
    print("📋 Checking output file...")
    print("=" * 60)
    
    if output_txt.exists():
        lines = output_txt.read_text().strip().split('\n')
        print(f"✅ Output file created: {output_txt}")
        print(f"   Total lines: {len(lines)}")
        
        if len(lines) > 0:
            # Check first few lines
            print(f"\n   First 3 lines:")
            for i, line in enumerate(lines[:3]):
                parts = line.split(',')
                print(f"   Line {i+1}: {parts}")
                if len(parts) >= 8:
                    class_id_in_file = int(parts[7])
                    print(f"      → Class ID: {class_id_in_file}")
                    if class_id_in_file not in [0, 1]:
                        print(f"      ⚠️  Warning: Class ID is not 0 or 1")
            
            # Check all class_ids in file
            all_class_ids = []
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 8:
                    all_class_ids.append(int(parts[7]))
            
            unique_classes = sorted(set(all_class_ids))
            print(f"\n   Unique class IDs in file: {unique_classes}")
            print(f"   Class ID distribution:")
            for cls_id in unique_classes:
                count = all_class_ids.count(cls_id)
                print(f"      Class {cls_id}: {count} detections")
            
            if len(unique_classes) > 1 or (len(unique_classes) == 1 and unique_classes[0] in [0, 1]):
                print(f"\n   ✅ SUCCESS: Class IDs are being saved correctly!")
            else:
                print(f"\n   ⚠️  Warning: Only one class ID found or unexpected values")
    else:
        print(f"❌ Output file not created!")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test class_id integration")
    
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
        help="Path to test video",
    )
    
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames to test",
    )
    
    args = parser.parse_args()
    
    test_class_id_integration(
        args.config_file,
        args.model_weights,
        args.video,
        args.num_frames
    )
