"""
Test All Trackers Script với Detectron2
Chạy tất cả các thuật toán tracking trên cùng một video để so sánh
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


def test_all_trackers(args):
    """Test all trackers on the same video"""
    input_video = Path(args.input)
    
    if not input_video.exists():
        print(f"❌ Input video not found: {input_video}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each tracker
    for tracker in args.trackers:
        (output_dir / tracker / "videos").mkdir(parents=True, exist_ok=True)
        (output_dir / tracker / "txt").mkdir(parents=True, exist_ok=True)
    
    video_stem = input_video.stem
    
    # List of trackers to test
    trackers = args.trackers
    
    print(f"🧪 Testing {len(trackers)} trackers on: {input_video.name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Trackers: {', '.join(trackers)}\n")
    
    # Process each tracker
    for i, tracker in enumerate(trackers, 1):
        print(f"\n[{i}/{len(trackers)}] Testing {tracker.upper()} tracker...")
        
        # Generate output paths
        out_vid = output_dir / tracker / "videos" / f"{video_stem}_{tracker}.avi"
        out_txt = output_dir / tracker / "txt" / f"{video_stem}_{tracker}.txt"
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "boxmot_tracking_detectron2.py"),
            "--input", str(input_video),
            "--config-file", args.config_file,
            "--model-weights", args.model_weights,
            "--tracker", tracker,
            "--out_vid", str(out_vid),
            "--out_txt", str(out_txt),
            "--device", args.device,
            "--confidence-threshold", str(args.confidence_threshold),
            "--region_based", str(args.region_based),
            "--num-classes", str(args.num_classes),
        ]
        
        if args.fps > 0:
            cmd.extend(["--fps", str(args.fps)])
        
        # Run command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {tracker.upper()} completed: {out_vid.name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error with {tracker}: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
    
    print(f"\n✅ All trackers tested!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"\n📊 Summary:")
    for tracker in trackers:
        out_vid = output_dir / tracker / "videos" / f"{video_stem}_{tracker}.avi"
        out_txt = output_dir / tracker / "txt" / f"{video_stem}_{tracker}.txt"
        if out_vid.exists():
            size_mb = out_vid.stat().st_size / (1024 * 1024)
            print(f"  {tracker.upper()}: Video ({size_mb:.2f} MB), TXT ({'✓' if out_txt.exists() else '✗'})")
        else:
            print(f"  {tracker.upper()}: ❌ Failed")


def main():
    parser = argparse.ArgumentParser(description="Test all BoxMOT trackers with Detectron2 on the same video")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="Output directory for results",
    )
    
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
        "--trackers",
        type=str,
        nargs='+',
        default=['bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort', 'sort', 'deepsort'],
        choices=['bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort', 'sort', 'deepsort'],
        help="List of trackers to test",
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes",
    )
    
    parser.add_argument(
        "--region_based",
        type=int,
        default=1,
        help="1 for region-based tracking, 0 for full image",
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device: 'cuda' or 'cpu'",
    )
    
    args = parser.parse_args()
    test_all_trackers(args)


if __name__ == "__main__":
    main()
