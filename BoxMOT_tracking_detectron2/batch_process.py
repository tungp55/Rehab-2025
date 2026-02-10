"""
Batch Processing Script cho Detectron2 + BoxMOT
Xử lý nhiều video trong một thư mục
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def get_video_files(input_dir):
    """Get all video files from directory"""
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f'*{ext}'))
        video_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    return sorted(video_files)


def process_directory(args):
    """Process all videos in a directory"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    output_vid_dir = output_dir / "videos"
    output_txt_dir = output_dir / "txt"
    output_vid_dir.mkdir(parents=True, exist_ok=True)
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_files = get_video_files(input_dir)
    
    if len(video_files) == 0:
        print(f"❌ No video files found in: {input_dir}")
        return
    
    print(f"📹 Found {len(video_files)} video files")
    
    # Process each video
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        
        # Generate output paths
        video_stem = video_file.stem
        out_vid = output_vid_dir / f"{video_stem}_{args.tracker}.avi"
        out_txt = output_txt_dir / f"{video_stem}_{args.tracker}.txt"
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "boxmot_tracking_detectron2.py"),
            "--input", str(video_file),
            "--config-file", args.config_file,
            "--model-weights", args.model_weights,
            "--tracker", args.tracker,
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
            print(f"✅ Completed: {video_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {video_file.name}: {e}")
            print(f"Error output: {e.stderr}")
    
    print(f"\n✅ Batch processing completed!")
    print(f"📁 Output videos: {output_vid_dir}")
    print(f"📁 Output texts: {output_txt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch process videos with Detectron2 + BoxMOT tracking")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing videos",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
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
        "--tracker",
        type=str,
        default='bytetrack',
        choices=['bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort', 'sort', 'deepsort'],
        help='Tracker type',
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
    process_directory(args)


if __name__ == "__main__":
    main()
