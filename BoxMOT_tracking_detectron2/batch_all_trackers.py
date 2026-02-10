"""
Batch Processing với Tất cả Trackers
Chạy tất cả các thuật toán tracking trên tất cả video trong một thư mục
Output: Mỗi tracker có thư mục riêng với txt/ và videos/ subdirectories
Cấu trúc: {output_dir}/{tracker}/txt/{video_name}_{tracker}.txt
          {output_dir}/{tracker}/videos/{video_name}_{tracker}.avi
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_video_files(input_dir):
    """Get all video files from directory"""
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f'*{ext}'))
        video_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    return sorted(video_files)


def process_video_with_all_trackers(video_path, output_dir, args):
    """
    Process a single video with all trackers
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for results
        args: Arguments containing config, model, etc.
    """
    video_stem = video_path.stem
    
    # List of all trackers to test
    # all_trackers = ['bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort', 'sort', 'deepsort']
    all_trackers = [ 'sort', 'deepsort']
    
    # Process each tracker
    for tracker in all_trackers:
        # Create tracker-specific directories
        tracker_txt_dir = output_dir / tracker / "txt"
        tracker_vid_dir = output_dir / tracker / "videos"
        tracker_txt_dir.mkdir(parents=True, exist_ok=True)
        tracker_vid_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output paths - organized by tracker
        out_txt = tracker_txt_dir / f"{video_stem}_{tracker}.txt"
        out_vid = tracker_vid_dir / f"{video_stem}_{tracker}.avi"
        
        # Build command - use absolute path and set working directory
        script_path = Path(__file__).parent.absolute() / "boxmot_tracking_detectron2.py"
        script_dir = str(Path(__file__).parent.absolute())
        
        cmd = [
            sys.executable,
            str(script_path),
            "--input", str(video_path),
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
        
        # Add ReID weights for trackers that need them (will auto-detect if not provided)
        if tracker in ['strongsort', 'botsort', 'hybridsort'] and args.reid_weights:
            cmd.extend(["--reid-weights", args.reid_weights])
        
        if args.fps > 0:
            cmd.extend(["--fps", str(args.fps)])
        
        # Run command - set working directory and PYTHONPATH to script directory
        try:
            env = os.environ.copy()
            script_dir = str(Path(__file__).parent.absolute())
            
            # Force PYTHONPATH to only include script directory (remove all others)
            # Also remove any paths that might cause conflicts
            env['PYTHONPATH'] = script_dir
            
            # Remove PYTHONPATH entries that might interfere
            if 'PYTHONPATH' in env:
                # Keep only our script directory
                env['PYTHONPATH'] = script_dir
            
            # Set CUDA_VISIBLE_DEVICES if not set and using CUDA
            if args.device == 'cuda' and 'CUDA_VISIBLE_DEVICES' not in env:
                env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            
            # Also set __file__ environment to help with path resolution
            env['SCRIPT_DIR'] = script_dir
            
            subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=3600,  # 1 hour timeout per video
                cwd=script_dir,  # Set working directory
                env=env  # Use modified environment
            )
            print(f"  ✅ {tracker.upper()}: {out_vid.name}")
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  {tracker.upper()}: Timeout (skipped)")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {tracker.upper()}: Failed")
            if args.verbose:
                print(f"     Error: {e.stderr[:200]}")
        except Exception as e:
            print(f"  ❌ {tracker.upper()}: Error - {e}")


def process_directory(args):
    """Process all videos in a directory with all trackers"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_files = get_video_files(input_dir)
    
    if len(video_files) == 0:
        print(f"❌ No video files found in: {input_dir}")
        return
    
    print(f"📹 Found {len(video_files)} video files")
    print(f"📁 Output directory: {output_dir}")
    print("🎯 Trackers: bytetrack, ocsort, botsort, strongsort, hybridsort, sort, deepsort\n")
    
    # Process each video
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        process_video_with_all_trackers(video_file, output_dir, args)
    
    print("\n✅ Batch processing completed!")
    print(f"📁 All results saved in: {output_dir}")
    
    # Summary
    print("\n📊 Summary:")
    for tracker in ['bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort', 'sort', 'deepsort']:
        tracker_txt_dir = output_dir / tracker / "txt"
        tracker_vid_dir = output_dir / tracker / "videos"
        txt_count = len(list(tracker_txt_dir.glob("*.txt"))) if tracker_txt_dir.exists() else 0
        vid_count = len(list(tracker_vid_dir.glob("*.avi"))) if tracker_vid_dir.exists() else 0
        print(f"  {tracker.upper()}: {txt_count} TXT files, {vid_count} video files")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos with all BoxMOT trackers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Process all videos in a directory
        python batch_all_trackers.py \\
            --input-dir /path/to/videos \\
            --output-dir /path/to/output \\
            --config-file /path/to/config.yaml \\
            --model-weights /path/to/model.pth

        # With custom ReID weights
        python batch_all_trackers.py \\
            --input-dir /path/to/videos \\
            --output-dir /path/to/output \\
            --config-file /path/to/config.yaml \\
            --model-weights /path/to/model.pth \\
            --reid-weights /path/to/reid.pt

        Output structure:
        output_dir/
        ├── bytetrack/
        │   ├── txt/
        │   │   ├── video1_bytetrack.txt
        │   │   └── video2_bytetrack.txt
        │   └── videos/
        │       ├── video1_bytetrack.avi
        │       └── video2_bytetrack.avi
        ├── ocsort/
        │   ├── txt/
        │   └── videos/
        ├── botsort/
        │   ├── txt/
        │   └── videos/
        ...
        """
    )
    
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
        help="Output directory. Structure: {output_dir}/{tracker}/txt/ and {output_dir}/{tracker}/videos/",
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
        "--reid-weights",
        "--reid-weight",  # Alias for backward compatibility
        type=str,
        default=None,
        dest="reid_weights",
        help="ReID weights for trackers that require it (strongsort, botsort, hybridsort). "
             "If not provided, will auto-detect from ./reID_weight/ directory",
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
    
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Show detailed error messages",
    )
    
    args = parser.parse_args()
    process_directory(args)


if __name__ == "__main__":
    main()
