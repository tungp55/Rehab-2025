#!/usr/bin/env python3
"""
Script kiểm tra số lượng frame trong video và so sánh với file nhãn
"""
import cv2
from pathlib import Path
from collections import defaultdict

def get_video_frame_count(video_path):
    """Lấy số frame từ video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def get_txt_frame_count(txt_path):
    """Lấy số frame từ file txt (frame_id lớn nhất)"""
    if not txt_path.exists():
        return None
    
    max_frame = 0
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) > 0:
                    try:
                        frame_id = int(parts[0])
                        max_frame = max(max_frame, frame_id)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"   ⚠️  Error reading {txt_path.name}: {e}")
        return None
    
    return max_frame if max_frame > 0 else None

def main():
    input_dir = Path("/home/vuhai/Rehab-Tung/test_input1")
    merge_txt_dir = Path("/home/vuhai/Rehab-Tung/test_output/bytetrack/merge_txt")
    txt_dir = Path("/home/vuhai/Rehab-Tung/test_output/bytetrack/txt")
    
    # Tìm tất cả video files
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    video_files = sorted(video_files)
    
    if len(video_files) == 0:
        print(f"❌ Không tìm thấy video nào trong: {input_dir}")
        return
    
    print("=" * 100)
    print("📊 Kiểm tra số lượng frame: Video vs File nhãn")
    print("=" * 100)
    print(f"📁 Video directory: {input_dir}")
    print(f"📁 Merge TXT directory: {merge_txt_dir}")
    print(f"📁 TXT directory: {txt_dir}")
    print()
    
    results = []
    issues = []
    
    print(f"{'Video Name':<50} {'Video Frames':<15} {'Merge TXT Frames':<18} {'TXT Frames':<15} {'Status':<15}")
    print("-" * 100)
    
    for video_file in video_files:
        video_stem = video_file.stem
        
        # Get video frame count
        video_frames = get_video_frame_count(video_file)
        if video_frames is None:
            print(f"{video_stem:<50} {'ERROR':<15} {'-':<18} {'-':<15} {'ERROR':<15}")
            continue
        
        # Get merge_txt frame count
        merge_txt_file = merge_txt_dir / f"{video_stem}_bytetrack_merged.txt"
        merge_txt_frames = get_txt_frame_count(merge_txt_file)
        
        # Get txt frame count
        txt_file = txt_dir / f"{video_stem}_bytetrack.txt"
        txt_frames = get_txt_frame_count(txt_file)
        
        # Check status
        status = []
        if merge_txt_frames is None:
            status.append("No merge_txt")
        elif merge_txt_frames != video_frames:
            status.append(f"Merge diff: {merge_txt_frames - video_frames:+d}")
            issues.append(f"{video_stem}: Video={video_frames}, Merge_TXT={merge_txt_frames}")
        
        if txt_frames is None:
            status.append("No txt")
        elif txt_frames != video_frames:
            status.append(f"TXT diff: {txt_frames - video_frames:+d}")
            issues.append(f"{video_stem}: Video={video_frames}, TXT={txt_frames}")
        
        if not status:
            status_str = "✅ OK"
        else:
            status_str = " | ".join(status)
        
        merge_txt_str = str(merge_txt_frames) if merge_txt_frames is not None else "N/A"
        txt_str = str(txt_frames) if txt_frames is not None else "N/A"
        
        print(f"{video_stem:<50} {video_frames:<15} {merge_txt_str:<18} {txt_str:<15} {status_str:<15}")
        
        results.append({
            'video': video_stem,
            'video_frames': video_frames,
            'merge_txt_frames': merge_txt_frames,
            'txt_frames': txt_frames,
            'status': status
        })
    
    print("-" * 100)
    print()
    
    # Summary
    print("=" * 100)
    print("📊 Tổng kết:")
    print("=" * 100)
    
    total_videos = len(results)
    merge_txt_ok = sum(1 for r in results if r['merge_txt_frames'] == r['video_frames'])
    txt_ok = sum(1 for r in results if r['txt_frames'] == r['video_frames'])
    merge_txt_missing = sum(1 for r in results if r['merge_txt_frames'] is None)
    txt_missing = sum(1 for r in results if r['txt_frames'] is None)
    
    print(f"Tổng số video: {total_videos}")
    print(f"Merge TXT:")
    print(f"   ✅ Khớp với video: {merge_txt_ok}/{total_videos}")
    print(f"   ❌ Không khớp: {total_videos - merge_txt_ok - merge_txt_missing}/{total_videos}")
    print(f"   ⚠️  Không có file: {merge_txt_missing}/{total_videos}")
    print(f"TXT:")
    print(f"   ✅ Khớp với video: {txt_ok}/{total_videos}")
    print(f"   ❌ Không khớp: {total_videos - txt_ok - txt_missing}/{total_videos}")
    print(f"   ⚠️  Không có file: {txt_missing}/{total_videos}")
    
    if issues:
        print()
        print("=" * 100)
        print("⚠️  Các vấn đề phát hiện:")
        print("=" * 100)
        for issue in issues:
            print(f"   - {issue}")
    
    print("=" * 100)

if __name__ == "__main__":
    main()
