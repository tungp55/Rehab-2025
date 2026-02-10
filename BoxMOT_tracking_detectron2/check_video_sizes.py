#!/usr/bin/env python3
"""
Script kiểm tra kích thước (width, height) của các video trong thư mục
"""
import cv2
from pathlib import Path

def get_video_info(video_path):
    """Lấy thông tin về video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    }

def main():
    input_dir = Path("/home/vuhai/Rehab-Tung/test_input1")
    
    if not input_dir.exists():
        print(f"❌ Thư mục không tồn tại: {input_dir}")
        return
    
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
    
    print("=" * 80)
    print(f"📹 Kiểm tra kích thước video trong: {input_dir}")
    print("=" * 80)
    print(f"Tìm thấy {len(video_files)} video file(s)\n")
    
    # Lưu thông tin để thống kê
    sizes = {}
    all_same = True
    first_size = None
    
    print(f"{'Tên file':<50} {'Width':<10} {'Height':<10} {'FPS':<8} {'Frames':<10} {'Duration (s)':<12}")
    print("-" * 80)
    
    for i, video_file in enumerate(video_files, 1):
        info = get_video_info(video_file)
        if info is None:
            print(f"{video_file.name:<50} {'ERROR':<10} {'ERROR':<10}")
            continue
        
        width = info['width']
        height = info['height']
        fps = info['fps']
        frame_count = info['frame_count']
        duration = info['duration']
        
        size_key = f"{width}x{height}"
        if size_key not in sizes:
            sizes[size_key] = 0
        sizes[size_key] += 1
        
        if first_size is None:
            first_size = size_key
        elif size_key != first_size:
            all_same = False
        
        print(f"{video_file.name:<50} {width:<10} {height:<10} {fps:<8.2f} {frame_count:<10} {duration:<12.2f}")
    
    print("-" * 80)
    print("\n📊 Thống kê:")
    print(f"   Tổng số video: {len(video_files)}")
    print(f"   Số kích thước khác nhau: {len(sizes)}")
    
    if all_same:
        print(f"   ✅ Tất cả video có cùng kích thước: {first_size}")
    else:
        print(f"   ⚠️  Video có kích thước khác nhau:")
        for size, count in sorted(sizes.items()):
            print(f"      - {size}: {count} video(s)")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
