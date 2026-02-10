#!/usr/bin/env python3
"""
Script phân tích các frame bị thiếu trong file TXT so với video
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

def analyze_txt_frames(txt_path, video_frames):
    """Phân tích các frame trong file TXT"""
    if not txt_path.exists():
        return None
    
    frames_in_txt = set()
    frame_ranges = []
    
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
                        frames_in_txt.add(frame_id)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"   ⚠️  Error reading {txt_path.name}: {e}")
        return None
    
    # Tìm các frame bị thiếu
    missing_frames = []
    for frame_id in range(1, video_frames + 1):
        if frame_id not in frames_in_txt:
            missing_frames.append(frame_id)
    
    # Tìm các khoảng frame liên tục bị thiếu
    missing_ranges = []
    if missing_frames:
        start = missing_frames[0]
        end = missing_frames[0]
        for i in range(1, len(missing_frames)):
            if missing_frames[i] == end + 1:
                end = missing_frames[i]
            else:
                if start == end:
                    missing_ranges.append(f"{start}")
                else:
                    missing_ranges.append(f"{start}-{end}")
                start = missing_frames[i]
                end = missing_frames[i]
        if start == end:
            missing_ranges.append(f"{start}")
        else:
            missing_ranges.append(f"{start}-{end}")
    
    return {
        'total_frames_in_txt': len(frames_in_txt),
        'max_frame_in_txt': max(frames_in_txt) if frames_in_txt else 0,
        'missing_count': len(missing_frames),
        'missing_frames': missing_frames,
        'missing_ranges': missing_ranges,
        'first_missing': missing_frames[0] if missing_frames else None,
        'last_missing': missing_frames[-1] if missing_frames else None
    }

def check_video_reading(video_path):
    """Kiểm tra xem video có đọc được hết frame không"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frames_read = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        actual_frames_read += 1
    
    cap.release()
    
    return {
        'reported_count': reported_frame_count,
        'actual_read': actual_frames_read,
        'difference': reported_frame_count - actual_frames_read
    }

def main():
    input_dir = Path("/home/vuhai/Rehab-Tung/test_input1")
    txt_dir = Path("/home/vuhai/Rehab-Tung/test_output/bytetrack/txt")
    
    # Các video có vấn đề
    problem_videos = [
        "GH010371_8_12834_15150",
        "GH010376_8_3477_4145",
        "GH010376_8_4621_5179"
    ]
    
    print("=" * 100)
    print("🔍 Phân tích các frame bị thiếu")
    print("=" * 100)
    
    for video_stem in problem_videos:
        print()
        print("=" * 100)
        print(f"📹 Video: {video_stem}")
        print("=" * 100)
        
        # Tìm video file
        video_file = None
        for ext in ['.avi', '.mp4', '.mov', '.mkv']:
            video_file = input_dir / f"{video_stem}{ext}"
            if video_file.exists():
                break
        
        if not video_file or not video_file.exists():
            print(f"❌ Không tìm thấy video file: {video_stem}")
            continue
        
        # Get video frame count
        video_frames = get_video_frame_count(video_file)
        print(f"📊 Video frame count (reported): {video_frames}")
        
        # Check actual frames readable
        print("\n🔍 Kiểm tra khả năng đọc video:")
        video_check = check_video_reading(video_file)
        if video_check:
            print(f"   Reported frames: {video_check['reported_count']}")
            print(f"   Actual frames read: {video_check['actual_read']}")
            if video_check['difference'] != 0:
                print(f"   ⚠️  Chênh lệch: {video_check['difference']} frames")
            else:
                print(f"   ✅ Video có thể đọc được tất cả frames")
        
        # Analyze TXT file
        txt_file = txt_dir / f"{video_stem}_bytetrack.txt"
        print(f"\n📄 Phân tích file TXT: {txt_file.name}")
        
        if not txt_file.exists():
            print(f"   ❌ File không tồn tại")
            continue
        
        analysis = analyze_txt_frames(txt_file, video_frames)
        if analysis:
            print(f"   Total frames in TXT: {analysis['total_frames_in_txt']}")
            print(f"   Max frame ID in TXT: {analysis['max_frame_in_txt']}")
            print(f"   Missing frames: {analysis['missing_count']}")
            
            if analysis['missing_count'] > 0:
                print(f"\n   ⚠️  Các frame bị thiếu:")
                print(f"      First missing: {analysis['first_missing']}")
                print(f"      Last missing: {analysis['last_missing']}")
                
                if len(analysis['missing_ranges']) <= 10:
                    print(f"      Ranges: {', '.join(analysis['missing_ranges'])}")
                else:
                    print(f"      First 10 ranges: {', '.join(analysis['missing_ranges'][:10])}")
                    print(f"      ... and {len(analysis['missing_ranges']) - 10} more ranges")
                
                # Kiểm tra xem frame bị thiếu ở đâu
                if analysis['first_missing'] == 1:
                    print(f"\n   🔍 Frame bị thiếu từ đầu video")
                elif analysis['last_missing'] == video_frames:
                    print(f"\n   🔍 Frame bị thiếu ở cuối video (từ frame {analysis['first_missing']} đến {analysis['last_missing']})")
                else:
                    print(f"\n   🔍 Frame bị thiếu ở giữa video")
        
        print()
    
    print("=" * 100)
    print("💡 Gợi ý:")
    print("=" * 100)
    print("Nếu frame bị thiếu ở cuối video, có thể do:")
    print("  1. Video có frame cuối không đọc được")
    print("  2. Code xử lý dừng sớm do lỗi")
    print("  3. Video metadata báo sai số frame")
    print()
    print("Nếu frame bị thiếu ở đầu video, có thể do:")
    print("  1. Frame đầu không được detect")
    print("  2. Code bắt đầu từ frame 1 nhưng video bắt đầu từ frame 0")
    print()
    print("Nếu frame bị thiếu ở giữa video, có thể do:")
    print("  1. Frame bị skip do lỗi đọc")
    print("  2. Frame không có detection nên không được ghi")

if __name__ == "__main__":
    main()
