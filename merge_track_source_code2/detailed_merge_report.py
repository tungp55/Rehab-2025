#!/usr/bin/env python3
"""
Script hiển thị chi tiết từng file về các tracks đã được merge.
"""

from pathlib import Path
from collections import defaultdict

def analyze_track_mapping(input_file, output_file):
    """Phân tích chi tiết mapping của tracks."""
    
    # Đọc input file
    input_tracks = defaultdict(set)  # track_id -> set of frame_ids
    input_track_frames = defaultdict(int)  # track_id -> số frames
    input_track_range = {}  # track_id -> (start_frame, end_frame)
    
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    input_tracks[track_id].add(frame_id)
                    input_track_frames[track_id] += 1
                    
                    # Cập nhật range
                    if track_id not in input_track_range:
                        input_track_range[track_id] = (frame_id, frame_id)
                    else:
                        start, end = input_track_range[track_id]
                        input_track_range[track_id] = (min(start, frame_id), max(end, frame_id))
                except:
                    pass
    
    # Đọc output file
    output_tracks = defaultdict(set)
    output_track_frames = defaultdict(int)
    output_track_range = {}
    
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    output_tracks[track_id].add(frame_id)
                    output_track_frames[track_id] += 1
                    
                    if track_id not in output_track_range:
                        output_track_range[track_id] = (frame_id, frame_id)
                    else:
                        start, end = output_track_range[track_id]
                        output_track_range[track_id] = (min(start, frame_id), max(end, frame_id))
                except:
                    pass
    
    # Tìm mapping: track nào input map sang track nào output
    input_to_output = {}
    for in_track, in_frames in input_tracks.items():
        best_match = None
        best_overlap = 0
        for out_track, out_frames in output_tracks.items():
            overlap = len(in_frames & out_frames)
            if overlap > best_overlap and overlap >= len(in_frames) * 0.9:  # 90% overlap
                best_overlap = overlap
                best_match = out_track
        if best_match is not None:
            input_to_output[in_track] = best_match
    
    # Tìm các tracks được merge: nhiều input tracks map sang cùng 1 output track
    output_to_inputs = defaultdict(list)
    for in_track, out_track in input_to_output.items():
        output_to_inputs[out_track].append(in_track)
    
    merged_tracks = {out: sorted(ins) for out, ins in output_to_inputs.items() if len(ins) > 1}
    
    return {
        'input_tracks': input_tracks,
        'output_tracks': output_tracks,
        'input_track_frames': input_track_frames,
        'output_track_frames': output_track_frames,
        'input_track_range': input_track_range,
        'output_track_range': output_track_range,
        'merged_tracks': merged_tracks,
        'mapping': input_to_output
    }


def print_file_details(filename, result):
    """In chi tiết cho một file."""
    print("\n" + "="*100)
    print(f"📄 FILE: {filename}")
    print("="*100)
    
    merged = result['merged_tracks']
    
    if not merged:
        print("\n⚠️  KHÔNG CÓ TRACKS NÀO ĐƯỢC MERGE TRONG FILE NÀY")
        print("   (Tất cả tracks giữ nguyên, chỉ có thể thay đổi track_id)")
        return
    
    print(f"\n✅ TỔNG QUAN:")
    print(f"   - Số tracks đầu vào: {len(result['input_tracks'])}")
    print(f"   - Số tracks đầu ra: {len(result['output_tracks'])}")
    print(f"   - Số nhóm tracks được merge: {len(merged)}")
    total_merged = sum(len(tracks) - 1 for tracks in merged.values())
    print(f"   - Tổng số tracks đã merge: {total_merged}")
    
    print(f"\n📊 CHI TIẾT CÁC TRACKS ĐƯỢC MERGE:")
    print("-"*100)
    
    # Sắp xếp theo số lượng tracks được merge (từ nhiều đến ít)
    sorted_merged = sorted(merged.items(), key=lambda x: len(x[1]), reverse=True)
    
    for idx, (output_track_id, input_track_ids) in enumerate(sorted_merged, 1):
        print(f"\n🔹 NHÓM {idx}: Output Track {output_track_id}")
        print(f"   Đã merge {len(input_track_ids)} tracks: {input_track_ids}")
        
        # Tính tổng số frames
        total_frames = sum(result['input_track_frames'][tid] for tid in input_track_ids)
        output_frames = result['output_track_frames'][output_track_id]
        
        print(f"   📈 Thống kê:")
        print(f"      - Tổng frames từ input tracks: {total_frames}")
        print(f"      - Frames trong output track: {output_frames}")
        print(f"      - Độ dài track sau merge: {output_frames} frames")
        
        # Hiển thị frame range của từng input track
        print(f"   📋 Chi tiết từng track đầu vào:")
        for in_track_id in sorted(input_track_ids):
            frames = result['input_track_frames'][in_track_id]
            if in_track_id in result['input_track_range']:
                start, end = result['input_track_range'][in_track_id]
                duration = end - start + 1
                print(f"      • Track {in_track_id:3d}: {frames:4d} frames, range [{start:5d} - {end:5d}], duration: {duration:4d} frames")
            else:
                print(f"      • Track {in_track_id:3d}: {frames:4d} frames")
        
        # Frame range của output track
        if output_track_id in result['output_track_range']:
            start, end = result['output_track_range'][output_track_id]
            duration = end - start + 1
            print(f"   📍 Output track range: [{start:5d} - {end:5d}], duration: {duration:4d} frames")
    
    # Hiển thị các tracks không được merge
    all_merged_input_tracks = set()
    for tracks in merged.values():
        all_merged_input_tracks.update(tracks)
    
    unmerged_tracks = sorted(set(result['input_tracks'].keys()) - all_merged_input_tracks)
    if unmerged_tracks:
        print(f"\n📌 CÁC TRACKS KHÔNG ĐƯỢC MERGE ({len(unmerged_tracks)} tracks):")
        for track_id in unmerged_tracks:
            frames = result['input_track_frames'][track_id]
            output_track = result['mapping'].get(track_id, 'N/A')
            if track_id in result['input_track_range']:
                start, end = result['input_track_range'][track_id]
                print(f"   • Input Track {track_id:3d} → Output Track {output_track:3d}: {frames:4d} frames, range [{start:5d} - {end:5d}]")
            else:
                print(f"   • Input Track {track_id:3d} → Output Track {output_track:3d}: {frames:4d} frames")


def main():
    input_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt")
    output_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt_merge_org")
    
    input_files = sorted(input_dir.glob("*.txt"))
    
    print("="*100)
    print("📊 BÁO CÁO CHI TIẾT: CÁC TRACKS ĐÃ ĐƯỢC MERGE")
    print("="*100)
    print(f"\nTổng số files: {len(input_files)}")
    
    for input_file in input_files:
        output_file = output_dir / input_file.name
        
        if not output_file.exists():
            print(f"\n⚠️  File output không tồn tại: {output_file.name}")
            continue
        
        try:
            result = analyze_track_mapping(input_file, output_file)
            print_file_details(input_file.name, result)
        except Exception as e:
            print(f"\n❌ Lỗi khi xử lý {input_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*100)
    print("✅ HOÀN TẤT")
    print("="*100)


if __name__ == '__main__':
    main()
