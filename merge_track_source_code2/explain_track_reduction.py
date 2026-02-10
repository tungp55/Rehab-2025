#!/usr/bin/env python3
"""
Giải thích chi tiết về cách tính phần trăm giảm tracks.
"""

from pathlib import Path
from collections import defaultdict

def analyze_track_reduction_detailed(input_file, output_file):
    """Phân tích chi tiết với giải thích rõ ràng."""
    
    # Đọc input file
    input_tracks = defaultdict(set)
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    input_tracks[track_id].add(frame_id)
                except:
                    pass
    
    # Đọc output file
    output_tracks = defaultdict(set)
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    output_tracks[track_id].add(frame_id)
                except:
                    pass
    
    # Tìm mapping
    input_to_output = {}
    for in_track, in_frames in input_tracks.items():
        best_match = None
        best_overlap = 0
        for out_track, out_frames in output_tracks.items():
            overlap = len(in_frames & out_frames)
            if overlap > best_overlap and overlap >= len(in_frames) * 0.9:
                best_overlap = overlap
                best_match = out_track
        if best_match is not None:
            input_to_output[in_track] = best_match
    
    # Tìm các tracks được merge
    output_to_inputs = defaultdict(list)
    for in_track, out_track in input_to_output.items():
        output_to_inputs[out_track].append(in_track)
    
    merged_tracks = {out: sorted(ins) for out, ins in output_to_inputs.items() if len(ins) > 1}
    
    # Tính toán
    num_input_tracks = len(input_tracks)
    num_output_tracks = len(output_tracks)
    
    # Số tracks đã được merge
    total_merged_tracks = sum(len(tracks) - 1 for tracks in merged_tracks.values())
    
    # Số tracks KHÔNG được merge (giữ nguyên)
    unmerged_tracks = num_input_tracks - sum(len(tracks) for tracks in merged_tracks.values())
    
    # Số tracks còn lại sau merge (theo lý thuyết nếu chỉ đếm unique tracks)
    # = số nhóm merge + số tracks không merge
    theoretical_unique_tracks = len(merged_tracks) + unmerged_tracks
    
    return {
        'input_tracks': num_input_tracks,
        'output_tracks': num_output_tracks,
        'merged_tracks': total_merged_tracks,
        'unmerged_tracks': unmerged_tracks,
        'merged_groups': len(merged_tracks),
        'theoretical_unique_tracks': theoretical_unique_tracks,
        'merged_details': merged_tracks
    }


def main():
    input_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt")
    output_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt_merge_org")
    
    # Top 5 files
    top_files = [
        "GH010375_7_1628_4805_deepsort.txt",
        "GH010371_5_1132_5000_deepsort.txt",
        "GH010358_5_16380_17200_deepsort.txt",
        "GH010382_5_5725_7093_deepsort.txt",
        "GH010371_6_9700_10700_deepsort.txt"
    ]
    
    print("="*100)
    print("📊 GIẢI THÍCH CHI TIẾT: Ý NGHĨA CỦA CON SỐ 31/36, 16/20, v.v.")
    print("="*100)
    print()
    print("⚠️  LƯU Ý QUAN TRỌNG:")
    print("   - 31/36 KHÔNG có nghĩa là 36 tracks giảm còn 31 tracks")
    print("   - 31/36 có nghĩa là: trong 36 tracks đầu vào, có 31 tracks đã được MERGE (gộp lại)")
    print("   - Số tracks output vẫn là 36 vì các tracks được merge thành track_id mới")
    print()
    print("="*100)
    
    for filename in top_files:
        input_file = input_dir / filename
        output_file = output_dir / filename
        
        if not input_file.exists() or not output_file.exists():
            continue
        
        result = analyze_track_reduction_detailed(input_file, output_file)
        
        print(f"\n📄 FILE: {filename}")
        print("-"*100)
        print(f"📊 SỐ LIỆU:")
        print(f"   • Tổng số tracks đầu vào: {result['input_tracks']}")
        print(f"   • Tổng số tracks đầu ra (unique track IDs): {result['output_tracks']}")
        print(f"   • Số tracks đã được MERGE: {result['merged_tracks']}")
        print(f"   • Số tracks KHÔNG được merge (giữ nguyên): {result['unmerged_tracks']}")
        print(f"   • Số nhóm tracks được merge: {result['merged_groups']}")
        print()
        
        print(f"💡 GIẢI THÍCH CON SỐ {result['merged_tracks']}/{result['input_tracks']}:")
        print(f"   • Trong {result['input_tracks']} tracks đầu vào:")
        print(f"     - Có {result['merged_tracks']} tracks đã được MERGE (gộp vào các nhóm khác)")
        print(f"     - Có {result['unmerged_tracks']} tracks KHÔNG được merge (giữ nguyên)")
        print(f"     - Tổng: {result['merged_tracks']} + {result['unmerged_tracks']} = {result['input_tracks']} ✓")
        print()
        
        print(f"📈 PHẦN TRĂM GIẢM:")
        reduction_pct = (result['merged_tracks'] / result['input_tracks'] * 100)
        print(f"   • {reduction_pct:.1f}% = ({result['merged_tracks']}/{result['input_tracks']}) × 100")
        print(f"   • Có nghĩa là: {reduction_pct:.1f}% số tracks đã được merge")
        print()
        
        if result['merged_details']:
            print(f"🔹 CHI TIẾT CÁC NHÓM MERGE:")
            for out_track, in_tracks in sorted(result['merged_details'].items(), key=lambda x: len(x[1]), reverse=True):
                print(f"   • Output Track {out_track}: merge {len(in_tracks)} tracks {in_tracks}")
                print(f"     → Trong nhóm này, {len(in_tracks) - 1} tracks đã được merge vào track chính")
        
        print()
        print(f"📌 TÓM TẮT:")
        print(f"   • Input: {result['input_tracks']} tracks")
        print(f"   • Output: {result['output_tracks']} unique track IDs (vẫn bằng input)")
        print(f"   • Đã merge: {result['merged_tracks']} tracks ({reduction_pct:.1f}%)")
        print(f"   • Không merge: {result['unmerged_tracks']} tracks")
        print(f"   • Theo lý thuyết (nếu chỉ đếm unique): {result['theoretical_unique_tracks']} tracks")
        print()
        print("="*100)
    
    print("\n✅ KẾT LUẬN:")
    print("   • Con số 31/36, 16/20, v.v. là số tracks ĐÃ ĐƯỢC MERGE / tổng số tracks đầu vào")
    print("   • KHÔNG phải là số tracks còn lại sau khi giảm")
    print("   • Phần trăm giảm cho biết tỷ lệ tracks đã được merge (gộp lại)")


if __name__ == '__main__':
    main()
