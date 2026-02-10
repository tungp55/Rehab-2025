#!/usr/bin/env python3
"""
Tính toán phần trăm giảm số tracks trên từng file sau khi merge.
"""

from pathlib import Path
from collections import defaultdict

def analyze_track_reduction(input_file, output_file):
    """Phân tích và tính phần trăm giảm tracks."""
    
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
    
    # Số tracks đã được merge (tổng số tracks trong các nhóm merge trừ đi số nhóm)
    # Ví dụ: merge 3 tracks thành 1 → giảm 2 tracks
    total_merged_tracks = sum(len(tracks) - 1 for tracks in merged_tracks.values())
    
    # Phần trăm giảm dựa trên số tracks đã merge
    reduction_pct = (total_merged_tracks / num_input_tracks * 100) if num_input_tracks > 0 else 0
    
    # Số tracks còn lại sau merge (theo lý thuyết)
    theoretical_output = num_input_tracks - total_merged_tracks
    
    return {
        'input_tracks': num_input_tracks,
        'output_tracks': num_output_tracks,
        'merged_tracks': total_merged_tracks,
        'reduction_pct': reduction_pct,
        'theoretical_output': theoretical_output,
        'merged_groups': len(merged_tracks)
    }


def main():
    input_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt")
    output_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt_merge_org")
    
    input_files = sorted(input_dir.glob("*.txt"))
    
    print("="*100)
    print("📊 PHẦN TRĂM GIẢM SỐ TRACKS TRÊN TỪNG FILE")
    print("="*100)
    print()
    print(f"{'File':<45} {'Input':<8} {'Output':<8} {'Merged':<8} {'Giảm':<10} {'% Giảm':<10}")
    print("-"*100)
    
    all_results = []
    
    for input_file in input_files:
        output_file = output_dir / input_file.name
        
        if not output_file.exists():
            continue
        
        try:
            result = analyze_track_reduction(input_file, output_file)
            result['filename'] = input_file.name
            all_results.append(result)
            
            filename = result['filename'][:43] if len(result['filename']) > 43 else result['filename']
            print(f"{filename:<45} {result['input_tracks']:<8} {result['output_tracks']:<8} "
                  f"{result['merged_tracks']:<8} {result['merged_tracks']:<10} {result['reduction_pct']:>6.1f}%")
        except Exception as e:
            print(f"❌ Lỗi: {input_file.name}: {e}")
    
    # Tổng kết
    print("-"*100)
    total_input = sum(r['input_tracks'] for r in all_results)
    total_output = sum(r['output_tracks'] for r in all_results)
    total_merged = sum(r['merged_tracks'] for r in all_results)
    avg_reduction = sum(r['reduction_pct'] for r in all_results) / len(all_results) if all_results else 0
    
    print(f"{'TỔNG KẾT':<45} {total_input:<8} {total_output:<8} "
          f"{total_merged:<8} {total_merged:<10} {avg_reduction:>6.1f}%")
    
    print()
    print("="*100)
    print("📈 CHI TIẾT TỪNG FILE:")
    print("="*100)
    
    # Sắp xếp theo phần trăm giảm (từ cao xuống thấp)
    sorted_results = sorted(all_results, key=lambda x: x['reduction_pct'], reverse=True)
    
    for result in sorted_results:
        print(f"\n📄 {result['filename']}")
        print(f"   • Số tracks đầu vào: {result['input_tracks']}")
        print(f"   • Số tracks đầu ra: {result['output_tracks']}")
        print(f"   • Số tracks đã merge: {result['merged_tracks']}")
        print(f"   • Số nhóm merge: {result['merged_groups']}")
        print(f"   • Phần trăm giảm: {result['reduction_pct']:.1f}%")
        if result['merged_tracks'] > 0:
            print(f"   • Lý thuyết: {result['input_tracks']} → {result['theoretical_output']} tracks "
                  f"(giảm {result['merged_tracks']} tracks)")
        else:
            print(f"   • ⚠️  Không có tracks nào được merge")
    
    print()
    print("="*100)
    print("📊 THỐNG KÊ TỔNG QUAN:")
    print("="*100)
    print(f"   Tổng số tracks đầu vào: {total_input}")
    print(f"   Tổng số tracks đầu ra: {total_output}")
    print(f"   Tổng số tracks đã merge: {total_merged}")
    print(f"   Tỷ lệ tracks được merge: {total_merged/total_input*100:.1f}%")
    print(f"   Phần trăm giảm trung bình: {avg_reduction:.1f}%")
    print(f"   Files có tracks được merge: {sum(1 for r in all_results if r['merged_tracks'] > 0)}/{len(all_results)}")
    
    # Top files
    print()
    print("="*100)
    print("🏆 TOP 5 FILES CÓ PHẦN TRĂM GIẢM CAO NHẤT:")
    print("="*100)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result['filename']}: {result['reduction_pct']:.1f}% "
              f"({result['merged_tracks']}/{result['input_tracks']} tracks)")


if __name__ == '__main__':
    main()
