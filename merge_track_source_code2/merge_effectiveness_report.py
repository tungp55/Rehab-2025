#!/usr/bin/env python3
"""
Generate a comprehensive report on merge_tracks effectiveness.
"""

from pathlib import Path
from collections import defaultdict

def analyze_track_mapping(input_file, output_file):
    """Analyze track mapping and merge statistics."""
    
    # Read input file
    input_tracks = defaultdict(set)  # track_id -> set of frame_ids
    input_track_frames = defaultdict(int)  # track_id -> frame count
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    input_tracks[track_id].add(frame_id)
                    input_track_frames[track_id] += 1
                except:
                    pass
    
    # Read output file
    output_tracks = defaultdict(set)
    output_track_frames = defaultdict(int)
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    output_tracks[track_id].add(frame_id)
                    output_track_frames[track_id] += 1
                except:
                    pass
    
    # Find mapping
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
    
    # Find merged tracks
    output_to_inputs = defaultdict(list)
    for in_track, out_track in input_to_output.items():
        output_to_inputs[out_track].append(in_track)
    
    merged_tracks = {out: ins for out, ins in output_to_inputs.items() if len(ins) > 1}
    
    # Calculate statistics
    total_merged_tracks = sum(len(ins) - 1 for ins in merged_tracks.values())  # -1 because one remains
    total_frames_in_merged = sum(sum(input_track_frames[t] for t in ins) for ins in merged_tracks.values())
    
    return {
        'input_tracks': len(input_tracks),
        'output_tracks': len(output_tracks),
        'merged_groups': len(merged_tracks),
        'total_merged_tracks': total_merged_tracks,
        'total_frames_in_merged': total_frames_in_merged,
        'merged_details': merged_tracks
    }


def main():
    input_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt")
    output_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt_merge_org")
    
    input_files = sorted(input_dir.glob("*.txt"))
    
    print("="*80)
    print("📊 BÁO CÁO HIỆU QUẢ THUẬT TOÁN MERGE_TRACKS")
    print("="*80)
    print()
    
    all_results = []
    
    for input_file in input_files:
        output_file = output_dir / input_file.name
        if not output_file.exists():
            continue
        
        result = analyze_track_mapping(input_file, output_file)
        result['filename'] = input_file.name
        all_results.append(result)
    
    # Individual file statistics
    print("📁 THỐNG KÊ TỪNG FILE:")
    print("-"*80)
    print(f"{'File':<40} {'Input':<8} {'Output':<8} {'Merged':<8} {'Groups':<8}")
    print("-"*80)
    
    for r in all_results:
        filename = r['filename'][:38] if len(r['filename']) > 38 else r['filename']
        print(f"{filename:<40} {r['input_tracks']:<8} {r['output_tracks']:<8} "
              f"{r['total_merged_tracks']:<8} {r['merged_groups']:<8}")
    
    # Summary
    total_input = sum(r['input_tracks'] for r in all_results)
    total_output = sum(r['output_tracks'] for r in all_results)
    total_merged = sum(r['total_merged_tracks'] for r in all_results)
    total_groups = sum(r['merged_groups'] for r in all_results)
    files_with_merges = sum(1 for r in all_results if r['merged_groups'] > 0)
    total_frames_merged = sum(r['total_frames_in_merged'] for r in all_results)
    
    print()
    print("="*80)
    print("📈 TỔNG KẾT:")
    print("="*80)
    print(f"   Tổng số file xử lý: {len(all_results)}")
    print(f"   Số file có tracks được merge: {files_with_merges} ({files_with_merges/len(all_results)*100:.1f}%)")
    print()
    print(f"   Tổng số tracks đầu vào: {total_input}")
    print(f"   Tổng số tracks đầu ra: {total_output}")
    print(f"   Số tracks đã được merge: {total_merged}")
    print(f"   Tỷ lệ tracks được merge: {total_merged/total_input*100:.1f}%")
    print()
    print(f"   Số nhóm tracks được merge: {total_groups}")
    print(f"   Trung bình tracks/group: {total_merged/total_groups:.1f}" if total_groups > 0 else "   Trung bình tracks/group: 0")
    print()
    print(f"   Tổng số frames trong các tracks đã merge: {total_frames_merged:,}")
    print()
    
    # Top merge examples
    print("="*80)
    print("🏆 TOP 5 FILES CÓ NHIỀU TRACKS MERGE NHẤT:")
    print("="*80)
    sorted_results = sorted(all_results, key=lambda x: x['total_merged_tracks'], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. {r['filename']}")
        print(f"   - Input tracks: {r['input_tracks']}")
        print(f"   - Output tracks: {r['output_tracks']}")
        print(f"   - Tracks merged: {r['total_merged_tracks']}")
        print(f"   - Merge groups: {r['merged_groups']}")
        if r['merged_details']:
            print(f"   - Chi tiết merge:")
            for out_track, in_tracks in sorted(r['merged_details'].items(), key=lambda x: len(x[1]), reverse=True)[:3]:
                print(f"     • Output track {out_track}: merged {len(in_tracks)} tracks {sorted(in_tracks)[:5]}{'...' if len(in_tracks) > 5 else ''}")
    
    print()
    print("="*80)
    print("✅ KẾT LUẬN:")
    print("="*80)
    if total_merged > 0:
        print(f"   ✅ Thuật toán merge_tracks đã hoạt động hiệu quả!")
        print(f"   ✅ Đã merge thành công {total_merged} tracks từ {total_input} tracks ban đầu")
        print(f"   ✅ Tỷ lệ merge: {total_merged/total_input*100:.1f}%")
        print(f"   ✅ {files_with_merges}/{len(all_results)} files có tracks được merge")
    else:
        print(f"   ⚠️  Không có tracks nào được merge")
    print()


if __name__ == '__main__':
    main()
