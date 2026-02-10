#!/usr/bin/env python3
"""
Thống kê hiệu quả merge_tracks cho tất cả các file trong txt_v1/download.
"""

from pathlib import Path
from collections import defaultdict

def analyze_track_mapping(input_file, output_file):
    """Phân tích track mapping và tính toán thống kê."""
    
    # Đọc input file
    input_tracks = defaultdict(set)
    input_track_frames = defaultdict(int)
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    input_tracks[track_id].add(frame_id)
                    input_track_frames[track_id] += 1
                except:
                    pass
    
    # Đọc output file
    output_tracks = defaultdict(set)
    output_track_frames = defaultdict(int)
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    output_tracks[track_id].add(frame_id)
                    output_track_frames[track_id] += 1
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
    total_merged_tracks = sum(len(tracks) - 1 for tracks in merged_tracks.values())
    unmerged_tracks = num_input_tracks - sum(len(tracks) for tracks in merged_tracks.values())
    reduction_pct = (total_merged_tracks / num_input_tracks * 100) if num_input_tracks > 0 else 0
    
    return {
        'input_tracks': num_input_tracks,
        'output_tracks': num_output_tracks,
        'merged_tracks': total_merged_tracks,
        'unmerged_tracks': unmerged_tracks,
        'merged_groups': len(merged_tracks),
        'reduction_pct': reduction_pct,
        'merged_details': merged_tracks
    }


def main():
    base_dir = Path("/home/vuhai/Rehab-Tung/txt_v1/download")
    
    # Tìm tất cả các thư mục con
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    print("="*100)
    print("📊 THỐNG KÊ HIỆU QUẢ MERGE_TRACKS CHO TXT_V1/DOWNLOAD")
    print("="*100)
    print()
    
    all_results = []
    
    for subdir in subdirs:
        print(f"\n{'='*100}")
        print(f"📂 THỐNG KÊ CHO: {subdir.name.upper()}")
        print(f"{'='*100}")
        
        txt_dir = subdir / "txt"
        merge_txt_dir = subdir / "merge_txt"
        
        if not txt_dir.exists() or not merge_txt_dir.exists():
            print(f"⚠️  Thư mục không tồn tại, bỏ qua...")
            continue
        
        txt_files = sorted(txt_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"⚠️  Không có file txt, bỏ qua...")
            continue
        
        print(f"\n📁 Tổng số files: {len(txt_files)}")
        print(f"{'File':<50} {'Input':<8} {'Output':<8} {'Merged':<8} {'% Giảm':<10}")
        print("-"*100)
        
        subdir_results = []
        
        for txt_file in txt_files:
            merge_file = merge_txt_dir / txt_file.name
            
            if not merge_file.exists():
                continue
            
            try:
                result = analyze_track_mapping(txt_file, merge_file)
                result['filename'] = txt_file.name
                result['tracker'] = subdir.name
                subdir_results.append(result)
                all_results.append(result)
                
                filename = result['filename'][:48] if len(result['filename']) > 48 else result['filename']
                print(f"{filename:<50} {result['input_tracks']:<8} {result['output_tracks']:<8} "
                      f"{result['merged_tracks']:<8} {result['reduction_pct']:>6.1f}%")
            except Exception as e:
                print(f"❌ Lỗi: {txt_file.name}: {e}")
        
        # Tổng kết cho subdirectory
        if subdir_results:
            total_input = sum(r['input_tracks'] for r in subdir_results)
            total_output = sum(r['output_tracks'] for r in subdir_results)
            total_merged = sum(r['merged_tracks'] for r in subdir_results)
            avg_reduction = sum(r['reduction_pct'] for r in subdir_results) / len(subdir_results)
            files_with_merge = sum(1 for r in subdir_results if r['merged_tracks'] > 0)
            
            print("-"*100)
            print(f"{'TỔNG KẾT':<50} {total_input:<8} {total_output:<8} {total_merged:<8} {avg_reduction:>6.1f}%")
            print(f"\n   • Files có tracks được merge: {files_with_merge}/{len(subdir_results)}")
            print(f"   • Tỷ lệ tracks được merge: {total_merged/total_input*100:.1f}%")
    
    # Tổng kết tổng thể
    if all_results:
        print(f"\n\n{'='*100}")
        print(f"📊 TỔNG KẾT TỔNG THỂ")
        print(f"{'='*100}")
        
        total_input = sum(r['input_tracks'] for r in all_results)
        total_output = sum(r['output_tracks'] for r in all_results)
        total_merged = sum(r['merged_tracks'] for r in all_results)
        avg_reduction = sum(r['reduction_pct'] for r in all_results) / len(all_results)
        files_with_merge = sum(1 for r in all_results if r['merged_tracks'] > 0)
        
        print(f"\n   Tổng số files: {len(all_results)}")
        print(f"   Tổng số tracks đầu vào: {total_input}")
        print(f"   Tổng số tracks đầu ra: {total_output}")
        print(f"   Tổng số tracks đã merge: {total_merged}")
        print(f"   Tỷ lệ tracks được merge: {total_merged/total_input*100:.1f}%")
        print(f"   Phần trăm giảm trung bình: {avg_reduction:.1f}%")
        print(f"   Files có tracks được merge: {files_with_merge}/{len(all_results)} ({files_with_merge/len(all_results)*100:.1f}%)")
        
        # Thống kê theo tracker
        print(f"\n{'='*100}")
        print(f"📈 THỐNG KÊ THEO TRACKER")
        print(f"{'='*100}")
        print(f"{'Tracker':<20} {'Files':<10} {'Input':<12} {'Merged':<12} {'% Giảm TB':<12} {'Files Merge':<12}")
        print("-"*100)
        
        tracker_stats = defaultdict(lambda: {'files': 0, 'input': 0, 'merged': 0, 'reductions': []})
        for r in all_results:
            tracker = r['tracker']
            tracker_stats[tracker]['files'] += 1
            tracker_stats[tracker]['input'] += r['input_tracks']
            tracker_stats[tracker]['merged'] += r['merged_tracks']
            tracker_stats[tracker]['reductions'].append(r['reduction_pct'])
        
        for tracker in sorted(tracker_stats.keys()):
            stats = tracker_stats[tracker]
            avg_red = sum(stats['reductions']) / len(stats['reductions']) if stats['reductions'] else 0
            files_merge = sum(1 for r in all_results if r['tracker'] == tracker and r['merged_tracks'] > 0)
            print(f"{tracker:<20} {stats['files']:<10} {stats['input']:<12} {stats['merged']:<12} "
                  f"{avg_red:>6.1f}%{'':<5} {files_merge:<12}")
        
        # Top 10 files có phần trăm giảm cao nhất
        print(f"\n{'='*100}")
        print(f"🏆 TOP 10 FILES CÓ PHẦN TRĂM GIẢM CAO NHẤT")
        print(f"{'='*100}")
        sorted_results = sorted(all_results, key=lambda x: x['reduction_pct'], reverse=True)
        print(f"{'#':<5} {'Tracker':<15} {'File':<40} {'Input':<8} {'Merged':<8} {'% Giảm':<10}")
        print("-"*100)
        for i, result in enumerate(sorted_results[:10], 1):
            filename = result['filename'][:38] if len(result['filename']) > 38 else result['filename']
            print(f"{i:<5} {result['tracker']:<15} {filename:<40} {result['input_tracks']:<8} "
                  f"{result['merged_tracks']:<8} {result['reduction_pct']:>6.1f}%")


if __name__ == '__main__':
    main()
