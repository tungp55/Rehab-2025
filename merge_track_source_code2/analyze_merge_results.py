#!/usr/bin/env python3
"""
Script to analyze the effectiveness of merge_tracks algorithm.
Compares input and output files to show:
- Number of tracks before and after merging
- Track statistics
- Merge effectiveness
"""

import os
from pathlib import Path
from collections import defaultdict, Counter


def analyze_txt_file(file_path):
    """
    Analyze a txt file and return statistics.
    
    Returns:
        dict with statistics: num_frames, num_tracks, track_lengths, etc.
    """
    stats = {
        'num_frames': 0,
        'num_tracks': set(),
        'track_lengths': defaultdict(int),
        'frame_tracks': defaultdict(set),
        'total_detections': 0
    }
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 2:
                continue
            
            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                
                stats['num_frames'] = max(stats['num_frames'], frame_id)
                stats['num_tracks'].add(track_id)
                stats['track_lengths'][track_id] += 1
                stats['frame_tracks'][frame_id].add(track_id)
                stats['total_detections'] += 1
            except (ValueError, IndexError):
                continue
    
    stats['num_tracks'] = len(stats['num_tracks'])
    stats['track_lengths'] = dict(stats['track_lengths'])
    
    # Calculate average track length
    if stats['track_lengths']:
        stats['avg_track_length'] = sum(stats['track_lengths'].values()) / len(stats['track_lengths'])
        stats['min_track_length'] = min(stats['track_lengths'].values())
        stats['max_track_length'] = max(stats['track_lengths'].values())
    else:
        stats['avg_track_length'] = 0
        stats['min_track_length'] = 0
        stats['max_track_length'] = 0
    
    # Calculate frames with multiple tracks
    stats['frames_with_multiple_tracks'] = sum(1 for tracks in stats['frame_tracks'].values() if len(tracks) > 1)
    
    return stats


def compare_files(input_file, output_file):
    """
    Compare input and output files to show merge effectiveness.
    """
    print(f"\n{'='*80}")
    print(f"File: {input_file.name}")
    print(f"{'='*80}")
    
    input_stats = analyze_txt_file(input_file)
    output_stats = analyze_txt_file(output_file)
    
    print(f"\n📊 INPUT FILE STATISTICS:")
    print(f"   Total frames: {input_stats['num_frames']}")
    print(f"   Total tracks: {input_stats['num_tracks']}")
    print(f"   Total detections: {input_stats['total_detections']}")
    print(f"   Average track length: {input_stats['avg_track_length']:.1f} frames")
    print(f"   Min track length: {input_stats['min_track_length']} frames")
    print(f"   Max track length: {input_stats['max_track_length']} frames")
    print(f"   Frames with multiple tracks: {input_stats['frames_with_multiple_tracks']}")
    
    print(f"\n📊 OUTPUT FILE STATISTICS (After Merge):")
    print(f"   Total frames: {output_stats['num_frames']}")
    print(f"   Total tracks: {output_stats['num_tracks']}")
    print(f"   Total detections: {output_stats['total_detections']}")
    print(f"   Average track length: {output_stats['avg_track_length']:.1f} frames")
    print(f"   Min track length: {output_stats['min_track_length']} frames")
    print(f"   Max track length: {output_stats['max_track_length']} frames")
    print(f"   Frames with multiple tracks: {output_stats['frames_with_multiple_tracks']}")
    
    print(f"\n📈 MERGE EFFECTIVENESS:")
    track_reduction = input_stats['num_tracks'] - output_stats['num_tracks']
    track_reduction_pct = (track_reduction / input_stats['num_tracks'] * 100) if input_stats['num_tracks'] > 0 else 0
    
    print(f"   Tracks reduced: {track_reduction} ({track_reduction_pct:.1f}%)")
    print(f"   Track reduction ratio: {input_stats['num_tracks']} → {output_stats['num_tracks']}")
    
    if output_stats['num_tracks'] > 0:
        avg_length_increase = output_stats['avg_track_length'] - input_stats['avg_track_length']
        avg_length_increase_pct = (avg_length_increase / input_stats['avg_track_length'] * 100) if input_stats['avg_track_length'] > 0 else 0
        print(f"   Average track length change: {avg_length_increase:+.1f} frames ({avg_length_increase_pct:+.1f}%)")
    
    # Show track length distribution
    print(f"\n📋 TRACK LENGTH DISTRIBUTION (Input):")
    length_dist = Counter(input_stats['track_lengths'].values())
    for length in sorted(length_dist.keys(), reverse=True)[:10]:
        count = length_dist[length]
        print(f"   {length:4d} frames: {count:3d} tracks")
    
    print(f"\n📋 TRACK LENGTH DISTRIBUTION (Output):")
    length_dist = Counter(output_stats['track_lengths'].values())
    for length in sorted(length_dist.keys(), reverse=True)[:10]:
        count = length_dist[length]
        print(f"   {length:4d} frames: {count:3d} tracks")
    
    # Check if detections are preserved
    detection_preserved = abs(input_stats['total_detections'] - output_stats['total_detections']) / input_stats['total_detections'] * 100 if input_stats['total_detections'] > 0 else 0
    print(f"\n✅ DATA PRESERVATION:")
    print(f"   Detections preserved: {abs(input_stats['total_detections'] - output_stats['total_detections'])} difference ({100-detection_preserved:.2f}%)")
    
    return {
        'input_tracks': input_stats['num_tracks'],
        'output_tracks': output_stats['num_tracks'],
        'reduction': track_reduction,
        'reduction_pct': track_reduction_pct,
        'input_avg_length': input_stats['avg_track_length'],
        'output_avg_length': output_stats['avg_track_length']
    }


def main():
    """Main function to analyze all merged files."""
    input_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt")
    output_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt_merge_org")
    
    input_files = sorted(input_dir.glob("*.txt"))
    
    if not input_files:
        print(f"❌ No input files found in {input_dir}")
        return
    
    print(f"🔍 Analyzing merge results for {len(input_files)} files...")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    
    all_results = []
    
    for input_file in input_files:
        output_file = output_dir / input_file.name
        
        if not output_file.exists():
            print(f"\n⚠️  Warning: Output file not found: {output_file.name}")
            continue
        
        try:
            result = compare_files(input_file, output_file)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error analyzing {input_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary statistics
    if all_results:
        print(f"\n\n{'='*80}")
        print(f"📊 OVERALL SUMMARY")
        print(f"{'='*80}")
        
        total_input_tracks = sum(r['input_tracks'] for r in all_results)
        total_output_tracks = sum(r['output_tracks'] for r in all_results)
        total_reduction = total_input_tracks - total_output_tracks
        total_reduction_pct = (total_reduction / total_input_tracks * 100) if total_input_tracks > 0 else 0
        avg_reduction_pct = sum(r['reduction_pct'] for r in all_results) / len(all_results)
        
        print(f"\n   Total input tracks: {total_input_tracks}")
        print(f"   Total output tracks: {total_output_tracks}")
        print(f"   Total tracks reduced: {total_reduction} ({total_reduction_pct:.1f}%)")
        print(f"   Average reduction per file: {avg_reduction_pct:.1f}%")
        
        avg_input_length = sum(r['input_avg_length'] for r in all_results) / len(all_results)
        avg_output_length = sum(r['output_avg_length'] for r in all_results) / len(all_results)
        print(f"\n   Average track length (input): {avg_input_length:.1f} frames")
        print(f"   Average track length (output): {avg_output_length:.1f} frames")
        print(f"   Average length increase: {avg_output_length - avg_input_length:+.1f} frames")
        
        print(f"\n✅ Analysis complete! Processed {len(all_results)} files.")


if __name__ == '__main__':
    main()
