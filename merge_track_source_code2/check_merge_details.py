#!/usr/bin/env python3
"""
Script to check detailed merge information by comparing track IDs.
"""

from pathlib import Path
from collections import defaultdict

def analyze_track_mapping(input_file, output_file):
    """Analyze how track IDs changed from input to output."""
    
    # Read input file
    input_tracks = defaultdict(set)  # track_id -> set of frame_ids
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
    
    # Read output file
    output_tracks = defaultdict(set)  # track_id -> set of frame_ids
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
    
    # Find mapping: which input tracks map to which output tracks
    # by comparing frame_id sets
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
    
    # Find merged tracks: multiple input tracks map to same output track
    output_to_inputs = defaultdict(list)
    for in_track, out_track in input_to_output.items():
        output_to_inputs[out_track].append(in_track)
    
    merged_tracks = {out: ins for out, ins in output_to_inputs.items() if len(ins) > 1}
    
    return {
        'input_tracks': len(input_tracks),
        'output_tracks': len(output_tracks),
        'merged_count': len(merged_tracks),
        'merged_details': merged_tracks,
        'mapping': input_to_output
    }


def main():
    input_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt")
    output_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt_merge_org")
    
    input_files = sorted(input_dir.glob("*.txt"))
    
    print(f"🔍 Checking merge details for {len(input_files)} files...\n")
    
    total_merged = 0
    files_with_merges = 0
    
    for input_file in input_files:
        output_file = output_dir / input_file.name
        
        if not output_file.exists():
            continue
        
        result = analyze_track_mapping(input_file, output_file)
        
        print(f"{'='*80}")
        print(f"File: {input_file.name}")
        print(f"{'='*80}")
        print(f"Input tracks: {result['input_tracks']}")
        print(f"Output tracks: {result['output_tracks']}")
        print(f"Merged track groups: {result['merged_count']}")
        
        if result['merged_details']:
            files_with_merges += 1
            total_merged += sum(len(ins) - 1 for ins in result['merged_details'].values())  # -1 because one track remains
            print(f"\n✅ MERGED TRACKS:")
            for out_track, in_tracks in sorted(result['merged_details'].items()):
                print(f"   Output track {out_track} ← Merged from input tracks: {sorted(in_tracks)}")
        else:
            print(f"\n⚠️  No tracks were merged in this file")
            print(f"   (All tracks kept separate, only track IDs may have changed)")
        
        # Show track ID changes
        if result['mapping']:
            print(f"\n📋 Track ID Mapping (first 10):")
            for in_track, out_track in sorted(list(result['mapping'].items())[:10]):
                if in_track != out_track:
                    print(f"   Input track {in_track} → Output track {out_track}")
        
        print()
    
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Files with merged tracks: {files_with_merges}/{len(input_files)}")
    print(f"Total tracks merged: {total_merged}")
    print(f"Average merges per file: {total_merged/len(input_files):.1f}" if input_files else "N/A")


if __name__ == '__main__':
    main()
