#!/usr/bin/env python3
"""
Batch processing script to apply merge_tracks algorithm to all txt files
from mask rcnn and deepsort segmentation output.
"""

import os
import sys
from pathlib import Path
import networkx as nx

# Add current directory to path to import merge_tracks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merge_tracks_org import merge_tracks
from utils import write_tracks_chain_to_file


def process_txt_file(input_file, output_file, fix_tracks=[(1, 1), (2, 2)]):
    """
    Process a single txt file using merge_tracks algorithm.
    
    Args:
        input_file: Path to input txt file
        output_file: Path to output txt file
        fix_tracks: List of (hand_side, track_id) tuples. Default: [(1, 1), (2, 2)]
                    hand_side: 1 = left, 2 = right
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*60}")
    
    try:
        # Process left hand (track 1, side 1)
        print("\n[1/2] Processing left hand (track 1)...")
        merger_left = merge_tracks()
        merger_left.TRACK_DATA_FILE = str(input_file)
        merger_left.START_TRACKING_TIME_SECONDS = 0
        merger_left.STOP_TRACKING_TIME_SECONDS = 0
        merger_left.FIX_TRACK_ID = fix_tracks[0][1]  # track_id = 1
        merger_left.FIX_HAND_SIDE = fix_tracks[0][0]  # side = 1 (left)
        
        merger_left.init()
        print(f"   Found {merger_left.max_track_id} tracks")
        
        if merger_left.max_track_id < fix_tracks[0][1]:
            print(f"   ⚠️  Warning: Track {fix_tracks[0][1]} not found, skipping left hand merge")
            left_nodes = []
        else:
            graph_left = merger_left.build_graph()
            try:
                path_length_left = nx.bellman_ford_path_length(
                    graph_left, 
                    source=merger_left.FIX_TRACK_ID, 
                    target=merger_left.max_track_id + 1, 
                    weight="length"
                )
                path_nodes_left = nx.bellman_ford_path(
                    graph_left, 
                    source=merger_left.FIX_TRACK_ID, 
                    target=merger_left.max_track_id + 1, 
                    weight="length"
                )
                path_nodes_left.pop()  # Remove sink node
                print(f"   ✅ Left hand path length: {path_length_left}")
                print(f"   ✅ Left hand merged tracks: {len(path_nodes_left)} segments")
                print(f"   ✅ Track IDs: {path_nodes_left[:10]}{'...' if len(path_nodes_left) > 10 else ''}")
                left_nodes = path_nodes_left.copy()
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                print(f"   ⚠️  Warning: No path found for left hand: {e}")
                left_nodes = []
        
        # Process right hand (track 2, side 2)
        print("\n[2/2] Processing right hand (track 2)...")
        merger_right = merge_tracks()
        merger_right.TRACK_DATA_FILE = str(input_file)
        merger_right.START_TRACKING_TIME_SECONDS = 0
        merger_right.STOP_TRACKING_TIME_SECONDS = 0
        merger_right.FIX_TRACK_ID = fix_tracks[1][1]  # track_id = 2
        merger_right.FIX_HAND_SIDE = fix_tracks[1][0]  # side = 2 (right)
        
        merger_right.init()
        print(f"   Found {merger_right.max_track_id} tracks")
        
        if merger_right.max_track_id < fix_tracks[1][1]:
            print(f"   ⚠️  Warning: Track {fix_tracks[1][1]} not found, skipping right hand merge")
            right_nodes = []
        else:
            graph_right = merger_right.build_graph()
            try:
                path_length_right = nx.bellman_ford_path_length(
                    graph_right, 
                    source=merger_right.FIX_TRACK_ID, 
                    target=merger_right.max_track_id + 1, 
                    weight="length"
                )
                path_nodes_right = nx.bellman_ford_path(
                    graph_right, 
                    source=merger_right.FIX_TRACK_ID, 
                    target=merger_right.max_track_id + 1, 
                    weight="length"
                )
                path_nodes_right.pop()  # Remove sink node
                print(f"   ✅ Right hand path length: {path_length_right}")
                print(f"   ✅ Right hand merged tracks: {len(path_nodes_right)} segments")
                print(f"   ✅ Track IDs: {path_nodes_right[:10]}{'...' if len(path_nodes_right) > 10 else ''}")
                right_nodes = path_nodes_right.copy()
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                print(f"   ⚠️  Warning: No path found for right hand: {e}")
                right_nodes = []
        
        # Write merged tracks to file
        print(f"\n💾 Writing merged tracks to: {output_file}")
        # Use merger_right's frames_data as it has the complete data
        write_tracks_chain_to_file(
            merger_right.frames_data,
            [(1, left_nodes), (2, right_nodes)],
            str(output_file),
            merger_right.START_TRACKING_TIME_SECONDS,
            merger_right.STOP_TRACKING_TIME_SECONDS,
            merger_right.FRAME_RATE,
            merger_right.tracks_data
        )
        print(f"✅ Successfully saved: {output_file}")
        
        return True, None
        
    except Exception as e:
        error_msg = f"Error processing {input_file.name}: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg


def main():
    """Main function to process all txt files in input directory."""
    # Input and output directories
    input_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt")
    output_dir = Path("/home/vuhai/Rehab-Tung/txt/download/deepsort/txt_merge_org")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all txt files
    txt_files = sorted(input_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"❌ No txt files found in {input_dir}")
        return
    
    print(f"📁 Found {len(txt_files)} txt files to process")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for txt_file in txt_files:
        # Create output filename (same name as input)
        output_file = output_dir / txt_file.name
        
        success, error = process_txt_file(txt_file, output_file)
        
        if success:
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Processing Summary:")
    print(f"   ✅ Success: {success_count}/{len(txt_files)}")
    print(f"   ❌ Errors: {error_count}/{len(txt_files)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
