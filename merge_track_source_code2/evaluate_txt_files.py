#!/usr/bin/env python3
"""
Đánh giá tracking results dựa trên txt files.
So sánh predicted tracks với groundtruth tracks.
"""

import os
import numpy as np
import math
from pathlib import Path
from collections import defaultdict


def bb_intersection_over_union(boxA, boxB):
    """Tính IoU giữa 2 bounding boxes."""
    # boxA và boxB: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if boxAArea + boxBArea - interArea == 0:
        return 0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def read_txt_file(file_path):
    """
    Đọc file txt và trả về dictionary:
    {frame_id: [(track_id, x1, y1, x2, y2), ...]}
    """
    data = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 6:
                continue
            
            try:
                frame_id = int(float(parts[0].strip()))
                track_id = int(float(parts[1].strip()))
                x1 = int(float(parts[2].strip()))
                y1 = int(float(parts[3].strip()))
                width = int(float(parts[4].strip()))
                height = int(float(parts[5].strip()))
                
                x2 = x1 + width
                y2 = y1 + height
                
                data[frame_id].append((track_id, x1, y1, x2, y2))
            except (ValueError, IndexError) as e:
                continue
    
    return data


# Hàm calc_conditions đã được thay thế bằng logic trong evaluate_file
# Giữ lại để tương thích nếu có code khác sử dụng
def calc_conditions(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Tính TP, FP, FN (không tính ID switches ở đây, sẽ tính trong evaluate_file).
    
    Args:
        gt_boxes: list of (track_id, x1, y1, x2, y2)
        pred_boxes: list of (track_id, x1, y1, x2, y2)
        iou_thresh: IoU threshold
    
    Returns:
        TP, FP, FN, [] (id_switches được tính trong evaluate_file)
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0, []
    
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []
    
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []
    
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    pred_matched = np.zeros(len(pred_boxes), dtype=bool)
    
    TP = 0
    FP = 0
    FN = 0
    
    # Tìm matches dựa trên IoU (sắp xếp theo IoU)
    matches = []
    for i, (gt_track_id, gt_x1, gt_y1, gt_x2, gt_y2) in enumerate(gt_boxes):
        gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]
        for j, (pred_track_id, pred_x1, pred_y1, pred_x2, pred_y2) in enumerate(pred_boxes):
            pred_box = [pred_x1, pred_y1, pred_x2, pred_y2]
            iou = bb_intersection_over_union(gt_box, pred_box)
            if iou >= iou_thresh:
                matches.append((iou, i, j))
    
    # Sắp xếp theo IoU giảm dần
    matches.sort(reverse=True, key=lambda x: x[0])
    
    # Match từ IoU cao nhất
    for iou, i, j in matches:
        if gt_matched[i] or pred_matched[j]:
            continue
        gt_matched[i] = True
        pred_matched[j] = True
        TP += 1
    
    FN = np.sum(~gt_matched)
    FP = np.sum(~pred_matched)
    
    return TP, FP, FN, []


def _align_pred_ids_first_frame(gt_data, pred_data, iou_thresh=0.5):
    """
    Re-ID: ép cho ID của predict khớp với GT ở frame đầu (hoặc frame đầu có overlap).
    
    - Lấy frame nhỏ nhất mà cả GT và predict đều có box.
    - Match GT ↔ predict bằng IoU (greedy, 1-1).
    - Tạo mapping: pred_track_id_old -> gt_track_id.
    - Áp dụng mapping này cho TOÀN BỘ pred_data.
    """
    if not gt_data or not pred_data:
        return pred_data
    
    common_frames = sorted(set(gt_data.keys()) & set(pred_data.keys()))
    if not common_frames:
        return pred_data
    
    first_frame = common_frames[0]
    gt_boxes = gt_data.get(first_frame, [])
    pred_boxes = pred_data.get(first_frame, [])
    
    if not gt_boxes or not pred_boxes:
        return pred_data
    
    # Chuẩn bị IoU giữa mọi cặp (gt, pred)
    candidates = []
    for i, (gt_id, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
        gt_box = [gx1, gy1, gx2, gy2]
        for j, (pr_id, px1, py1, px2, py2) in enumerate(pred_boxes):
            pred_box = [px1, py1, px2, py2]
            iou = bb_intersection_over_union(gt_box, pred_box)
            if iou >= iou_thresh:
                candidates.append((iou, i, j, gt_id, pr_id))
    
    if not candidates:
        return pred_data
    
    # Greedy match theo IoU giảm dần
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_gt = set()
    used_pred = set()
    mapping = {}  # pred_id_old -> gt_id
    
    for iou, i, j, gt_id, pr_id in candidates:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        mapping[pr_id] = gt_id
    
    if not mapping:
        return pred_data
    
    # Áp dụng mapping cho toàn bộ pred_data
    new_pred_data = {}
    for frame_id, boxes in pred_data.items():
        new_boxes = []
        for (track_id, x1, y1, x2, y2) in boxes:
            new_id = mapping.get(track_id, track_id)
            new_boxes.append((new_id, x1, y1, x2, y2))
        new_pred_data[frame_id] = new_boxes
    
    return new_pred_data


def evaluate_file(gt_file, pred_file, iou_thresh=0.5, align_first_frame_ids=True):
    """
    Đánh giá một cặp file GT và predicted.
    
    Returns:
        dict với các metrics
    """
    gt_data = read_txt_file(gt_file)
    pred_data = read_txt_file(pred_file)
    
    # Tuỳ chọn: re-ID để ID ở frame đầu khớp GT
    if align_first_frame_ids:
        pred_data = _align_pred_ids_first_frame(gt_data, pred_data, iou_thresh=iou_thresh)
    
    # Lấy tất cả frame_ids
    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
    
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_GT = 0
    overlap_sum = 0
    num_matches = 0
    
    gt_track_counts = defaultdict(int)
    pred_track_counts = defaultdict(int)
    
    # Theo dõi mapping GT track -> Predicted track qua các frames để tính ID switches
    # ID switch: khi một GT track được match với predicted track KHÁC với frame trước
    # Quan trọng: Tính ID switch dựa trên spatial continuity, không phải track_id
    # Lưu cả track_id và box position để so sánh continuity
    last_frame_matches = {}  # {gt_track_id: (pred_track_id, pred_box)} từ frame trước
    id_switch_count = 0
    
    for frame_id in all_frames:
        gt_boxes = gt_data.get(frame_id, [])
        pred_boxes = pred_data.get(frame_id, [])
        
        total_GT += len(gt_boxes)
        
        # Đếm tracks
        for track_id, _, _, _, _ in gt_boxes:
            gt_track_counts[track_id] += 1
        for track_id, _, _, _, _ in pred_boxes:
            pred_track_counts[track_id] += 1
        
        # Tìm matches dựa trên IoU và theo dõi ID switches
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)
        
        # Sắp xếp để match các boxes có IoU cao nhất trước
        matches = []
        for i, (gt_track_id, gt_x1, gt_y1, gt_x2, gt_y2) in enumerate(gt_boxes):
            gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]
            for j, (pred_track_id, pred_x1, pred_y1, pred_x2, pred_y2) in enumerate(pred_boxes):
                pred_box = [pred_x1, pred_y1, pred_x2, pred_y2]
                iou = bb_intersection_over_union(gt_box, pred_box)
                if iou >= iou_thresh:
                    matches.append((iou, i, j, gt_track_id, pred_track_id))
        
        # Sắp xếp theo IoU giảm dần
        matches.sort(reverse=True, key=lambda x: x[0])
        
        # Match từ IoU cao nhất
        # Lưu mapping: gt_track_id -> (pred_track_id, pred_box) để theo dõi continuity
        current_frame_matches = {}  # {gt_track_id: (pred_track_id, pred_box)} cho frame hiện tại
        
        for iou, i, j, gt_track_id, pred_track_id in matches:
            if gt_matched[i] or pred_matched[j]:
                continue
            
            gt_matched[i] = True
            pred_matched[j] = True
            total_TP += 1
            
            # Lưu match cho frame hiện tại (cả track_id và box position)
            pred_box = [pred_boxes[j][1], pred_boxes[j][2], pred_boxes[j][3], pred_boxes[j][4]]
            current_frame_matches[gt_track_id] = (pred_track_id, pred_box)
            
            # Tính overlap
            gt_box = [gt_boxes[i][1], gt_boxes[i][2], gt_boxes[i][3], gt_boxes[i][4]]
            overlap_sum += 1 - iou
            num_matches += 1
        
        # Kiểm tra ID switches: so sánh với frame trước dựa trên spatial continuity
        # ID switch xảy ra khi:
        # 1. GT track được match với predicted track có track_id KHÁC với frame trước
        # 2. VÀ spatial position của predicted track (so với GT) thay đổi đáng kể
        # Điều này đảm bảo rằng chỉ tính ID switch khi thực sự có sự thay đổi object, không phải do merge
        for gt_track_id, (pred_track_id, pred_box) in current_frame_matches.items():
            if gt_track_id in last_frame_matches:
                prev_pred_track_id, prev_pred_box = last_frame_matches[gt_track_id]
                
                # Kiểm tra xem có phải cùng một predicted track (dựa trên track_id) không
                if prev_pred_track_id != pred_track_id:
                    # Track_id khác nhau - kiểm tra spatial continuity
                    # So sánh predicted box hiện tại với predicted box ở frame trước
                    # Nếu IoU cao → có thể là cùng object (sau merge), không tính ID switch
                    # Nếu IoU thấp → thực sự là ID switch (object khác)
                    box_iou = bb_intersection_over_union(prev_pred_box, pred_box)
                    
                    # Threshold: Nếu IoU < 0.3, đây là ID switch thực sự
                    # Nếu IoU >= 0.3, có thể là cùng object sau merge (track_id thay đổi nhưng vị trí tương tự)
                    # Giảm threshold xuống 0.3 để tránh tính nhầm khi merge thay đổi track_id
                    if box_iou < 0.3:
                        # ID switch: GT track này được match với predicted track khác
                        # và spatial position thay đổi đáng kể (IoU < 0.3)
                        id_switch_count += 1
                    # Nếu box_iou >= 0.3, không tính là ID switch vì có thể là cùng object sau merge
        
        # Cập nhật last_frame_matches cho frame tiếp theo
        # Chỉ giữ lại các GT tracks đã được match trong frame hiện tại
        last_frame_matches = current_frame_matches.copy()
        
        # FN: GT boxes không được match
        total_FN += np.sum(~gt_matched)
        
        # FP: Predicted boxes không được match
        total_FP += np.sum(~pred_matched)
    
    # Tính MT (Mostly Tracked) và ML (Mostly Lost)
    gt_unique_tracks = set(gt_track_counts.keys())
    MT = 0
    ML = 0
    
    for track_id in gt_unique_tracks:
        gt_count = gt_track_counts[track_id]
        pred_count = pred_track_counts[track_id]
        
        if gt_count > 0:
            ratio = pred_count / gt_count
            if ratio >= 0.8:
                MT += 1
            elif ratio <= 0.2:
                ML += 1
    
    # Tính MOTA và MOTP
    if total_GT > 0:
        MOTA = (1 - (total_FN + total_FP + id_switch_count) / total_GT) * 100
    else:
        MOTA = 0
    
    if num_matches > 0:
        MOTP = (overlap_sum / num_matches) * 100
    else:
        MOTP = 0
    
    return {
        'MOTA': round(MOTA, 2),
        'MOTP': round(MOTP, 2),
        'TP': total_TP,
        'FP': total_FP,
        'FN': total_FN,
        'GT': total_GT,
        'ID_switches': id_switch_count,
        'MT': MT,
        'ML': ML,
        'num_matches': num_matches,
        'overlap_sum': overlap_sum,
        'gt_tracks': len(gt_unique_tracks),
        'pred_tracks': len(set(pred_track_counts.keys()))
    }


def find_matching_files(gt_dir, pred_dir, tracker_name):
    """
    Tìm các cặp file matching giữa GT và predicted.
    """
    gt_files = {}
    pred_files = {}
    
    # Đọc GT files
    for file in os.listdir(gt_dir):
        if file.endswith('.txt'):
            # Extract base name (remove tracker suffix if exists)
            base_name = file.replace('_merged.txt', '').replace('_' + tracker_name + '.txt', '')
            gt_files[base_name] = os.path.join(gt_dir, file)
    
    # Đọc predicted files
    for subdir in os.listdir(pred_dir):
        subdir_path = os.path.join(pred_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        merge_txt_dir = os.path.join(subdir_path, 'merge_txt')
        if not os.path.exists(merge_txt_dir):
            continue
        
        for file in os.listdir(merge_txt_dir):
            if file.endswith('.txt'):
                # Extract base name
                base_name = file.replace('_' + subdir + '.txt', '')
                if base_name not in pred_files:
                    pred_files[base_name] = {}
                pred_files[base_name][subdir] = os.path.join(merge_txt_dir, file)
    
    return gt_files, pred_files


def evaluate_tracker_directory(gt_files, pred_dir, subdir, folder_name="merge_txt"):
    """
    Đánh giá một thư mục cụ thể (txt hoặc merge_txt).
    
    Args:
        gt_files: dict of groundtruth files
        pred_dir: predicted directory
        subdir: tracker subdirectory name
        folder_name: "txt" or "merge_txt"
    
    Returns:
        list of results
    """
    subdir_path = os.path.join(pred_dir, subdir)
    if not os.path.isdir(subdir_path):
        return []
    
    pred_folder = os.path.join(subdir_path, folder_name)
    if not os.path.exists(pred_folder):
        return []
    
    pred_files = {}
    for file in os.listdir(pred_folder):
        if file.endswith('.txt'):
            # Extract base name: GH010354_5_17718_19366_botsort.txt -> GH010354_5_17718_19366
            base_name = file.replace('_' + subdir + '.txt', '')
            pred_files[base_name] = os.path.join(pred_folder, file)
    
    tracker_results = []
    
    for base_name in sorted(gt_files.keys()):
        if base_name not in pred_files:
            continue
        
        gt_file = gt_files[base_name]
        pred_file = pred_files[base_name]
        
        try:
            result = evaluate_file(gt_file, pred_file, iou_thresh=0.5)
            result['file'] = base_name
            result['tracker'] = subdir
            result['folder'] = folder_name
            tracker_results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {base_name} ({folder_name}): {e}")
    
    return tracker_results


def main():
    """Main function để đánh giá tất cả các files."""
    gt_dir = "/home/vuhai/Rehab-Tung/txt_v1/groundtruth"
    pred_dir = "/home/vuhai/Rehab-Tung/txt_v1/download"
    
    print("="*100)
    print("📊 ĐÁNH GIÁ TRACKING RESULTS")
    print("="*100)
    print(f"Groundtruth directory: {gt_dir}")
    print(f"Predicted directory: {pred_dir}")
    print()
    
    # Tìm các file GT
    gt_files = {}
    for file in os.listdir(gt_dir):
        if file.endswith('.txt'):
            # Extract base name (remove _bytetrack_merged.txt or similar)
            base_name = file.replace('_bytetrack_merged.txt', '').replace('_merged.txt', '')
            # Try to extract video name
            parts = base_name.split('_')
            if len(parts) >= 4:
                # Format: GH010354_5_17718_19366
                base_name = '_'.join(parts[:4])
            gt_files[base_name] = os.path.join(gt_dir, file)
    
    print(f"📁 Found {len(gt_files)} groundtruth files")
    
    # Tìm tất cả các trackers
    trackers = []
    for subdir in sorted(os.listdir(pred_dir)):
        subdir_path = os.path.join(pred_dir, subdir)
        if os.path.isdir(subdir_path):
            trackers.append(subdir)
    
    # Đánh giá cho từng tracker, cả txt và merge_txt
    all_results_txt = []
    all_results_merge = []
    
    for tracker in trackers:
        print(f"\n{'='*100}")
        print(f"📂 Evaluating tracker: {tracker.upper()}")
        print(f"{'='*100}")
        
        # Đánh giá thư mục txt (gốc)
        results_txt = evaluate_tracker_directory(gt_files, pred_dir, tracker, "txt")
        all_results_txt.extend(results_txt)
        
        # Đánh giá thư mục merge_txt (sau merge)
        results_merge = evaluate_tracker_directory(gt_files, pred_dir, tracker, "merge_txt")
        all_results_merge.extend(results_merge)
        
        # Hiển thị kết quả cho txt
        if results_txt:
            print(f"\n📊 KẾT QUẢ: TXT (GỐC - TRƯỚC MERGE)")
            print(f"{'File':<50} {'MOTA':<10} {'MOTP':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'ID_SW':<8}")
            print("-"*100)
            
            for result in sorted(results_txt, key=lambda x: x['file']):
                filename = result['file'][:48] if len(result['file']) > 48 else result['file']
                print(f"{filename:<50} {result['MOTA']:>6.2f}%   {result['MOTP']:>6.2f}%   "
                      f"{result['TP']:<8} {result['FP']:<8} {result['FN']:<8} {result['ID_switches']:<8}")
            
            # Tổng kết txt
            avg_mota = sum(r['MOTA'] for r in results_txt) / len(results_txt)
            avg_motp = sum(r['MOTP'] for r in results_txt) / len(results_txt)
            total_TP = sum(r['TP'] for r in results_txt)
            total_FP = sum(r['FP'] for r in results_txt)
            total_FN = sum(r['FN'] for r in results_txt)
            total_id_sw = sum(r['ID_switches'] for r in results_txt)
            
            print("-"*100)
            print(f"{'TỔNG KẾT (TXT)':<50} {avg_mota:>6.2f}%   {avg_motp:>6.2f}%   "
                  f"{total_TP:<8} {total_FP:<8} {total_FN:<8} {total_id_sw:<8}")
        
        # Hiển thị kết quả cho merge_txt
        if results_merge:
            print(f"\n📊 KẾT QUẢ: MERGE_TXT (SAU MERGE)")
            print(f"{'File':<50} {'MOTA':<10} {'MOTP':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'ID_SW':<8}")
            print("-"*100)
            
            for result in sorted(results_merge, key=lambda x: x['file']):
                filename = result['file'][:48] if len(result['file']) > 48 else result['file']
                print(f"{filename:<50} {result['MOTA']:>6.2f}%   {result['MOTP']:>6.2f}%   "
                      f"{result['TP']:<8} {result['FP']:<8} {result['FN']:<8} {result['ID_switches']:<8}")
            
            # Tổng kết merge_txt
            avg_mota = sum(r['MOTA'] for r in results_merge) / len(results_merge)
            avg_motp = sum(r['MOTP'] for r in results_merge) / len(results_merge)
            total_TP = sum(r['TP'] for r in results_merge)
            total_FP = sum(r['FP'] for r in results_merge)
            total_FN = sum(r['FN'] for r in results_merge)
            total_id_sw = sum(r['ID_switches'] for r in results_merge)
            
            print("-"*100)
            print(f"{'TỔNG KẾT (MERGE_TXT)':<50} {avg_mota:>6.2f}%   {avg_motp:>6.2f}%   "
                  f"{total_TP:<8} {total_FP:<8} {total_FN:<8} {total_id_sw:<8}")
    
    # Tổng kết tổng thể - BẢNG 1: TXT (GỐC)
    if all_results_txt:
        print(f"\n\n{'='*100}")
        print(f"📊 BẢNG 1: TỔNG KẾT - TXT (GỐC - TRƯỚC MERGE)")
        print(f"{'='*100}")
        
        # Group by tracker
        tracker_stats = defaultdict(lambda: {'results': [], 'count': 0})
        for r in all_results_txt:
            tracker_stats[r['tracker']]['results'].append(r)
        
        print(f"\n{'Tracker':<20} {'Files':<10} {'Avg MOTA':<12} {'Avg MOTP':<12} {'Total TP':<12} {'Total FP':<12} {'Total FN':<12} {'ID_SW':<10}")
        print("-"*100)
        
        for tracker in sorted(tracker_stats.keys()):
            results = tracker_stats[tracker]['results']
            avg_mota = sum(r['MOTA'] for r in results) / len(results)
            avg_motp = sum(r['MOTP'] for r in results) / len(results)
            total_TP = sum(r['TP'] for r in results)
            total_FP = sum(r['FP'] for r in results)
            total_FN = sum(r['FN'] for r in results)
            total_id_sw = sum(r['ID_switches'] for r in results)
            
            print(f"{tracker:<20} {len(results):<10} {avg_mota:>8.2f}%    {avg_motp:>8.2f}%    "
                  f"{total_TP:<12} {total_FP:<12} {total_FN:<12} {total_id_sw:<10}")
    
    # Tổng kết tổng thể - BẢNG 2: MERGE_TXT (SAU MERGE)
    if all_results_merge:
        print(f"\n\n{'='*100}")
        print(f"📊 BẢNG 2: TỔNG KẾT - MERGE_TXT (SAU MERGE)")
        print(f"{'='*100}")
        
        # Group by tracker
        tracker_stats = defaultdict(lambda: {'results': [], 'count': 0})
        for r in all_results_merge:
            tracker_stats[r['tracker']]['results'].append(r)
        
        print(f"\n{'Tracker':<20} {'Files':<10} {'Avg MOTA':<12} {'Avg MOTP':<12} {'Total TP':<12} {'Total FP':<12} {'Total FN':<12} {'ID_SW':<10}")
        print("-"*100)
        
        for tracker in sorted(tracker_stats.keys()):
            results = tracker_stats[tracker]['results']
            avg_mota = sum(r['MOTA'] for r in results) / len(results)
            avg_motp = sum(r['MOTP'] for r in results) / len(results)
            total_TP = sum(r['TP'] for r in results)
            total_FP = sum(r['FP'] for r in results)
            total_FN = sum(r['FN'] for r in results)
            total_id_sw = sum(r['ID_switches'] for r in results)
            
            print(f"{tracker:<20} {len(results):<10} {avg_mota:>8.2f}%    {avg_motp:>8.2f}%    "
                  f"{total_TP:<12} {total_FP:<12} {total_FN:<12} {total_id_sw:<10}")
    
    # So sánh cải thiện
    if all_results_txt and all_results_merge:
        print(f"\n\n{'='*100}")
        print(f"📈 SO SÁNH: CẢI THIỆN SAU KHI MERGE")
        print(f"{'='*100}")
        
        # Group by tracker
        txt_by_tracker = defaultdict(list)
        merge_by_tracker = defaultdict(list)
        
        for r in all_results_txt:
            txt_by_tracker[r['tracker']].append(r)
        for r in all_results_merge:
            merge_by_tracker[r['tracker']].append(r)
        
        print(f"\n{'Tracker':<20} {'MOTA (TXT)':<15} {'MOTA (Merge)':<15} {'Cải thiện':<15} {'MOTP (TXT)':<15} {'MOTP (Merge)':<15}")
        print("-"*100)
        
        for tracker in sorted(set(txt_by_tracker.keys()) | set(merge_by_tracker.keys())):
            txt_results = txt_by_tracker.get(tracker, [])
            merge_results = merge_by_tracker.get(tracker, [])
            
            if txt_results and merge_results:
                txt_mota = sum(r['MOTA'] for r in txt_results) / len(txt_results)
                merge_mota = sum(r['MOTA'] for r in merge_results) / len(merge_results)
                improvement = merge_mota - txt_mota
                
                txt_motp = sum(r['MOTP'] for r in txt_results) / len(txt_results)
                merge_motp = sum(r['MOTP'] for r in merge_results) / len(merge_results)
                
                print(f"{tracker:<20} {txt_mota:>10.2f}%    {merge_mota:>10.2f}%    "
                      f"{improvement:>+10.2f}%    {txt_motp:>10.2f}%    {merge_motp:>10.2f}%")
        
        # Bảng riêng cho ID switches để thể hiện rõ sự thay đổi
        print(f"\n\n{'='*100}")
        print(f"🔄 SO SÁNH ID SWITCHES: CẢI THIỆN SAU KHI MERGE")
        print(f"{'='*100}")
        print(f"\n{'Tracker':<20} {'ID_SW (TXT)':<15} {'ID_SW (Merge)':<15} {'Thay đổi':<15} {'Cải thiện %':<15}")
        print("-"*100)
        
        for tracker in sorted(set(txt_by_tracker.keys()) | set(merge_by_tracker.keys())):
            txt_results = txt_by_tracker.get(tracker, [])
            merge_results = merge_by_tracker.get(tracker, [])
            
            if txt_results and merge_results:
                txt_id_sw = sum(r['ID_switches'] for r in txt_results)
                merge_id_sw = sum(r['ID_switches'] for r in merge_results)
                id_sw_change = merge_id_sw - txt_id_sw
                id_sw_improvement_pct = ((txt_id_sw - merge_id_sw) / txt_id_sw * 100) if txt_id_sw > 0 else 0
                
                print(f"{tracker:<20} {txt_id_sw:>12}    {merge_id_sw:>12}    {id_sw_change:>+12}    {id_sw_improvement_pct:>+12.1f}%")


if __name__ == '__main__':
    main()
