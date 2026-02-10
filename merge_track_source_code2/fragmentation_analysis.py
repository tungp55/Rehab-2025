#!/usr/bin/env python3
"""
Phân tích song song:
- Fragmentation NỘI BỘ của tracker (không dùng GT):
    + Một track_id input bị tách thành bao nhiêu track_id output sau merge.
- Fragmentation THEO GT:
    + Một GT track_id bị gán bao nhiêu ID predict khác nhau (trước/sau merge).

Chạy trên cùng cặp:
- GT:   txt_v1/groundtruth
- TXT:  txt_v1/download/<tracker>/txt
- MERGE:txt_v1/download/<tracker>/merge_txt
"""

from pathlib import Path
from collections import defaultdict

from evaluate_txt_files import read_txt_file, bb_intersection_over_union


def analyze_internal_fragmentation(input_file: Path, output_file: Path):
    """
    Đo fragmentation NỘI BỘ của tracker (không dùng GT):
    - Đọc file input & output (merge).
    - Với mỗi input track_id, xem nó map sang bao nhiêu output track_id (theo frame overlap).
    """
    input_tracks = defaultdict(set)   # track_id -> set(frame_id)
    output_tracks = defaultdict(set)  # track_id -> set(frame_id)

    with input_file.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                frame_id = int(float(parts[0].strip()))
                track_id = int(float(parts[1].strip()))
            except ValueError:
                continue
            input_tracks[track_id].add(frame_id)

    with output_file.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                frame_id = int(float(parts[0].strip()))
                track_id = int(float(parts[1].strip()))
            except ValueError:
                continue
            output_tracks[track_id].add(frame_id)

    # Map input track -> các output track có ≥ 1 frame overlap
    input_to_outputs = defaultdict(set)
    for in_id, in_frames in input_tracks.items():
        for out_id, out_frames in output_tracks.items():
            if in_frames & out_frames:
                input_to_outputs[in_id].add(out_id)

    # Thống kê
    num_input_tracks = len(input_tracks)
    # Số track input bị "vỡ" thành >1 ID output
    fragmented_input_tracks = [tid for tid, outs in input_to_outputs.items() if len(outs) > 1]
    avg_outputs_per_input = (
        sum(len(outs) for outs in input_to_outputs.values()) / max(1, len(input_to_outputs))
        if input_to_outputs else 0.0
    )

    return {
        "num_input_tracks": num_input_tracks,
        "num_fragmented_input_tracks": len(fragmented_input_tracks),
        "avg_outputs_per_input": avg_outputs_per_input,
    }


def analyze_gt_fragmentation(gt_file: Path, pred_file: Path, iou_thresh: float = 0.5):
    """
    Đo fragmentation THEO GT:
    - Mỗi GT track_id → bao nhiêu ID predict khác nhau trong suốt video.
    - Dựa trên matching theo IoU (giống evaluate_txt_files).
    """
    gt_data = read_txt_file(str(gt_file))
    pred_data = read_txt_file(str(pred_file))

    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))

    gt_to_pred_ids = defaultdict(set)  # gt_id -> set(pred_ids mà nó từng match)

    for frame_id in all_frames:
        gt_boxes = gt_data.get(frame_id, [])
        pred_boxes = pred_data.get(frame_id, [])
        if not gt_boxes or not pred_boxes:
            continue

        # Tính IoU giữa tất cả cặp
        matches = []
        for i, (gt_id, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            gt_box = [gx1, gy1, gx2, gy2]
            for j, (pr_id, px1, py1, px2, py2) in enumerate(pred_boxes):
                pred_box = [px1, py1, px2, py2]
                iou = bb_intersection_over_union(gt_box, pred_box)
                if iou >= iou_thresh:
                    matches.append((iou, i, j, gt_id, pr_id))

        if not matches:
            continue

        matches.sort(reverse=True, key=lambda x: x[0])
        used_gt = set()
        used_pred = set()

        for iou, i, j, gt_id, pr_id in matches:
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            gt_to_pred_ids[gt_id].add(pr_id)

    num_gt_tracks = len(gt_to_pred_ids)
    fragmented_gt_tracks = [gid for gid, preds in gt_to_pred_ids.items() if len(preds) > 1]
    avg_pred_ids_per_gt = (
        sum(len(preds) for preds in gt_to_pred_ids.values()) / max(1, len(gt_to_pred_ids))
        if gt_to_pred_ids else 0.0
    )

    return {
        "num_gt_tracks": num_gt_tracks,
        "num_fragmented_gt_tracks": len(fragmented_gt_tracks),
        "avg_pred_ids_per_gt": avg_pred_ids_per_gt,
    }


def main():
    """
    Chạy thử cho 1 tracker (ví dụ bytetrack) để so sánh:
    - Fragmentation nội bộ: TXT vs MERGE_TXT
    - Fragmentation theo GT: TXT vs MERGE_TXT
    """
    root = Path("/home/vuhai/Rehab-Tung/txt_v1")
    gt_dir = root / "groundtruth"
    tracker = "bytetrack"  # bạn có thể đổi thành botsort/deepsort/... nếu muốn
    txt_dir = root / "download" / tracker / "txt"
    merge_dir = root / "download" / tracker / "merge_txt"

    print(f"=== PHÂN TÍCH FRAGMENTATION - TRACKER: {tracker} ===\n")

    # Lấy một vài file để demo
    for gt_file in sorted(gt_dir.glob("*_bytetrack_merged.txt"))[:5]:
        base_name = gt_file.name.replace("_merged", "")
        txt_file = txt_dir / base_name
        merge_file = merge_dir / base_name

        if not txt_file.exists() or not merge_file.exists():
            continue

        print(f"File: {base_name}")

        # Fragmentation nội bộ (input txt vs merge_txt)
        internal_txt = analyze_internal_fragmentation(txt_file, merge_file)

        # Fragmentation theo GT
        gt_frag_txt = analyze_gt_fragmentation(gt_file, txt_file)
        gt_frag_merge = analyze_gt_fragmentation(gt_file, merge_file)

        print("  - Nội bộ (tracker tự thân, không dùng GT):")
        print(
            f"    Số track input: {internal_txt['num_input_tracks']}, "
            f"số track input bị tách >1 ID output: {internal_txt['num_fragmented_input_tracks']}, "
            f"avg output IDs / input track: {internal_txt['avg_outputs_per_input']:.2f}"
        )

        print("  - Theo GT (GT_ID → các ID predict khác nhau):")
        print(
            f"    TXT:    #GT={gt_frag_txt['num_gt_tracks']}, "
            f"GT bị vỡ (>1 ID)={gt_frag_txt['num_fragmented_gt_tracks']}, "
            f"avg IDs/GT={gt_frag_txt['avg_pred_ids_per_gt']:.2f}"
        )
        print(
            f"    MERGE:  #GT={gt_frag_merge['num_gt_tracks']}, "
            f"GT bị vỡ (>1 ID)={gt_frag_merge['num_fragmented_gt_tracks']}, "
            f"avg IDs/GT={gt_frag_merge['avg_pred_ids_per_gt']:.2f}"
        )
        print()


if __name__ == "__main__":
    main()

