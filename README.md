# Rehab-2025

Supplementary source code for **merge-track evaluation** on new data: hand tracking with Detectron2 + BoxMOT, merging tracks per left/right hand, and MOT evaluation (MOTA, MOTP, ID switches) before and after merge. This repository supports the paper *MergeTrack: A Graph-Based Tracking Refinement for Patient Hand Identification in Egocentric Rehabilitation Videos*.

---

## Data and video links

Evaluation data is packaged in **one folder** containing two subfolders: `test_input` and `test_output`.

- **Google Drive link (download full folder):**  
  **[https://drive.google.com/drive/folders/REPLACE_WITH_YOUR_FOLDER_ID](https://drive.google.com/drive/folders/REPLACE_WITH_YOUR_FOLDER_ID)**

### Directory structure

```
<root>/
├── test_input/          # 12 original video clips (~21,500 frames)
│   ├── GH010354_5_17718_19366.avi
│   ├── GH010358_5_16380_17200.avi
│   ├── GH010371_5_1132_5000.avi
│   ├── ...              # 12 .avi files, naming: GH0xxxxx_N_start_end.avi
│   └── GH010383_5_1483_1777.avi
│
└── test_output/         # Tracking results (before and after MergeTrack) for 7 trackers
    ├── botsort/
    │   ├── txt/         # raw tracker output (.txt)
    │   ├── videos/      # tracking videos (.avi)
    │   ├── merge_txt/   # after MergeTrack (.txt)
    │   └── merge_videos/
    ├── bytetrack/
    ├── deepsort/
    ├── hybridsort/
    ├── ocsort/
    ├── sort/
    └── strongsort/
```

- **test_input:** 12 clips extracted from rehabilitation sessions (four exercise types, nine sessions). File naming: `GH0xxxxx_N_start_end` (e.g. `GH010374_7_15760_20222`). Ground-truth: left hand (class 0), right hand (class 1).
- **test_output:** Each tracker has `txt/`, `videos/`, `merge_txt/`, `merge_videos/` for outputs before and after MergeTrack on the 12 videos.

---

## Evaluation report summary (from the paper)

Tracking is evaluated on **12 video clips** with **seven trackers** (BoT-SORT, ByteTrack, DeepSORT, HybridSort, OC-SORT, SORT, StrongSORT), **before** (raw tracker output, `txt`) and **after** applying **MergeTrack** (`merge_txt`).

### Main results (Table: tracking performance comparison)

| Method    | Version   | Num Videos | Avg MOTA (%) | Avg ID Sw. | Total ID Sw. | Avg MT | Avg ML |
|-----------|-----------|-------------|--------------|------------|--------------|--------|--------|
| botsort   | txt       | 12          | 41.25        | 7.11       | 256          | 4.17   | 0.94   |
| botsort   | merge_txt | 12          | 41.28        | 6.58       | 237          | 4.33   | 0.94   |
| bytetrack | txt       | 12          | 40.81        | 7.39       | 266          | 5.50   | 0.22   |
| bytetrack | merge_txt | 12          | 40.91        | 7.00       | 252          | 5.61   | 0.22   |
| deepsort  | txt       | 12          | 38.41        | 3.61       | 130          | 2.33   | 2.28   |
| deepsort  | merge_txt | 12          | 38.14        | 3.56       | 128          | 2.33   | 2.28   |
| hybridsort| txt       | 12          | 30.05        | 4.39       | 158          | 3.03   | 1.33   |
| hybridsort| merge_txt| 12          | 30.15        | 3.67        | 132          | 3.06   | 1.36   |
| ocsort    | txt       | 12          | 37.92        | 8.86        | 319          | 2.42   | 1.67   |
| ocsort    | merge_txt| 12          | 37.91        | 6.86        | 247          | 2.50   | 1.53   |
| sort      | txt       | 12          | 30.19        | 23.75       | 855          | 2.06   | 1.72   |
| sort      | merge_txt | 12          | 30.49        | 18.33       | 660          | 2.06   | 1.69   |
| strongsort| txt       | 12          | 30.30        | 7.64        | 275          | 3.39   | 0.94   |
| strongsort| merge_txt| 12          | 30.74        | 6.61        | 238          | 3.39   | 0.89   |

**Summary (from paper):**

- **Best average MOTA (after merge):** BoT-SORT (41.28%).
- **Lowest average ID switch (after merge):** DeepSORT (3.56).
- **Overall:** Total ID switch reduction 363; number of cases with ID switch reduction: 78. IoU threshold for evaluation: 0.5. **MT** = Mostly Tracked (tracked ≥ 80% of lifespan); **ML** = Mostly Lost (tracked ≤ 20% of lifespan).

---

## Evaluation metrics: formulas and how they are computed

### 1. MOTA (Multi-Object Tracking Accuracy)

MOTA measures overall tracking accuracy by penalizing false negatives, false positives, and identity switches:

$$\text{MOTA} = 1 - \frac{FN + FP + |\text{ID-Switch}|}{gtDet}$$

- **gtDet:** Total number of ground-truth detections (over all frames).
- **FN:** False negatives — ground-truth objects that were not matched to any prediction (IoU ≥ threshold).
- **FP:** False positives — predictions that were not matched to any ground-truth.
- **ID-Switch:** Number of identity switches (see below).

**In this code:** MOTA is reported in percent: `MOTA = (1 - (FN + FP + id_switch_count) / total_GT) * 100`. Higher is better; can be negative when errors exceed gtDet.

### 2. MOTP (Multi-Object Tracking Precision)

MOTP measures localization quality of **matched** prediction–GT pairs:

- **Definition (CLEAR MOT–style):** Average localization error over all matches. In this implementation, error per match is taken as \(1 - \text{IoU}\).
- **Computation:** For each frame, for each matched (GT, pred) pair with IoU ≥ threshold, add `(1 - IoU)` to a sum; then  
  `MOTP = (sum of (1 - IoU) / number of matches) * 100`.

So MOTP is the **average (1 − IoU) × 100** for matched pairs. **Lower MOTP is better** (less localization error). The script reports this value as “MOTP (%)”.

### 3. ID switch (identity switch)

- **Definition:** The number of times a ground-truth track is matched to a **different** predicted track than in the previous frame, with a significant change in spatial position (to avoid counting benign ID changes after merge).
- **In this code:** For each frame, GT–pred matches are found by IoU (greedy, threshold 0.5). For each GT track, if the matched predicted track ID changes from the previous frame **and** the IoU between the current and previous predicted boxes is &lt; 0.3, one ID switch is counted. So only clear “swaps” of identity are counted.

### 4. Hand identification (from paper)

- **Sensitivity (True Positive Rate):** \(\displaystyle Sens = \frac{TP}{TP + FN}\).
- **False Positive Rate:** \(\displaystyle FPR = \frac{FP}{FP + TP}\).

Where **TP** = patient hand correctly detected; **FP** = detection that is not a patient hand; **FN** = patient hand missed. The paper reports that the proposed method reduces FPR (e.g. from 28.1% to 2.1% in one setting) while keeping sensitivity similar to Mask R-CNN.

### 5. IoU (Intersection over Union)

A prediction and a ground-truth box are **matched** in a frame if their IoU ≥ 0.5 (default). IoU is used for:

- Deciding TP/FP/FN and thus MOTA.
- Building the sum of \(1 - \text{IoU}\) for MOTP.
- First-frame ID alignment and ID-switch detection (with an extra 0.3 IoU threshold for continuity).

---

## Repository structure

```
Rehab-2025/
├── BoxMOT_tracking_detectron2/   # Tracking: Detectron2 + BoxMOT
│   ├── boxmot_tracking_detectron2.py
│   ├── batch_process.py
│   ├── batch_all_trackers.py
│   ├── test_all_trackers.py
│   └── README.md
├── merge_track_source_code2/     # Merge tracks + evaluation
│   ├── batch_merge_tracks_org.py # Batch merge all txt files
│   ├── merge_tracks_org.py      # Merge algorithm (graph + Bellman–Ford)
│   ├── evaluate_txt_files.py    # Evaluate tracking: GT vs txt / merge_txt
│   ├── merge_tracks_convert_labelme.py
│   ├── utils.py, video_utils.py
│   └── ... (analysis and helper scripts)
└── README.md
```

**Workflow:**

1. **Tracking** (BoxMOT_tracking_detectron2): video → Detectron2 segmentation → BoxMOT tracker → MOT-format `.txt` and output videos.
2. **Merge** (merge_track_source_code2): read tracker `.txt` → merge tracks per left (track 1) and right (track 2) using a graph and Bellman–Ford → output merged `.txt`.
3. **Evaluation** (merge_track_source_code2): compare ground-truth to `txt` (before merge) and `merge_txt` (after merge); output MOTA, MOTP, TP/FP/FN, ID switches, and summary tables.

---

## How to use the code

### Prerequisites

- Python 3.8+
- For tracking: Detectron2, BoxMOT (or fallback SORT/DeepSORT).
- For merge and evaluation: `networkx`, `numpy`, `opencv-python`.

See `BoxMOT_tracking_detectron2/README.md` for tracking setup and paths.

### Step 1: Run tracking (optional if you already have tracker output)

From the project root (paths may need to be adjusted):

```bash
cd Rehab-2025/BoxMOT_tracking_detectron2
pip install -r requirements.txt
# Set PYTHONPATH for Detectron2 if needed, then e.g.:
python batch_all_trackers.py \
  --input-dir /path/to/videos \
  --output-dir /path/to/output \
  --config-file /path/to/detectron2_config.yaml \
  --model-weights /path/to/model.pth
```

This produces per-tracker folders, each with `txt/` and `videos/`. Use the same structure for the evaluation step (see below).

### Step 2: Merge tracks (batch)

Edit `merge_track_source_code2/batch_merge_tracks_org.py`: set `input_dir` to the folder containing tracker `.txt` files and `output_dir` to where you want merged `.txt` files. Then:

```bash
cd Rehab-2025/merge_track_source_code2
python batch_merge_tracks_org.py
```

- **Input:** Directory of MOT-format `.txt` files (e.g. one per video per tracker).
- **Output:** One merged `.txt` per input file. Left hand is fixed as track 1, right hand as track 2 (`fix_tracks=[(1,1), (2,2)]`; change in code if your convention differs).

### Step 3: Run evaluation (MOTA, MOTP, ID switches)

Edit `merge_track_source_code2/evaluate_txt_files.py`: in `main()` set

- `gt_dir` — directory of ground-truth `.txt` files (same MOT format),
- `pred_dir` — root directory under which each tracker has two subfolders: `txt/` (raw tracker output) and `merge_txt/` (merged output).

Then:

```bash
cd Rehab-2025/merge_track_source_code2
python evaluate_txt_files.py
```

The script will:

- Match GT files to predicted files (by base name),
- For each tracker, evaluate both `txt` and `merge_txt`,
- Print per-file and per-tracker tables (MOTA, MOTP, TP, FP, FN, ID_SW),
- Print comparison “before vs after merge” (MOTA/MOTP improvement and ID switch reduction).

### Data format (MOT)

- **Txt (tracking / ground truth):** One line per detection:  
  `frame_id, track_id, x, y, width, height [, conf, class_id, visibility]`  
  (x, y = top-left; width, height in pixels.)

### Changing paths for your data

- **Merge:** In `batch_merge_tracks_org.py`, in `main()`, set `input_dir` and `output_dir`.
- **Evaluation:** In `evaluate_txt_files.py`, in `main()`, set `gt_dir` and `pred_dir`.

Ground-truth and predicted file names should align (e.g. same video/base name) so the script can pair them. Directory layout under `pred_dir` should be: `pred_dir/<tracker_name>/txt/*.txt` and `pred_dir/<tracker_name>/merge_txt/*.txt`.

---

## Notes

- Default paths in the scripts point to directories on the original dev machine (`/home/vuhai/Rehab-Tung/...`). Update them for your environment or dataset.
- Rehab-2025 is the supplementary evaluation code for merge-track; the main Rehab-Tung repo contains the full tracking and pipeline code.
- For full methodology and MergeTrack algorithm, see `paper/main_revised_ver3.tex`.
