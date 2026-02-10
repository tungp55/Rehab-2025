"""
BoxMOT Tracking với Detectron2 Segmentation Models
Hỗ trợ nhiều thuật toán tracking: ByteTrack, OcSort, BotSort, StrongSort, HybridSort, SORT, DeepSort
Tương tự dt2ds.py nhưng sử dụng BoxMOT trackers thay vì chỉ SORT/DeepSort
"""
import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Add current directory to path to ensure local imports work
# Remove any paths containing /mnt/disk2/egouser_data to avoid importing from wrong location
current_dir = Path(__file__).parent.absolute()
sys.path = [p for p in sys.path if '/mnt/disk2/egouser_data/BoxMOT_tracking_detectron2' not in p]
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import BoxMOT trackers
BOXMOT_AVAILABLE = False
try:
    # Try different import methods for BoxMOT
    try:
        from boxmot import create_tracker
        BOXMOT_AVAILABLE = True
    except ImportError:
        try:
            from boxmot.trackers import create_tracker
            BOXMOT_AVAILABLE = True
        except ImportError:
            try:
                from boxmot.trackers.bytetrack import BYTETracker
                from boxmot.trackers.ocsort import OCSort
                from boxmot.trackers.botsort import BoTSORT
                from boxmot.trackers.strongsort import StrongSORT
                from boxmot.trackers.hybridsort import HybridSORT
                BOXMOT_AVAILABLE = True
                # Create wrapper function
                def create_tracker(tracking_method, device='cuda', **kwargs):
                    tracker_map = {
                        'bytetrack': BYTETracker,
                        'ocsort': OCSort,
                        'botsort': BoTSORT,
                        'strongsort': StrongSORT,
                        'hybridsort': HybridSORT,
                    }
                    tracker_class = tracker_map.get(tracking_method.lower())
                    if tracker_class:
                        return tracker_class(**kwargs)
                    raise ValueError(f"Unknown tracker: {tracking_method}")
            except ImportError:
                BOXMOT_AVAILABLE = False
                print("⚠️  BoxMOT not found. Will use SORT/DeepSort fallback from Detectron2DeepSortPlus")
except Exception as e:
    BOXMOT_AVAILABLE = False
    print(f"⚠️  BoxMOT import error: {e}. Will use SORT/DeepSort fallback")

# Import SORT and DeepSort from Detectron2DeepSortPlus as fallback
sys.path.append('/home/vuhai/Rehab-Tung/Detectron2DeepSortPlus')
try:
    from sort import Sort as SortTracker
    from deep_sort import DeepSort
    SORT_AVAILABLE = True
except ImportError:
    SORT_AVAILABLE = False
    print("⚠️  SORT/DeepSort not available from Detectron2DeepSortPlus")

# Import utilities from Detectron2DeepSortPlus
sys.path.append('/home/vuhai/Rehab-Tung/Detectron2DeepSortPlus')
try:
    from util import draw_bboxes
except ImportError:
    # Fallback to local util
    sys.path.append('/home/vuhai/Rehab-Tung/BoxMOT_tracking')
    from util import draw_bboxes

# Import detector
from detector_detectron2 import Detectron2SegmentationDetector


def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_class_ids_by_iou(output_boxes, det_boxes, det_class_ids, iou_threshold=0.1):
    """
    Match class_ids from detections to tracking outputs using IoU matching
    
    Args:
        output_boxes: Tracking output boxes [[x1, y1, x2, y2], ...]
        det_boxes: Original detection boxes [[x1, y1, x2, y2], ...]
        det_class_ids: Class IDs from original detections
        iou_threshold: Minimum IoU for matching (default: 0.1, lower for better matching)
    
    Returns:
        Matched class_ids for output_boxes
    """
    if len(output_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros(len(output_boxes), dtype=int)
    
    matched_class_ids = np.zeros(len(output_boxes), dtype=int)
    used_det_indices = set()
    
    # For each output box, find best matching detection by IoU
    for i, out_box in enumerate(output_boxes):
        best_iou = 0.0
        best_det_idx = -1
        
        for j, det_box in enumerate(det_boxes):
            if j in used_det_indices:
                continue
            
            iou = compute_iou(out_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_det_idx = j
        
        # Use best match if IoU is above threshold, otherwise try fallback
        if best_det_idx >= 0 and best_iou >= iou_threshold:
            matched_class_ids[i] = det_class_ids[best_det_idx]
            used_det_indices.add(best_det_idx)
        else:
            # No good match found, try fallback strategies
            # Strategy 1: If same count, match by index (likely same order)
            if len(output_boxes) == len(det_boxes) and i < len(det_class_ids):
                matched_class_ids[i] = det_class_ids[i]
            # Strategy 2: Find closest box by center distance if IoU failed
            elif best_det_idx >= 0:
                # Use best match even if IoU is low (better than default 0)
                matched_class_ids[i] = det_class_ids[best_det_idx]
                used_det_indices.add(best_det_idx)
            # Strategy 3: Default to 0 (will be handled by main function if needed)
            else:
                matched_class_ids[i] = 0
    
    return matched_class_ids


def find_default_reid_weights():
    """
    Find default ReID weights in reID_weight directory
    
    Returns:
        Path to default ReID weights file, or None if not found
    """
    reid_dir = Path(__file__).parent / "reID_weight"
    
    # Priority order: osnet > mobilenetv2_1.4 > mobilenetv2_1.0
    # BoxMOT accepts both .pt and .pth, but prefers .pt
    default_weights = [
        "osnet_x1_0_imagenet.pt",
        "osnet_x1_0_imagenet.pth",
        "mobilenetv2_1.4-bc1cc36b.pt",
        "mobilenetv2_1.4-bc1cc36b.pth",
        "mobilenetv2_1.0-0f96a698.pt",
        "mobilenetv2_1.0-0f96a698.pth",
    ]
    
    for weight_file in default_weights:
        weight_path = reid_dir / weight_file
        if weight_path.exists():
            # If .pth file, try to convert to .pt or use as-is
            if weight_path.suffix == '.pth':
                # BoxMOT might accept .pth, but let's try to use it
                return str(weight_path)
            return str(weight_path)
    
    # If no default found, return first .pt or .pth file found
    if reid_dir.exists():
        pt_files = list(reid_dir.glob("*.pt"))
        if pt_files:
            return str(pt_files[0])
        pth_files = list(reid_dir.glob("*.pth"))
        if pth_files:
            return str(pth_files[0])
    
    return None


def create_boxmot_tracker(tracker_type, args):
    """
    Create BoxMOT tracker
    
    Args:
        tracker_type: Type of tracker ('bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort')
        args: Arguments containing tracker configuration
    
    Returns:
        Tracker instance
    """
    if not BOXMOT_AVAILABLE:
        raise ImportError("BoxMOT is not available")
    
    # Map tracker names to BoxMOT names
    tracker_map = {
        'bytetrack': 'bytetrack',
        'ocsort': 'ocsort',
        'botsort': 'botsort',
        'strongsort': 'strongsort',
        'hybridsort': 'hybridsort',
    }
    
    boxmot_name = tracker_map.get(tracker_type.lower())
    if boxmot_name is None:
        raise ValueError(f"Unknown BoxMOT tracker: {tracker_type}")
    
    # Handle ReID weights for trackers that require them
    reid_weights = args.reid_weights
    if tracker_type in ['strongsort', 'botsort', 'hybridsort']:
        if not reid_weights:
            # Try to find default ReID weights
            default_reid = find_default_reid_weights()
            if default_reid:
                reid_weights = default_reid
                print(f"✅ Using default ReID weights: {reid_weights}")
            else:
                raise RuntimeError(
                    f"{tracker_type} requires ReID weights.\n"
                    f"Please provide --reid-weights PATH or place ReID weights in:\n"
                    f"  {Path(__file__).parent / 'reID_weight'}\n"
                    f"Available ReID models: osnet_x1_0_imagenet.pth, mobilenetv2_1.4-bc1cc36b.pth, mobilenetv2_1.0-0f96a698.pth"
                )
        else:
            # Verify ReID weights file exists
            if not os.path.isfile(reid_weights):
                raise FileNotFoundError(f"ReID weights file not found: {reid_weights}")
        
        # Convert .pth to .pt if needed (BoxMOT prefers .pt)
        if reid_weights.endswith('.pth'):
            pt_path = reid_weights.replace('.pth', '.pt')
            if os.path.exists(pt_path):
                reid_weights = pt_path
                print(f"✅ Using .pt version: {reid_weights}")
            else:
                # Try to convert
                try:
                    import torch
                    weights = torch.load(reid_weights, map_location='cpu')
                    torch.save(weights, pt_path)
                    reid_weights = pt_path
                    print(f"✅ Converted .pth to .pt: {reid_weights}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not convert .pth to .pt: {e}")
                    print(f"   Using .pth file as-is (may cause issues)")

    # Handle device: BoxMOT may need '0' instead of 'cuda'
    device = args.device
    if device == 'cuda':
        try:
            import torch
            import warnings
            # Suppress CUDA initialization warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                # Try to initialize CUDA
                try:
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        device = '0'  # Use first GPU
                        print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
                    else:
                        device = 'cpu'
                        print("⚠️  CUDA not available, using CPU")
                except Exception as e:
                    # If CUDA init fails, try to use it anyway (might work despite warning)
                    if torch.cuda.device_count() > 0:
                        device = '0'
                        print(f"⚠️  CUDA warning suppressed, using device 0")
                    else:
                        device = 'cpu'
                        print(f"⚠️  CUDA initialization failed: {e}, using CPU")
        except ImportError:
            device = 'cpu'

    try:
        tracker = create_tracker(
            boxmot_name,  # tracker_type is positional argument
            tracker_config=args.tracker_config if hasattr(args, "tracker_config") else None,
            reid_weights=reid_weights,
            device=device,
        )
    except Exception as e:
        # Try with additional parameters if needed
        try:
            tracker = create_tracker(
                boxmot_name,  # tracker_type is positional argument
                tracker_config=args.tracker_config if hasattr(args, "tracker_config") else None,
                reid_weights=reid_weights,
                device=device,
                half=False,  # Use FP32
            )
        except Exception as e2:
            raise RuntimeError(f"Failed to create {tracker_type} tracker: {e2}")
    
    return tracker


def process_frame_boxmot(tracker, dets, im, frame_id):
    """
    Process frame with BoxMOT tracker
    
    Args:
        tracker: BoxMOT tracker instance
        dets: Detections in format [[x1, y1, x2, y2, score, class_id], ...] or [[x1, y1, x2, y2, score], ...]
        im: Input image
        frame_id: Current frame ID
    
    Returns:
        outputs: Tracking results in format [[x1, y1, x2, y2, track_id, class_id], ...]
    """
    if len(dets) == 0:
        return np.empty((0, 6))
    
    # Convert detections to format expected by BoxMOT
    # BoxMOT requires: [x1, y1, x2, y2, conf, cls] (6 columns)
    dets_array = np.array(dets)
    
    # Extract class_ids if available (dets has 6 columns: [x1, y1, x2, y2, score, class_id])
    # Also save original detection boxes for IoU matching if needed
    if dets_array.shape[1] == 6:
        # Class ID is already in detections
        class_ids = dets_array[:, 5].astype(int)
        original_det_boxes = dets_array[:, :4].copy()  # Save for IoU matching
        dets_array = dets_array[:, :5]  # Remove class_id temporarily for BoxMOT input format
    else:
        # No class_id provided, default to 0
        class_ids = np.zeros(dets_array.shape[0], dtype=int)
        original_det_boxes = dets_array[:, :4].copy() if dets_array.shape[1] >= 4 else np.empty((0, 4))
    
    # Add class ID column for BoxMOT (BoxMOT requires [x1, y1, x2, y2, conf, cls])
    if dets_array.shape[1] == 5:
        class_col = class_ids.reshape(-1, 1)
        dets_array = np.hstack([dets_array, class_col])
    
    # BoxMOT update method signature varies by tracker
    try:
        outputs = tracker.update(dets_array, im)
    except TypeError:
        try:
            outputs = tracker.update(dets_array)
        except Exception as e:
            print(f"⚠️  Error updating tracker: {e}")
            return np.empty((0, 6))
    
    # Convert outputs to format [[x1, y1, x2, y2, track_id, class_id], ...]
    # BoxMOT outputs typically: [x1, y1, x2, y2, track_id, conf, cls] (7 columns) or [x1, y1, x2, y2, track_id, cls] (6 columns)
    if len(outputs) > 0:
        outputs = np.array(outputs)
        # Extract x1, y1, x2, y2, track_id, class_id
        if outputs.shape[1] >= 7:
            # Output has: [x1, y1, x2, y2, track_id, conf, cls]
            outputs = outputs[:, [0, 1, 2, 3, 4, 6]]  # Keep x1, y1, x2, y2, track_id, class_id
        elif outputs.shape[1] == 6:
            # Output has: [x1, y1, x2, y2, track_id, cls] or [x1, y1, x2, y2, track_id, conf]
            # Check if last column is class_id (usually small integers 0, 1, 2...)
            last_col = outputs[:, 5]
            if np.all(last_col == last_col.astype(int)) and np.max(last_col) < 10:
                # Last column is likely class_id
                outputs = outputs[:, [0, 1, 2, 3, 4, 5]]  # Keep x1, y1, x2, y2, track_id, class_id
            else:
                # Last column is conf, need to get class_id from original detections
                # Use IoU matching for better accuracy
                outputs_with_class = np.zeros((outputs.shape[0], 6))
                outputs_with_class[:, :5] = outputs[:, :5]
                output_boxes = outputs[:, :4]
                class_ids_matched = match_class_ids_by_iou(output_boxes, original_det_boxes, class_ids)
                outputs_with_class[:, 5] = class_ids_matched
                outputs = outputs_with_class
        elif outputs.shape[1] >= 5:
            # Output has: [x1, y1, x2, y2, track_id] - need to add class_id
            # Use IoU matching for better accuracy
            output_boxes = outputs[:, :4]
            class_ids_output = match_class_ids_by_iou(output_boxes, original_det_boxes, class_ids)
            outputs = np.column_stack([outputs, class_ids_output])
        elif outputs.shape[1] == 4:
            # Output has: [x1, y1, x2, y2] - add track_id and class_id
            # Use IoU matching for better accuracy
            track_ids = np.arange(len(outputs))
            output_boxes = outputs[:, :4]
            class_ids_output = match_class_ids_by_iou(output_boxes, original_det_boxes, class_ids)
            outputs = np.column_stack([outputs, track_ids, class_ids_output])
    
    return outputs


def process_frame_sort(tracker, dets, im):
    """
    Process frame with SORT tracker (fallback)
    
    Args:
        tracker: SORT tracker instance
        dets: Detections in format [[x1, y1, x2, y2, score, class_id], ...] or [[x1, y1, x2, y2, score], ...]
        im: Input image
    
    Returns:
        outputs: Tracking results in format [[x1, y1, x2, y2, track_id, class_id], ...]
    """
    if len(dets) == 0:
        return np.empty((0, 6))
    
    # Extract class_ids if available (dets has 6 columns: [x1, y1, x2, y2, score, class_id])
    dets_array = np.array(dets)
    if dets_array.shape[1] == 6:
        class_ids = dets_array[:, 5].astype(int)
        original_det_boxes = dets_array[:, :4].copy()  # Save for IoU matching
        # SORT expects [x1, y1, x2, y2, score] format (5 columns)
        dets_for_sort = dets_array[:, :5]
    else:
        # No class_id provided, default to 0
        class_ids = np.zeros(dets_array.shape[0], dtype=int)
        original_det_boxes = dets_array[:, :4].copy() if dets_array.shape[1] >= 4 else np.empty((0, 4))
        dets_for_sort = dets_array
    
    # Update tracker (SORT expects [x1, y1, x2, y2, score] format)
    outputs = tracker.update(dets_for_sort)
    
    # Handle empty outputs
    if len(outputs) == 0:
        return np.empty((0, 6))
    
    outputs = np.array([element.clip(min=0) for element in outputs]).astype(int)
    
    # Convert outputs to format [[x1, y1, x2, y2, track_id, class_id], ...]
    # SORT outputs format: [x1, y1, x2, y2, track_id] (5 columns)
    # Ensure outputs has at least 5 columns
    if outputs.shape[1] < 5:
        # Pad with zeros if needed
        padded_outputs = np.zeros((len(outputs), 5), dtype=int)
        padded_outputs[:, :outputs.shape[1]] = outputs
        outputs = padded_outputs
    
    # Extract output boxes for IoU matching
    output_boxes = outputs[:, :4]
    
    # Use IoU matching to get class_ids from original detections
    class_ids_output = match_class_ids_by_iou(output_boxes, original_det_boxes, class_ids)
    
    # Stack: [x1, y1, x2, y2, track_id, class_id]
    # Keep first 5 columns (x1, y1, x2, y2, track_id) and add class_id
    outputs = np.column_stack([outputs[:, :5], class_ids_output])
    
    return outputs


def process_frame_deepsort(tracker, dets, im):
    """
    Process frame with DeepSort tracker (fallback)
    
    Args:
        tracker: DeepSort tracker instance
        dets: Detections in format [[x1, y1, x2, y2, score, class_id], ...] or [[x1, y1, x2, y2, score], ...]
        im: Input image
    
    Returns:
        outputs: Tracking results in format [[x1, y1, x2, y2, track_id, class_id], ...]
    """
    if len(dets) == 0:
        return np.empty((0, 6))
    
    # Extract class_ids if available (dets has 6 columns: [x1, y1, x2, y2, score, class_id])
    dets_array = np.array(dets)
    if dets_array.shape[1] == 6:
        class_ids = dets_array[:, 5].astype(int)
        original_det_boxes = dets_array[:, :4].copy()  # Save for IoU matching
    else:
        # No class_id provided, default to 0
        class_ids = np.zeros(dets_array.shape[0], dtype=int)
        original_det_boxes = dets_array[:, :4].copy() if dets_array.shape[1] >= 4 else np.empty((0, 4))
    
    # Convert detections to ccwh format for DeepSort
    ccwh_boxes = []
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        ccwh_boxes.append([center_x, center_y, width, height])
    ccwh_boxes = np.array(ccwh_boxes)
    confidences = np.array([det[4] for det in dets])
    
    # Update tracker
    outputs, __ = tracker.update(ccwh_boxes, confidences, im)
    
    # Convert outputs to format [[x1, y1, x2, y2, track_id, class_id], ...]
    # DeepSort outputs format: [x1, y1, x2, y2, track_id] or [x1, y1, x2, y2, track_id, conf]
    if len(outputs) > 0:
        outputs = np.array(outputs)
        # DeepSort typically returns [x1, y1, x2, y2, track_id] or [x1, y1, x2, y2, track_id, conf]
        if outputs.shape[1] >= 5:
            # Output has at least [x1, y1, x2, y2, track_id]
            # Use IoU matching for better accuracy
            output_boxes = outputs[:, :4]
            class_ids_output = match_class_ids_by_iou(output_boxes, original_det_boxes, class_ids)
            
            # Stack: [x1, y1, x2, y2, track_id, class_id]
            if outputs.shape[1] == 5:
                outputs = np.column_stack([outputs, class_ids_output])
            elif outputs.shape[1] == 6:
                # Output has [x1, y1, x2, y2, track_id, conf], replace conf with class_id
                outputs = np.column_stack([outputs[:, :5], class_ids_output])
            else:
                # More columns, keep first 5 and add class_id
                outputs = np.column_stack([outputs[:, :5], class_ids_output])
        else:
            # Fallback: add track_id and class_id
            track_ids = np.arange(len(outputs))
            output_boxes = outputs[:, :4] if outputs.shape[1] >= 4 else np.empty((len(outputs), 4))
            class_ids_output = match_class_ids_by_iou(output_boxes, original_det_boxes, class_ids)
            outputs = np.column_stack([outputs, track_ids, class_ids_output])
    else:
        outputs = np.empty((0, 6))
    
    return outputs


def main():
    args = get_parser().parse_args()
    
    if args.display:
        cv2.namedWindow("out_vid", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("out_vid", 960, 720)
    
    # Initialize tracker based on type
    tracker = None
    tracker_type = args.tracker.lower()
    
    if tracker_type in ['bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort']:
        # Use BoxMOT tracker
        if not BOXMOT_AVAILABLE:
            print(f"⚠️  BoxMOT not available. Cannot use {tracker_type} tracker.")
            print("💡 Please install BoxMOT: pip install boxmot")
            print("💡 Or use 'sort' or 'deepsort' trackers instead.")
            raise ImportError(f"BoxMOT is required for {tracker_type} tracker")
        
        try:
            tracker = create_boxmot_tracker(tracker_type, args)
            print(f"✅ Initialized BoxMOT tracker: {tracker_type}")
        except Exception as e:
            print(f"❌ Failed to create BoxMOT tracker: {e}")
            raise RuntimeError(f"Failed to initialize {tracker_type} tracker: {e}")
    elif tracker_type == 'sort':
        if SORT_AVAILABLE:
            tracker = SortTracker()
            print("✅ Initialized SORT tracker")
        else:
            raise ImportError("SORT tracker not available")
    elif tracker_type == 'deepsort':
        if SORT_AVAILABLE:
            from distutils.util import strtobool
            tracker = DeepSort(
                args.deepsort_checkpoint,
                nms_max_overlap=args.nms_max_overlap,
                use_cuda=bool(strtobool(args.use_cuda))
            )
            print("✅ Initialized DeepSort tracker")
        else:
            raise ImportError("DeepSort tracker not available")
    else:
        raise ValueError(f"Unknown tracker type: {args.tracker}")
    
    assert os.path.isfile(args.input), f"Error: input file not found: {args.input}"
    assert os.path.isfile(args.config_file), f"Error: config file not found: {args.config_file}"
    assert os.path.isfile(args.model_weights), f"Error: model weights not found: {args.model_weights}"
    
    # Initialize Detectron2 detector
    detector = Detectron2SegmentationDetector(
        config_file=args.config_file,
        model_weights=args.model_weights,
        conf_threshold=args.confidence_threshold,
        device=args.device,
        num_classes=args.num_classes
    )
    
    # Open input video
    inp_vid = cv2.VideoCapture(args.input)
    num_frames = int(inp_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(inp_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(inp_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = inp_vid.get(cv2.CAP_PROP_FPS)
    
    # Setup output video
    if args.out_vid:
        os.makedirs(os.path.dirname(args.out_vid) if os.path.dirname(args.out_vid) else '.', exist_ok=True)
        out_vid = cv2.VideoWriter(
            filename=args.out_vid,
            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
            fps=args.fps if args.fps > 0 else fps_input,
            frameSize=(frame_width, frame_height),
        )
    else:
        out_vid = None
    
    # Setup output text file
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) if os.path.dirname(args.out_txt) else '.', exist_ok=True)
        out_txt = open(args.out_txt, "w+")
    else:
        out_txt = None
    
    total_counter = [0] * 1000000
    
    print(f"📹 Processing {num_frames} frames with {tracker_type} tracker...")
    
    for frameID in tqdm(range(num_frames)):
        ret, im = inp_vid.read()
        if not ret:
            break
        
        start = time.time()
        
        # Detect và segment với Detectron2
        dets, masks, region = detector.detect(im)
        
        # Region-based tracking
        if args.region_based:
            im = region
        
        # Tracking
        if tracker_type in ['bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort']:
            outputs = process_frame_boxmot(tracker, dets, im, frameID)
        elif tracker_type == 'sort':
            outputs = process_frame_sort(tracker, dets, im)
        elif tracker_type == 'deepsort':
            outputs = process_frame_deepsort(tracker, dets, im)
        else:
            outputs = []
        
        # Process outputs
        current_counter = []
        if len(outputs):
            tlbr_boxes = outputs[:, :4]
            # Extract track_id (column 4) and class_id (column 5 if available)
            if outputs.shape[1] >= 6:
                identities = current_counter = outputs[:, 4].astype(int)  # track_id
                class_ids_output = outputs[:, 5].astype(int)  # class_id
            else:
                identities = current_counter = outputs[:, -1].astype(int)  # track_id
                class_ids_output = np.ones(len(outputs), dtype=int)  # Default class_id = 1 if not available
            
            ordered_identities = []
            for identity in identities:
                if identity >= len(total_counter):
                    print(f'⚠️  Out of size {len(total_counter)}/{identity}')
                if not total_counter[identity]:
                    total_counter[identity] = max(total_counter) + 1 if max(total_counter) > 0 else 1
                ordered_identities.append(total_counter[identity])
            
            # Draw results
            im = draw_bboxes(im, tlbr_boxes, ordered_identities, binary_masks=masks)
            
            # Write to output text file (MOT format: frame_id, track_id, x1, y1, width, height, score, class_id, visibility)
            if out_txt:
                for i in range(len(ordered_identities)):
                    tlbr = tlbr_boxes[i]
                    score = 1.0  # Default score (could extract from outputs if available)
                    class_id = int(class_ids_output[i]) if i < len(class_ids_output) else 1
                    visibility = 1  # Default visibility
                    line = [
                        frameID + 1,
                        ordered_identities[i],
                        tlbr[0],
                        tlbr[1],
                        tlbr[2] - tlbr[0],  # width
                        tlbr[3] - tlbr[1],  # height
                        score,
                        class_id,  # Use actual class_id from detection/tracking
                        visibility
                    ]
                    out_txt.write(",".join(str(item) for item in line) + "\n")
        
        # Draw info text
        end = time.time()
        im = cv2.putText(im, "Frame ID: " + str(frameID + 1), (20, 30), 0, 5e-3 * 200, (0, 255, 0), 2)
        time_fps = "Time: {}s, fps: {}".format(round(end - start, 2), round(1 / (end - start), 2))
        im = cv2.putText(im, time_fps, (20, 60), 0, 5e-3 * 200, (0, 255, 0), 3)
        im = cv2.putText(im, f"Detectron2 + {tracker_type.upper()}", (20, 90), 0, 5e-3 * 200, (0, 255, 0), 3)
        im = cv2.putText(im, "Current Hand Counter: " + str(len(current_counter)), (20, 120), 0, 5e-3 * 200, (0, 255, 0), 2)
        im = cv2.putText(im, "Total Hand Counter: " + str(max(total_counter) if max(total_counter) > 0 else 0), (20, 150), 0, 5e-3 * 200, (0, 255, 0), 2)
        
        if args.display:
            cv2.imshow("out_vid", im)
            cv2.waitKey(1)
        
        if out_vid:
            out_vid.write(im)
    
    # Cleanup
    if out_vid:
        out_vid.release()
    if out_txt:
        out_txt.close()
    inp_vid.release()
    
    print(f"\n✅ Processing completed!")
    if args.out_vid:
        print(f"📁 Output video: {args.out_vid}")
    if args.out_txt:
        print(f"📁 Output text: {args.out_txt}")


def get_parser():
    parser = argparse.ArgumentParser(description="BoxMOT Tracking with Detectron2 Segmentation Models")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help='Path to input video file',
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        default="/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        help="Path to Detectron2 config file (.yaml)",
    )
    
    parser.add_argument(
        "--model-weights",
        type=str,
        default="/home/vuhai/Rehab-Tung/Detectron2DeepSortPlus/model_0004999.pth",
        help="Path to Detectron2 model weights (.pth)",
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (default: 2 for hand tracking)",
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions",
    )
    
    parser.add_argument(
        "--region_based",
        type=int,
        default=1,
        help="1 if track on segmented region only, 0 for full image",
    )
    
    parser.add_argument(
        "--tracker",
        type=str,
        default='bytetrack',
        choices=['bytetrack', 'ocsort', 'botsort', 'strongsort', 'hybridsort', 'sort', 'deepsort'],
        help='Tracker type: bytetrack, ocsort, botsort, strongsort, hybridsort, sort, deepsort',
    )

    parser.add_argument(
        "--reid-weights",
        type=str,
        default=None,
        help="ReID weights for trackers that require appearance features (strongsort, botsort, hybridsort). "
             "If not provided, will auto-detect from ./reID_weight/ directory. "
             "Priority: osnet_x1_0_imagenet.pth > mobilenetv2_1.4-bc1cc36b.pth > mobilenetv2_1.0-0f96a698.pth",
    )

    parser.add_argument(
        "--tracker-config",
        type=str,
        default=None,
        help="Optional tracker config file for BoxMOT trackers",
    )
    
    parser.add_argument(
        "--deepsort_checkpoint",
        type=str,
        default="/home/vuhai/Rehab-Tung/Detectron2DeepSortPlus/deep_sort/deep/checkpoint/ckpt.t7",
        help='DeepSort checkpoint path (only for deepsort tracker)',
    )
    
    parser.add_argument(
        "--nms_max_overlap",
        type=float,
        default=0.5,
        help='Non-max suppression threshold (for DeepSort)',
    )
    
    parser.add_argument(
        "--display",
        type=bool,
        default=False,
        help="Display frames during processing",
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS (0 = use input video FPS)",
    )
    
    parser.add_argument(
        "--out_vid",
        type=str,
        default=None,
        help="Output video path",
    )
    
    parser.add_argument(
        "--out_txt",
        type=str,
        default=None,
        help="Output text file path (MOT format)",
    )
    
    parser.add_argument(
        "--use_cuda",
        type=str,
        default="True",
        help="Use GPU if true, else use CPU only (for DeepSort)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device for model and tracker: 'cuda' or 'cpu'",
    )
    
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    
    return parser


if __name__ == "__main__":
    main()
