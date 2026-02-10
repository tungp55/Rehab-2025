"""
Detectron2 Segmentation Detector Wrapper
Hỗ trợ Detectron2 Mask R-CNN segmentation models
"""
import os
import sys
import numpy as np

# Add Detectron2 path
sys.path.append('/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2')

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class Detectron2SegmentationDetector:
    """Detectron2 Segmentation Detector"""
    
    def __init__(self, config_file, model_weights, conf_threshold=0.5, device='cuda', num_classes=2):
        """
        Initialize Detectron2 detector
        
        Args:
            config_file: Path to Detectron2 config file (.yaml)
            model_weights: Path to trained model weights (.pth)
            conf_threshold: Confidence threshold
            device: Device ('cuda' or 'cpu')
            num_classes: Number of classes (default: 2 for hand tracking)
        """
        self.config_file = config_file
        self.model_weights = model_weights
        self.conf_threshold = conf_threshold
        self.device = device
        self.num_classes = num_classes
        
        print("📥 Loading Detectron2 model...")
        print(f"   Config: {config_file}")
        print(f"   Weights: {model_weights}")
        
        # Setup config
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        cfg.MODEL.WEIGHTS = model_weights
        
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        
        # Check if this is a mask model
        self.is_mask_model = os.path.basename(config_file).split('_')[0] == 'mask'
        
        print("✅ Detectron2 model loaded!")
        print(f"   Mask model: {self.is_mask_model}")
        print(f"   Confidence threshold: {conf_threshold}")
    
    def detect(self, image):
        """
        Detect and segment objects in image
        
        Args:
            image: Input image (numpy array, BGR format)
        
        Returns:
            dets: List of detections [[x1, y1, x2, y2, score, class_id], ...]
            masks: List of binary masks (numpy arrays) - empty if not mask model
            region: Image with region extraction (only segmented areas) - original if not mask model
        """
        # Run inference
        predictions = self.predictor(image)
        
        # Extract boxes, scores, and class_ids
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()
        classes = predictions["instances"].pred_classes.cpu().numpy()  # Extract class IDs
        
        dets = []
        masks = []
        
        # Format detections: [x1, y1, x2, y2, score, class_id]
        for (box, score, cls) in zip(boxes, scores, classes):
            top, left, bottom, right = box  # top, left, bottom, right
            dets.append([top, left, bottom, right, score, int(cls)])
        
        # Extract masks if mask model
        if self.is_mask_model:
            predict_masks = predictions["instances"].pred_masks
            masks = predict_masks.cpu().numpy()
            
            # Keep original video as background, masks can be used for overlay later
            # Don't set background to black - preserve the original video
            region = image.copy()
        else:
            masks = np.array([])
            region = image.copy()
        
        return dets, np.array(masks), region
