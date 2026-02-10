"""
BoxMOT Tracking với Detectron2 Package
Hỗ trợ nhiều thuật toán tracking với Detectron2 segmentation models
"""

__version__ = "1.0.0"

from .detector_detectron2 import Detectron2SegmentationDetector

__all__ = ['Detectron2SegmentationDetector']
