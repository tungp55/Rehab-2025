"""
Inspect Mask R-CNN architecture (Detectron2): number of layers, bias usage.
Uses same config as batch_all_trackers.py: mask_rcnn_R_50_FPN_3x.yaml.

Run: conda activate rehab && python BoxMOT_tracking_detectron2/inspect_maskrcnn_architecture.py
(From repo root; or from BoxMOT_tracking_detectron2: python inspect_maskrcnn_architecture.py)
"""
import os
import sys
from pathlib import Path

# Same path setup as batch_all_trackers.py / detector_detectron2.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DETECTRON2_ROOT = REPO_ROOT / "bach_mask_rcnn" / "detectron2"
if DETECTRON2_ROOT.is_dir():
    sys.path.insert(0, str(DETECTRON2_ROOT))
else:
    sys.path.insert(0, "/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2")

from detectron2.config import get_cfg
from detectron2.modeling import build_model


def count_layers_and_bias(model):
    """Count parameterized layers and which have bias."""
    conv2d_total = 0
    conv2d_with_bias = 0
    linear_total = 0
    linear_with_bias = 0
    bn_total = 0
    other_param_layers = 0
    backbone_conv = 0
    backbone_conv_bias = 0
    head_conv = 0
    head_conv_bias = 0
    head_linear = 0
    head_linear_bias = 0

    for name, m in model.named_modules():
        if not any(p is not None for p in m.parameters(recurse=False)):
            continue
        in_backbone = "backbone" in name or "stem" in name or "res2" in name or "res3" in name or "res4" in name or "res5" in name
        in_head = "roi_heads" in name or "rpn" in name or "proposal" in name

        if m.__class__.__name__ == "Conv2d":
            conv2d_total += 1
            has_bias = getattr(m, "bias", None) is not None
            if has_bias:
                conv2d_with_bias += 1
            if in_backbone:
                backbone_conv += 1
                if has_bias:
                    backbone_conv_bias += 1
            elif in_head:
                head_conv += 1
                if has_bias:
                    head_conv_bias += 1
        elif m.__class__.__name__ == "Linear":
            linear_total += 1
            has_bias = getattr(m, "bias", None) is not None
            if has_bias:
                linear_with_bias += 1
            if in_head:
                head_linear += 1
                if has_bias:
                    head_linear_bias += 1
        elif "BatchNorm" in m.__class__.__name__ or "GroupNorm" in m.__class__.__name__:
            bn_total += 1
        else:
            other_param_layers += 1

    return {
        "conv2d_total": conv2d_total,
        "conv2d_with_bias": conv2d_with_bias,
        "linear_total": linear_total,
        "linear_with_bias": linear_with_bias,
        "bn_total": bn_total,
        "other_param_layers": other_param_layers,
        "backbone_conv": backbone_conv,
        "backbone_conv_bias": backbone_conv_bias,
        "head_conv": head_conv,
        "head_conv_bias": head_conv_bias,
        "head_linear": head_linear,
        "head_linear_bias": head_linear_bias,
    }


def run_config(config_path, num_classes=4):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Build without loading weights
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    model = build_model(cfg)
    model.eval()

    depth = cfg.MODEL.RESNETS.DEPTH
    backbone_name = cfg.MODEL.BACKBONE.NAME
    stats = count_layers_and_bias(model)
    return depth, backbone_name, stats, cfg


# Default config path (same as batch_all_trackers.py)
DEFAULT_CONFIG = (
    "/home/vuhai/Rehab-Tung/bach_mask_rcnn/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)


def main():
    config_path = DEFAULT_CONFIG
    if not os.path.isfile(config_path):
        # Resolve relative to repo root
        config_path = str(DETECTRON2_ROOT / "configs" / "COCO-InstanceSegmentation" / "mask_rcnn_R_50_FPN_3x.yaml")
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        return

    print("=" * 70)
    print("Mask R-CNN (Detectron2) architecture — mask_rcnn_R_50_FPN_3x.yaml")
    print("(Same config as batch_all_trackers.py)")
    print("=" * 70)

    depth, backbone_name, stats, _ = run_config(config_path)
    print(f"\n  Config: {os.path.basename(config_path)}")
    print(f"  Backbone: ResNet-{depth} (RESNETS.DEPTH={depth})")
    print(f"  Backbone build: {backbone_name}")
    blocks = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}.get(depth)
    if blocks:
        expected = 1 + sum(blocks) * 3
        print(f"  Backbone conv layers: {stats['backbone_conv']} (expected {expected}: 1 stem + 4 stages {blocks} blocks × 3 convs)")
    print(f"  Backbone Conv2d: {stats['backbone_conv']} total, {stats['backbone_conv_bias']} with bias")
    print(f"  Head Conv2d: {stats['head_conv']} total, {stats['head_conv_bias']} with bias")
    print(f"  Head Linear: {stats['head_linear']} total, {stats['head_linear_bias']} with bias")
    print(f"  Overall: Conv2d {stats['conv2d_total']} (bias: {stats['conv2d_with_bias']}), Linear {stats['linear_total']} (bias: {stats['linear_with_bias']}), BN/GN {stats['bn_total']}")
    print("\n  => ResNet backbone convs: no bias (BatchNorm). FPN adds extra convs. Heads: bias in Linear (cls_score, bbox_pred) and in mask predictor conv.")

    print("\n" + "=" * 70)
    print("Conclusion for paper (Concern #4):")
    print("  - Config mask_rcnn_R_50_FPN_3x.yaml → ResNet-50 (DEPTH=50): 1 stem + 16 BottleneckBlocks (3+4+6+3) × 3 conv = 49 conv in ResNet; FPN adds more convs (total backbone Conv2d above).")
    print("  - ResNet backbone: Conv2d layers use bias=False (BatchNorm).")
    print("  - ROI heads: FastRCNN uses nn.Linear (bias=True); Mask head uses Conv2d predictor with bias (default).")
    print("=" * 70)


if __name__ == "__main__":
    main()
