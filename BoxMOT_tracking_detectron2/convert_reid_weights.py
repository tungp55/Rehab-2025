"""
Convert ReID weights from .pth to .pt format for BoxMOT compatibility
"""
import torch
from pathlib import Path

def convert_pth_to_pt(pth_path, pt_path=None):
    """
    Convert .pth file to .pt format
    
    Args:
        pth_path: Path to .pth file
        pt_path: Output .pt path (if None, replaces .pth with .pt)
    """
    if pt_path is None:
        pt_path = str(pth_path).replace('.pth', '.pt')
    
    print(f"Converting {pth_path} to {pt_path}...")
    
    # Load .pth file
    weights = torch.load(pth_path, map_location='cpu')
    
    # Save as .pt
    torch.save(weights, pt_path)
    
    print(f"✅ Converted: {pt_path}")
    return pt_path

def main():
    reid_dir = Path(__file__).parent / "reID_weight"
    
    if not reid_dir.exists():
        print(f"❌ ReID weights directory not found: {reid_dir}")
        return
    
    # Find all .pth files
    pth_files = list(reid_dir.glob("*.pth"))
    
    if not pth_files:
        print("✅ No .pth files to convert")
        return
    
    print(f"Found {len(pth_files)} .pth files to convert:")
    for pth_file in pth_files:
        pt_file = reid_dir / (pth_file.stem + ".pt")
        if pt_file.exists():
            print(f"⏭️  Skipping {pth_file.name} (already converted)")
        else:
            convert_pth_to_pt(pth_file, pt_file)
    
    print("\n✅ Conversion completed!")

if __name__ == "__main__":
    main()
