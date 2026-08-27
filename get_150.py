"""
extract_150_good.py
-------------------
Uses saved U-shape split model (front+middle+tail) to scan new_train/,
predict on each .h5 file, and copy the first 150 predicted as GOOD 
into a new folder '150_good'.

Usage:
    python extract_150_good.py

Requirements:
    - front_M01_ushape.pth
    - middle_model.pth  
    - tail_M01_ushape.pth
    - new_train/ directory with .h5 files
"""

import os
import sys
import shutil
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
MODEL_PATHS = {
    "front": Path("front_M01_ushape.pth"),
    "middle": Path("middle_model.pth"),
    "tail": Path("tail_M01_ushape.pth")
}
SOURCE_DIR = Path("new_train")
OUTPUT_DIR = Path("150_good")
TARGET_COUNT = 150
THRESHOLD = 0.5  # prob > 0.5 → GOOD

# ============================================================
# PYTORCH MODEL DEFINITIONS (Must match training exactly)
# ============================================================
class FrontModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (B, 32, 512)
        x = self.pool(self.relu(self.conv2(x)))  # (B, 64, 256)
        return x


class MiddleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))        # (B, 128, 256)
        x = self.gap(x).squeeze(-1)        # (B, 128)
        x = self.relu2(self.fc(x))         # (B, 64)
        return x


class TailModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.sigmoid(self.fc(x))       # (B, 1)
        return x


# ============================================================
# MODEL LOADING
# ============================================================
def load_models():
    """Load the three split model checkpoints."""
    front = FrontModel()
    middle = MiddleModel()
    tail = TailModel()
    
    front.load_state_dict(torch.load(MODEL_PATHS["front"], weights_only=True, map_location='cpu'))
    middle.load_state_dict(torch.load(MODEL_PATHS["middle"], weights_only=True, map_location='cpu'))
    tail.load_state_dict(torch.load(MODEL_PATHS["tail"], weights_only=True, map_location='cpu'))
    
    front.eval()
    middle.eval()
    tail.eval()
    
    print(f"✅ Models loaded:")
    print(f"   Front: {MODEL_PATHS['front']}")
    print(f"   Middle: {MODEL_PATHS['middle']}")
    print(f"   Tail: {MODEL_PATHS['tail']}")
    
    return front, middle, tail


def predict_split(front, middle, tail, vibration_data):
    """
    Run inference using split U-shape model.
    
    Args:
        vibration_data: numpy array of shape (1024, 3)
    
    Returns:
        float: prediction probability (0 to 1)
    """
    # Reshape for PyTorch Conv1d: (batch, channels, sequence) = (1, 3, 1024)
    x = torch.tensor(vibration_data.transpose(1, 0), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        act = front(x)           # (1, 64, 256)
        out = middle(act)        # (1, 64)
        pred = tail(out)         # (1, 1)
        return pred.item()


def find_h5_files(base_dir):
    """Recursively find all .h5 files in base_dir."""
    h5_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".h5"):
                h5_files.append(Path(root) / f)
    return sorted(h5_files)


def main():
    print("=" * 70)
    print("🔍 Extract 150 GOOD predictions from new_train/")
    print("=" * 70)
    
    # Check model files exist
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"❌ Missing model: {path}")
            sys.exit(1)
    
    # Check source directory
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        sys.exit(1)
    
    # Load models
    front, middle, tail = load_models()
    
    # Find all H5 files
    print(f"\n📁 Scanning {SOURCE_DIR} for .h5 files...")
    h5_files = find_h5_files(SOURCE_DIR)
    print(f"   Found {len(h5_files)} .h5 files")
    
    if not h5_files:
        print("❌ No .h5 files found. Exiting.")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Tracking
    good_count = 0
    processed = 0
    skipped = 0
    errors = 0
    
    print(f"\n🎯 Collecting {TARGET_COUNT} GOOD predictions (threshold > {THRESHOLD})...")
    print("-" * 70)
    
    # Process files with progress bar
    for file_path in tqdm(h5_files, desc="Predicting", unit="file"):
        if good_count >= TARGET_COUNT:
            break
            
        try:
            with h5py.File(file_path, 'r') as hf:
                # Load vibration data
                data = hf["vibration_data"][:]  # (1024, 3)
                
                # Predict
                prob = predict_split(front, middle, tail, data)
                is_good = prob > THRESHOLD
                
                processed += 1
                
                if is_good:
                    # Copy file to output directory
                    dest = OUTPUT_DIR / file_path.name
                    shutil.copy2(file_path, dest)
                    good_count += 1
                    tqdm.write(f"[{good_count:3d}/{TARGET_COUNT}] ✅ GOOD ({prob:.4f}): {file_path.name}")
                    
                    if good_count >= TARGET_COUNT:
                        tqdm.write(f"\n🎯 Target reached! Stopping.")
                        break
                else:
                    skipped += 1
                    
        except Exception as e:
            errors += 1
            tqdm.write(f"❌ Error processing {file_path.name}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"   Total files scanned:  {len(h5_files)}")
    print(f"   Files processed:      {processed}")
    print(f"   GOOD predictions:     {good_count}")
    print(f"   BAD predictions:      {skipped}")
    print(f"   Errors:               {errors}")
    print(f"   Output folder:        {OUTPUT_DIR.resolve()}")
    print(f"   Files in output:      {len(list(OUTPUT_DIR.glob('*.h5')))}")
    print("=" * 70)
    
    if good_count < TARGET_COUNT:
        print(f"\n⚠️  Warning: Only found {good_count} GOOD predictions (target: {TARGET_COUNT})")
        print("   Consider: lowering threshold, checking model, or scanning more data")
    else:
        print(f"\n✅ Success! {TARGET_COUNT} GOOD files saved to '{OUTPUT_DIR}'")
    
    return 0 if good_count >= TARGET_COUNT else 1


if __name__ == "__main__":
    sys.exit(main())