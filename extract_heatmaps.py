import pandas as pd
import numpy as np
import cv2
import sys
import os

# ================= CONFIGURATION =================
ROI_CENTER = (753, 485)  # (X, Y)
ROI_RADIUS = 505
IMAGE_SUBFOLDER = "images" 
SPEED_ROUNDING = 0 
ORDER_KEYS = ["500", "750", "1000"]
# =================================================

def apply_circular_mask(img):
    """Zeros out everything outside the hardcoded circle."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, ROI_CENTER, ROI_RADIUS, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def process_volume_group(volume_folder_path, vol_key, cache_dir):
    print(f"\nScanning Volume: {vol_key} in {os.path.basename(volume_folder_path)}...")
    
    # 1. Look for all trial folders inside the volume directory
    trial_folders = [f.path for f in os.scandir(volume_folder_path) if f.is_dir()]
    
    if not trial_folders:
        print(f"  -> No trial subfolders found inside {volume_folder_path}")
        return
        
    accumulators = {}
    counts = {}

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]

    # 2. Iterate through every single trial folder
    for trial_dir in trial_folders:
        csv_path = os.path.join(trial_dir, "experiment_log.csv")
        if not os.path.exists(csv_path):
            continue
            
        print(f"  -> Crunching {os.path.basename(trial_dir)}...")
        
        try:
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            
            # --- Fallback: Match images by timestamp if CSV is missing frame names ---
            if df["frame_name"].isnull().all():
                image_dir = os.path.join(trial_dir, IMAGE_SUBFOLDER)
                if not os.path.exists(image_dir): image_dir = os.path.join(trial_dir, "frames")
                
                if os.path.exists(image_dir):
                    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
                    img_data = []
                    for img in image_files:
                        try:
                            ts_ms = int(img.split('_')[0])
                            img_data.append({"actual_frame": img, "timestamp": ts_ms / 1000.0})
                        except ValueError:
                            continue 
                            
                    if img_data:
                        img_df = pd.DataFrame(img_data).sort_values("timestamp")
                        df = df.sort_values("timestamp")
                        df = pd.merge_asof(df, img_df, on="timestamp", direction="nearest", tolerance=0.1)
                        df["frame_name"] = df["actual_frame"]
            
            df = df.dropna(subset=["frame_name"])
            max_val = df["motor_speed"].max()
            max_indices = df.index[df["motor_speed"] == max_val].tolist()
            
            if not max_indices: continue
                
            mid_idx = max_indices[len(max_indices) // 2]
            df["direction"] = "Increasing"
            df.loc[mid_idx + 1:, "direction"] = "Decreasing"
            df["grouped_speed"] = df["motor_speed"].round(SPEED_ROUNDING).astype(int)
            
        except Exception as e: 
            print(f"  -> ERROR reading CSV in {trial_dir}: {e}")
            continue

        unique_speeds = sorted(df["grouped_speed"].unique())
        unique_speeds = [s for s in unique_speeds if s >= 1]
        
        img_dir = os.path.join(trial_dir, IMAGE_SUBFOLDER)
        if not os.path.exists(img_dir):
            img_dir = os.path.join(trial_dir, "frames")
            if not os.path.exists(img_dir): continue

        # 3. Process the images for this specific trial
        for speed in unique_speeds:
            for direction in ["Increasing", "Decreasing"]:
                subset = df[(df["grouped_speed"] == speed) & (df["direction"] == direction)]
                if subset.empty: continue
                
                frames = subset["frame_name"].unique()
                
                # Setup global accumulator for this speed/direction across ALL trials
                state_key = (speed, direction)
                if state_key not in accumulators:
                    accumulators[state_key] = None
                    counts[state_key] = 0
                
                for fname in frames:
                    fpath = os.path.join(img_dir, str(fname).strip())
                    if not os.path.exists(fpath): continue
                    
                    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    
                    img = apply_circular_mask(img) 
                    
                    if accumulators[state_key] is None: 
                        accumulators[state_key] = np.zeros_like(img, dtype=np.float32)
                    
                    _, mask = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)
                    accumulators[state_key] += mask
                    counts[state_key] += 1

    # 4. Save all accumulated heatmaps for this volume
    cached_count = 0
    for (speed, direction), acc in accumulators.items():
        if acc is not None and counts[(speed, direction)] > 0:
            heatmap = (acc / counts[(speed, direction)]) * 100.0
            safe_name = f"{vol_key}_{speed}_{direction}.npy"
            save_path = os.path.join(cache_dir, safe_name)
            np.save(save_path, heatmap)
            cached_count += 1
            
    print(f"  -> Finished caching {cached_count} true ensemble heatmaps for Volume {vol_key}.")

def run_extraction(parent_dir):
    normalized_path = parent_dir.replace('\\', '/').rstrip('/')
    material_name = os.path.basename(normalized_path)
    
    print(f"=== Starting Heatmap Extraction for: {material_name} ===")
    
    cache_dir = os.path.join(parent_dir, "heatmap_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory created at: {cache_dir}")

    all_subs = [f.path for f in os.scandir(parent_dir) if f.is_dir() and "heatmap_cache" not in f.name]
    
    target_folders = {}
    for folder in all_subs:
        fname = os.path.basename(folder)
        for key in ORDER_KEYS:
            if key.lower() in fname.lower() and "clean" not in fname.lower():
                target_folders[key] = folder

    if not target_folders:
        print("No target volume folders found matching ORDER_KEYS.")
        return

    for key in ORDER_KEYS:
        if key in target_folders:
            process_volume_group(target_folders[key], key, cache_dir)
            
    print("\n=== Extraction Complete! ===")
    print(f"Your .npy files are ready in: {cache_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_heatmaps.py <path_to_PARENT_folder>")
    else:
        run_extraction(sys.argv[1])