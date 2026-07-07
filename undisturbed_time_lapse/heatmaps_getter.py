import pandas as pd
import numpy as np
import cv2
import sys
import os

# ================= CONFIGURATION =================
ROI_CENTER = (753, 485)  # (X, Y)
ROI_RADIUS = 505
IMAGE_SUBFOLDER = "images" 
TARGET_SPEED = 16 # Locked to 16 RPM
DAY_KEYS = [f"day{i}" for i in range(1, 8)]

# --- Mechanical-issue exclusion windows (cumulative hours across all days) ---
# These must be computed with the SAME cumulative time-stitching logic used
# here (day1 -> day7, in order) or the hour ranges won't line up.
# EXCLUSION_WINDOWS_HOURS = [ (27.1, 42.0), (76.50, 86.0),]
# =================================================

def is_excluded(time_hours_array, exclusion_windows):
    """Returns a boolean mask, True where the timestamp falls OUTSIDE all
    exclusion windows (i.e. True = keep this row)."""
    keep_mask = np.ones(len(time_hours_array), dtype=bool)
    for start_hr, end_hr in exclusion_windows:
        bad = (time_hours_array >= start_hr) & (time_hours_array <= end_hr)
        keep_mask &= ~bad
    return keep_mask

def apply_circular_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, ROI_CENTER, ROI_RADIUS, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def process_day_group(day_folder_path, day_key, cache_dir, cumulative_ms):
    print(f"\nScanning {day_key} in {os.path.basename(day_folder_path)}...")
    
    csv_path = os.path.join(day_folder_path, "experiment_log.csv")
    if not os.path.exists(csv_path):
        print(f"  -> No experiment_log.csv found directly in {day_folder_path}")
        return cumulative_ms
        
    print(f"  -> Crunching 16 RPM data...")
    
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]

    try:
        df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
        
        # Match images by timestamp if CSV is missing frame names
        if df["frame_name"].isnull().all():
            image_dir = os.path.join(day_folder_path, IMAGE_SUBFOLDER)
            if not os.path.exists(image_dir): image_dir = os.path.join(day_folder_path, "frames")
            
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
        df["grouped_speed"] = df["motor_speed"].round(0).astype(int)

        # --- Cumulative time stitching (must match continuous_ensemble.py logic) ---
        df = df.dropna(subset=["ms"]).sort_values("ms")
        start_ms = df["ms"].min()
        df["Continuous_ms"] = (df["ms"] - start_ms) + cumulative_ms
        new_cumulative_ms = df["Continuous_ms"].max() + 10.0
        df["rel_time_hours"] = df["Continuous_ms"] / 3600000.0

        # --- Exclude the mechanical-issue windows ---
        pre_count = len(df)
        keep_mask = is_excluded(df["rel_time_hours"].values, EXCLUSION_WINDOWS_HOURS)
        df = df[keep_mask]
        excluded_count = pre_count - len(df)
        if excluded_count > 0:
            print(f"  -> Excluded {excluded_count} rows falling inside mechanical-issue windows")
        
    except Exception as e: 
        print(f"  -> ERROR reading CSV: {e}")
        return cumulative_ms

    img_dir = os.path.join(day_folder_path, IMAGE_SUBFOLDER)
    if not os.path.exists(img_dir):
        img_dir = os.path.join(day_folder_path, "frames")
        if not os.path.exists(img_dir): 
            print(f"  -> Could not find images folder in {day_folder_path}.")
            return new_cumulative_ms

    # Filter purely for the target speed
    subset = df[df["grouped_speed"] == TARGET_SPEED]
    if subset.empty:
        print(f"  -> No data found at {TARGET_SPEED} RPM for {day_key}.")
        return new_cumulative_ms
        
    frames = subset["frame_name"].unique()
    
    accumulator = None
    count = 0
    
    for fname in frames:
        fpath = os.path.join(img_dir, str(fname).strip())
        if not os.path.exists(fpath): continue
        
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        img = apply_circular_mask(img) 
        
        if accumulator is None: 
            accumulator = np.zeros_like(img, dtype=np.float32)
        
        _, mask = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)
        accumulator += mask
        count += 1

    if accumulator is not None and count > 0:
        heatmap = (accumulator / count) * 100.0
        safe_name = f"{day_key}_{TARGET_SPEED}RPM.npy"
        save_path = os.path.join(cache_dir, safe_name)
        np.save(save_path, heatmap)
        print(f"  -> Saved {safe_name} (Averaged over {count} frames)")
    else:
        print(f"  -> Failed to generate heatmap for {day_key}.")

    return new_cumulative_ms

def run_longitudinal_extraction(parent_dir):
    normalized_path = parent_dir.replace('\\', '/').rstrip('/')
    material_name = os.path.basename(normalized_path)
    
    print(f"=== Starting Constant-Speed Heatmap Extraction for: {material_name} ===")
    
    cache_dir = os.path.join(parent_dir, "heatmap_cache_longitudinal")
    os.makedirs(cache_dir, exist_ok=True)

    all_subs = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    
    target_folders = {key: folder for folder in all_subs for key in DAY_KEYS if key.lower() == os.path.basename(folder).lower()}

    # cumulative_ms MUST be threaded through days 1 -> 7 in order, matching
    # the stitching logic used to originally derive EXCLUSION_WINDOWS_HOURS.
    cumulative_ms = 0.0
    for key in DAY_KEYS:
        if key in target_folders:
            cumulative_ms = process_day_group(target_folders[key], key, cache_dir, cumulative_ms)
            
    print("\n=== Extraction Complete! ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_longitudinal_heatmaps.py <path_to_PARENT_folder>")
    else:
        run_longitudinal_extraction(sys.argv[1])