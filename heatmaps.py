import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import math

# ================= CONFIGURATION =================
IMAGE_SUBFOLDER = "images" 
OUTPUT_FOLDER_NAME = "Ensemble_Results"
SPEED_ROUNDING = 0 
COLS_PER_PAGE = 3           
# =================================================

def process_ensemble_heatmaps(parent_dir):
    print(f"--- Starting Heatmap Analysis for: {parent_dir} ---")
    
    # 1. DISCOVER TRIALS
    trial_folders = []
    for root, dirs, files in os.walk(parent_dir):
        if "experiment_log.csv" in files:
            trial_folders.append(root)
    
    if not trial_folders:
        print("No trials found!")
        return

    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 2. LOAD METADATA
    print("Loading Trial Metadata...")
    master_df = pd.DataFrame()
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]

    for trial_path in trial_folders:
        try:
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            df = df.dropna(subset=["frame_name"])
            
            # Split Directions
            max_val = df["motor_speed"].max()
            max_indices = df.index[df["motor_speed"] == max_val].tolist()
            if not max_indices: continue
            mid_idx = max_indices[len(max_indices) // 2]
            
            df["direction"] = "Increasing"
            df.loc[mid_idx + 1:, "direction"] = "Decreasing"
            df["grouped_speed"] = df["motor_speed"].round(SPEED_ROUNDING).astype(int)
            df["source_trial"] = trial_path
            
            master_df = pd.concat([master_df, df], ignore_index=True)
        except: continue

    if master_df.empty: return

    # 3. GENERATE HEATMAPS
    unique_speeds = sorted(master_df["grouped_speed"].unique())
    unique_speeds = [s for s in unique_speeds if s >= 1]
    
    heatmap_store = {}

    for speed in unique_speeds:
        heatmap_store[speed] = {}
        for direction in ["Increasing", "Decreasing"]:
            
            relevant_rows = master_df[(master_df["grouped_speed"] == speed) & 
                                      (master_df["direction"] == direction)]
            if relevant_rows.empty: continue
            
            # Group by trial to handle file paths correctly
            trial_groups = relevant_rows.groupby("source_trial")["frame_name"].unique()
            
            global_accumulator = None
            total_valid_frames = 0
            
            print(f"Processing {speed} RPM ({direction})...")
            
            for trial_path, frames in trial_groups.items():
                # Find image folder
                img_dir = os.path.join(trial_path, IMAGE_SUBFOLDER)
                if not os.path.exists(img_dir):
                    if os.path.exists(os.path.join(trial_path, "frames")):
                        img_dir = os.path.join(trial_path, "frames")
                    else: continue

                for fname in frames:
                    full_path = os.path.join(img_dir, str(fname).strip())
                    if not os.path.exists(full_path): continue
                    
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    
                    if global_accumulator is None:
                        global_accumulator = np.zeros_like(img, dtype=np.float32)
                    
                    _, mask = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)
                    global_accumulator += mask
                    total_valid_frames += 1
            
            if global_accumulator is not None and total_valid_frames > 0:
                avg_map = (global_accumulator / total_valid_frames) * 100.0
                heatmap_store[speed][direction] = avg_map

    # 4. GRID REPORT
    total_pages = math.ceil(len(unique_speeds) / COLS_PER_PAGE)
    for page_idx in range(total_pages):
        start_idx = page_idx * COLS_PER_PAGE
        end_idx = start_idx + COLS_PER_PAGE
        current_speeds = unique_speeds[start_idx:end_idx]
        
        fig, axes = plt.subplots(nrows=2, ncols=len(current_speeds), 
                                 figsize=(len(current_speeds)*5, 8), 
                                 constrained_layout=True, squeeze=False)
        fig.suptitle(f"Ensemble Heatmap Report: Part {page_idx + 1}", fontsize=22)
        
        shared_im = None
        for col_i, speed in enumerate(current_speeds):
            for row_i, direction in enumerate(["Increasing", "Decreasing"]):
                ax = axes[row_i, col_i]
                
                # Plot Data
                if direction in heatmap_store.get(speed, {}):
                    im = ax.imshow(heatmap_store[speed][direction], cmap='inferno', vmin=0, vmax=100)
                    if row_i == 0: 
                        ax.set_title(f"{speed} RPM", fontsize=18, fontweight='bold')
                        shared_im = im
                else: 
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')
                
                # Clean up axes (no ticks)
                ax.set_xticks([])
                ax.set_yticks([])
                # Remove border lines (spines) for cleaner look
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # --- ADD ROW LABELS (Increasing/Decreasing) ---
        # We attach these to the first column's axes so layout handles them
        axes[0, 0].set_ylabel("INCREASING \u2191", fontsize=20, fontweight='bold', labelpad=15)
        axes[1, 0].set_ylabel("DECREASING \u2193", fontsize=20, fontweight='bold', labelpad=15)
        
        # Colorbar
        if shared_im:
            cbar = fig.colorbar(shared_im, ax=axes, shrink=0.8, location='right', aspect=30)
            cbar.set_label("Frequency (%)", fontsize=14)

        report_name = f"Heatmap_Report_Part_{page_idx+1}.png"
        plt.savefig(os.path.join(output_dir, report_name), dpi=300)
        plt.close()
        print(f"Saved: {report_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_heatmaps.py <path_to_PARENT_folder>")
    else:
        process_ensemble_heatmaps(sys.argv[1])