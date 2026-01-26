import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Ensemble_Macro_Results"
HARD_CUTOFF_MIN = 60.0 # Full duration
# =================================================

def process_continuous_clean(parent_dir):
    print(f"--- Starting Continuous Clean Analysis (0-{HARD_CUTOFF_MIN} min) for: {parent_dir} ---")
    
    # 1. DISCOVER TRIALS
    trial_folders = []
    for root, dirs, files in os.walk(parent_dir):
        if "experiment_log.csv" in files:
            trial_folders.append(root)
    
    if not trial_folders:
        print("No trial folders found!")
        return

    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 2. MERGE DATA
    print("Merging Data...")
    all_data = []

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]

    for trial_path in trial_folders:
        try:
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            df = df.dropna(subset=["ellipse_angle_deg", "motor_speed", "ms"])
            
            if df.empty: continue

            # Calculate Global Time relative to start of trial
            start_ms = df["ms"].iloc[0]
            df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
            
            # Filter by Hard Cutoff
            df = df[df["rel_time_min"] <= HARD_CUTOFF_MIN]
            
            if not df.empty:
                all_data.append(df)

        except Exception as e:
            print(f"Skipping {os.path.basename(trial_path)}: {e}")

    if not all_data:
        print("No valid data found.")
        return

    master_df = pd.concat(all_data)

    # 3. PLOTTING
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_all = master_df["rel_time_min"].values
    y_all = master_df["ellipse_angle_deg"].values

    # Plot Raw Data Cloud Only
    ax.scatter(x_all, y_all, s=0.5, color='black', alpha=0.02, rasterized=True)

    # Formatting
    ax.set_title(f"Angle Sampled At 100Hz", fontsize=20, fontweight='bold')
    ax.set_xlabel("Time (min)", fontsize=14)
    ax.set_ylabel("Angle (deg)", fontsize=14)
    
    y_min, y_max = y_all.min(), y_all.max()
    buff = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - buff, y_max + buff)
    
    ax.set_xlim(0, HARD_CUTOFF_MIN)
    ax.grid(True, alpha=0.3)
    
    # NO LEGEND
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "Global_Continuous_Clean.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_60min_continuous_clean.py <path_to_PARENT_folder>")
    else:
        process_continuous_clean(sys.argv[1])