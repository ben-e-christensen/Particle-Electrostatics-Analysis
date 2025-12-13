#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import medfilt, find_peaks

# === CONFIG ===
BASE_DIR = "F:/particle-data/"
SAMPLE_RATE = 100 # Hz
BASELINE_WINDOW_SEC = 4 # smoothing window duration
AXIS_BUFFER_PCT = 0.15 # Add 15% buffer to min/max axis limits
SPEED_ROUNDING_PRECISION = 0 # Round motor speed to the nearest integer
MIN_SECONDS_PER_BIN = 30 # Drop minute bins with < 30 seconds of data

# --- Parse argument ---
if len(sys.argv) < 2:
    print("Usage: python3 Particle_Analyzer.py <Top_Level_RunFolder_Path>")
    print("Example: python3 Particle_Analyzer.py 500/Dirty/12mins")
    sys.exit(1)

rel_path = sys.argv[1]
top_dir = os.path.join(BASE_DIR, rel_path)

if not os.path.isdir(top_dir):
    print(f"❌ Error: {top_dir} is not a directory")
    sys.exit(1)

# === OUTPUT PATHS ===
output_dir = os.path.join(top_dir, "comparative_analysis")
os.makedirs(output_dir, exist_ok=True)
master_csv_path = os.path.join(output_dir, "master_angle_charge_data.csv")
run_name = os.path.basename(os.path.normpath(top_dir))

print(f"--- 🚀 Starting Unified Analysis for Group: {run_name} ---")

# Storage for ALL materials
all_minute_data = []

# --- Find Material Subfolders ---
material_folders = [f.path for f in os.scandir(top_dir) if f.is_dir() and f.name != "comparative_analysis"]

# =========================================================
# PART 1: DATA PROCESSING AND CONSOLIDATION
# =========================================================

print("\n--- 1. DATA PROCESSING ---")

for material_folder in material_folders:
    material_name = os.path.basename(material_folder)

    trial_folders = [f.path for f in os.scandir(material_folder) if f.is_dir()]

    print(f"\n    🏭 Processing Material: **{material_name}**")

    for trial_folder in trial_folders:
        input_csv = os.path.join(trial_folder, "experiment_log.csv")
        if not os.path.isfile(input_csv): continue

        cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
                 "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
                 "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]

        df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")

        # --- 1. FIND ANGLE OF REPOSE PEAKS ---
        peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=3.5)
        result_df = df.iloc[peak_indices].copy()
        result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

        if len(result_df) == 0: continue

        # --- 2. PRE-PROCESSING FOR CHARGE (Minute Bins) ---
        kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE) | 1

        df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
        df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]

        df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
        df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

        t0 = df["timestamp"].iloc[0]
        df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)

        angle_minutes = result_df.copy()
        angle_minutes["minute_bin"] = ((angle_minutes["timestamp"] - t0) / 60).astype(int)

        melted_df = df.melt(
            id_vars=["minute_bin", "motor_speed"],
            value_vars=["CH2_clean", "CH3_clean"],
            value_name="combined_clean"
        )

        # Count samples to detect partial bins
        charge_per_minute = melted_df.groupby("minute_bin").agg(
            charge_std=("combined_clean", "std"),
            motor_speed=("motor_speed", "mean"),
            sample_count=("combined_clean", "count")
        ).reset_index()
        
        # Filter out bins with too few samples
        min_samples = SAMPLE_RATE * MIN_SECONDS_PER_BIN * 2 
        charge_per_minute = charge_per_minute[charge_per_minute["sample_count"] >= min_samples]

        angle_per_minute = angle_minutes.groupby("minute_bin").agg(
            angle_mean=("ellipse_angle_deg", "mean")
        ).reset_index()

        minute_data = pd.merge(charge_per_minute, angle_per_minute, on="minute_bin", how="inner")
        minute_data["material"] = material_name
        all_minute_data.append(minute_data)

if not all_minute_data:
    print("❌ No valid data found across all materials.")
    sys.exit(0)

master_minute_df = pd.concat(all_minute_data, ignore_index=True)

# ----------------------------------------------------
# Round the motor speed to group similar values
# ----------------------------------------------------
master_minute_df["motor_speed_rounded"] = master_minute_df["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
# ----------------------------------------------------

final_df = master_minute_df.rename(columns={"motor_speed_rounded": "grouped_speed"})

final_df.to_csv(master_csv_path, index=False)
print(f"💾 Master data saved to: **{master_csv_path}**")

# =========================================================
# PART 2: COMPARATIVE PLOTTING (Scatter + Regression)
# =========================================================

print("\n--- 2. PLOTTING COMPARISON GRAPHS ---")

df = final_df
speeds = sorted(df["grouped_speed"].unique())
materials = df["material"].unique()

try:
    colors = plt.colormaps["Set1"]
except AttributeError:
    colors = plt.cm.get_cmap("Set1", len(materials))

for i, speed in enumerate(speeds):
    speed_df = df[df["grouped_speed"] == speed]

    if speed_df.empty:
        print(f"    Skipping {speed} RPM: No data points.")
        continue
    
    # --- DYNAMIC AXIS CALCULATION ---
    min_angle = speed_df["angle_mean"].min()
    max_angle = speed_df["angle_mean"].max()
    range_angle = max_angle - min_angle
    
    min_charge = speed_df["charge_std"].min()
    max_charge = speed_df["charge_std"].max()
    range_charge = max_charge - min_charge
    
    y_min = min_angle - (range_angle * AXIS_BUFFER_PCT)
    y_max = max_angle + (range_angle * AXIS_BUFFER_PCT)
    x_min = min_charge - (range_charge * AXIS_BUFFER_PCT)
    x_max = max_charge + (range_charge * AXIS_BUFFER_PCT)
    
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    # ---------------------------------

    plt.figure(figsize=(10, 7))
    plt.title(f"{run_name} Run | Angle vs. Charge Trend\nMotor Speed: {speed} RPM")
    plt.xlabel(r"Charge Noise (Std Dev of $\Delta$V)")
    plt.ylabel("Angle of Repose (°)")

    if range_charge > 1e-6:
        plt.xlim(x_min, x_max)
    if range_angle > 1e-6:
        plt.ylim(y_min, y_max)

    for j, material in enumerate(materials):
        mat_speed_df = speed_df[speed_df["material"] == material]
        
        if mat_speed_df.empty: continue
        
        # Group by minute bin (averaging trials if multiple exist for the same minute)
        scatter_data = mat_speed_df.groupby("minute_bin").agg(
            angle_mean=("angle_mean", "mean"),
            charge_std=("charge_std", "mean")
        ).reset_index()

        if len(scatter_data) < 2:
            print(f"    ⚠️ Not enough points to plot trend for {material} at {speed} RPM")
            continue

        color_val = colors(j) if callable(colors) else colors[j]
        x = scatter_data["charge_std"]
        y = scatter_data["angle_mean"]

        # 1. SCATTER PLOT (Individual Points)
        plt.scatter(
            x, y,
            label=material,
            color=color_val,
            alpha=0.7,
            s=40, # Marker size
            edgecolors='white',
            linewidth=0.5
        )

        # 2. LINE OF BEST FIT (Linear Regression)
        # np.polyfit(x, y, 1) returns [slope, intercept]
        try:
            m, b = np.polyfit(x, y, 1)
            # Create a line range based on the min/max X values for this material
            x_trend = np.linspace(x.min(), x.max(), 10)
            plt.plot(x_trend, m*x_trend + b, linestyle="--", color=color_val, alpha=0.8, linewidth=1.5)
        except Exception as e:
            print(f"    Could not fit line for {material}: {e}")

    plt.legend(title="Material (with Best Fit Line)")
    plt.grid(True, alpha=0.4)

    filename = f"Comp_Scatter_Speed_{speed}RPM.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"    Saved plot for {speed} RPM.")

print(f"\n✅ Analysis complete. All outputs saved to: **{output_dir}**")