#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import medfilt, find_peaks

# === CONFIG ===
# !!! WARNING: UPDATE THIS PATH TO MATCH YOUR SYSTEM !!!
BASE_DIR = "D:/particle-data/"
# ======================================================
SAMPLE_RATE = 100 # Hz
BASELINE_WINDOW_SEC = 4 # smoothing window duration
AXIS_BUFFER_PCT = 0.15 # Add 15% buffer to min/max axis limits
SPEED_ROUNDING_PRECISION = 0 # Round motor speed to the nearest integer
MIN_SECONDS_PER_BIN = 30 # Drop minute bins with < 30 seconds of data

# --- FIXED PROMINENCE LEVEL ---
FIXED_PROMINENCE = 1.5 
# ------------------------------

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_dynamic_limits(df, x_col, y_col, buffer_pct):
    """Calculates min/max limits for plotting with a buffer."""
    if df.empty:
        return 0, 1, 0, 1 # Default limits if no data
        
    min_y = df[y_col].min()
    max_y = df[y_col].max()
    range_y = max_y - min_y
    
    min_x = df[x_col].min()
    max_x = df[x_col].max()
    range_x = max_x - min_x
    
    # Calculate buffered limits
    y_min = min_y - (range_y * buffer_pct)
    y_max = max_y + (range_y * buffer_pct)
    x_min = min_x - (range_x * buffer_pct)
    x_max = max_x + (range_x * buffer_pct)
    
    # Ensure minimums don't go below zero
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    
    # Handle cases where range is zero (all points are the same)
    if range_y < 1e-6:
        y_min = min_y * 0.9 if min_y > 0 else 0
        y_max = max_y * 1.1 if max_y > 0 else 1
    if range_x < 1e-6:
        x_min = min_x * 0.9 if min_x > 0 else 0
        x_max = max_x * 1.1 if max_x > 0 else 1

    return x_min, x_max, y_min, y_max

def plot_by_speed_comparison(df, speeds, materials, run_name, prominence_level, output_dir, buffer_pct):
    """Generates 6 PNGs, one per speed, comparing all 4 materials."""
    p_str = str(prominence_level).replace('.', '_')
    plot_subdir = os.path.join(output_dir, "Plots_By_Speed_Comparative")
    os.makedirs(plot_subdir, exist_ok=True)
    
    try:
        colors = plt.colormaps["Set1"]
    except AttributeError:
        colors = plt.cm.get_cmap("Set1", len(materials))

    print("\n--- Generating Plots By Speed (Comparative) ---")

    for i, speed in enumerate(speeds):
        speed_df = df[df["grouped_speed"] == speed]

        if speed_df.empty:
            print(f"    Skipping {speed} RPM: No data points.")
            continue
        
        # --- GLOBAL AXIS CALCULATION (for this speed) ---
        x_min_global, x_max_global, y_min, y_max = get_dynamic_limits(speed_df, "charge_std", "angle_mean", buffer_pct)
        
        plt.figure(figsize=(10, 7))
        plt.title(f"{run_name} | Angle vs. Charge Trend (P={prominence_level})\nMotor Speed: {speed} RPM (Comparative)")
        plt.xlabel(r"Charge Noise (Std Dev of $\Delta$V)")
        plt.ylabel("Angle of Repose (°)")

        plt.xlim(x_min_global, x_max_global)
        plt.ylim(y_min, y_max)

        for j, material in enumerate(materials):
            mat_speed_df = speed_df[speed_df["material"] == material]
            
            if mat_speed_df.empty: continue
            
            scatter_data = mat_speed_df.groupby("minute_bin").agg(
                angle_mean=("angle_mean", "mean"),
                charge_std=("charge_std", "mean")
            ).reset_index()

            if len(scatter_data) < 2: continue

            color_val = colors(j) if callable(colors) else colors[j]
            x, y = scatter_data["charge_std"], scatter_data["angle_mean"]

            # 1. SCATTER PLOT
            plt.scatter(x, y, label=material, color=color_val, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)

            # 2. LINE OF BEST FIT (CLIPPED)
            try:
                m, b = np.polyfit(x, y, 1)
                # --- FIX: Clip line to material's data range ---
                x_trend = np.linspace(x.min(), x.max(), 10) 
                plt.plot(x_trend, m*x_trend + b, linestyle="--", color=color_val, alpha=0.8, linewidth=1.5)
            except Exception as e:
                pass 

        plt.legend(title="Material (with Best Fit Line)")
        plt.grid(True, alpha=0.4)

        filename = f"Comp_Scatter_Speed_{speed}RPM_P{p_str}.png"
        plt.savefig(os.path.join(plot_subdir, filename))
        plt.close()
        print(f"    Saved comparative plot for {speed} RPM.")


def plot_by_material_grid(df, speeds, materials, run_name, prominence_level, output_dir, buffer_pct):
    """Generates 4 PNGs, one per material, with all speeds on the plot."""
    p_str = str(prominence_level).replace('.', '_')
    plot_subdir = os.path.join(output_dir, "Plots_By_Material_Grid")
    os.makedirs(plot_subdir, exist_ok=True)
    
    try:
        colors = plt.colormaps["Set1"]
    except AttributeError:
        colors = plt.cm.get_cmap("Set1", len(speeds)) # Color now represents speed

    print("\n--- Generating Plots By Material (All Speeds on Grid) ---")
    
    for i, material in enumerate(materials):
        mat_df = df[df["material"] == material]
        
        if mat_df.empty:
            print(f"    Skipping {material}: No data points.")
            continue

        # --- GLOBAL AXIS CALCULATION (for this material) ---
        x_min_global, x_max_global, y_min, y_max = get_dynamic_limits(mat_df, "charge_std", "angle_mean", buffer_pct)
        
        plt.figure(figsize=(10, 7))
        plt.title(f"{run_name} | Angle vs. Charge Trend (P={prominence_level})\nMaterial: {material} (All Speeds)")
        plt.xlabel(r"Charge Noise (Std Dev of $\Delta$V)")
        plt.ylabel("Angle of Repose (°)")

        plt.xlim(x_min_global, x_max_global)
        plt.ylim(y_min, y_max)
        
        for j, speed in enumerate(speeds):
            speed_df = mat_df[mat_df["grouped_speed"] == speed]
            
            if speed_df.empty: continue
            
            scatter_data = speed_df.groupby("minute_bin").agg(
                angle_mean=("angle_mean", "mean"),
                charge_std=("charge_std", "mean")
            ).reset_index()

            if len(scatter_data) < 2: continue

            color_val = colors(j) if callable(colors) else colors[j]
            x, y = scatter_data["charge_std"], scatter_data["angle_mean"]

            # 1. SCATTER PLOT
            plt.scatter(x, y, label=f"{speed} RPM", color=color_val, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)

            # 2. LINE OF BEST FIT (CLIPPED)
            try:
                m, b = np.polyfit(x, y, 1)
                # --- FIX: Clip line to material's data range for this speed ---
                x_trend = np.linspace(x.min(), x.max(), 10) 
                plt.plot(x_trend, m*x_trend + b, linestyle="--", color=color_val, alpha=0.8, linewidth=1.5)
            except Exception as e:
                pass 

        plt.legend(title="Motor Speed (with Best Fit Line)")
        plt.grid(True, alpha=0.4)

        filename = f"Mat_Scatter_{material}_P{p_str}.png"
        plt.savefig(os.path.join(plot_subdir, filename))
        plt.close()
        print(f"    Saved material plot for {material}.")


# =========================================================
# ANALYSIS FUNCTION (Called once)
# =========================================================

def process_and_plot_single_run(top_dir, run_name, prominence_level):
    print(f"\n==================================================================")
    print(f"--- ⚙️ STARTING SINGLE ANALYSIS FOR PROMINENCE: {prominence_level} ---")
    print(f"==================================================================")

    # === DYNAMIC OUTPUT PATHS ===
    p_str = str(prominence_level).replace('.', '_')
    output_dir = os.path.join(top_dir, f"comparative_analysis_P{p_str}")
    os.makedirs(output_dir, exist_ok=True)
    master_csv_path = os.path.join(output_dir, "master_angle_charge_data.csv")
    
    all_minute_data = []

    # Exclude output folders
    material_folders = [f.path for f in os.scandir(top_dir) 
                        if f.is_dir() and not f.name.startswith("comparative_analysis")]

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

            # --- FIND ANGLE OF REPOSE PEAKS ---
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=prominence_level)
            result_df = df.iloc[peak_indices].copy()
            result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

            if len(result_df) == 0: continue

            # --- PRE-PROCESSING FOR CHARGE (Minute Bins) ---
            kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE) | 1

            df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
            df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
            df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
            df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

            t0 = df["timestamp"].iloc[0]
            df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)

            angle_minutes = result_df.copy()
            angle_minutes["minute_bin"] = ((angle_minutes["timestamp"] - t0) / 60).astype(int)

            melted_df = df.melt(id_vars=["minute_bin", "motor_speed"], value_vars=["CH2_clean", "CH3_clean"], value_name="combined_clean")

            charge_per_minute = melted_df.groupby("minute_bin").agg(
                charge_std=("combined_clean", "std"),
                motor_speed=("motor_speed", "mean"),
                sample_count=("combined_clean", "count")
            ).reset_index()
            
            min_samples = SAMPLE_RATE * MIN_SECONDS_PER_BIN * 2 
            charge_per_minute = charge_per_minute[charge_per_minute["sample_count"] >= min_samples]

            angle_per_minute = angle_minutes.groupby("minute_bin").agg(
                angle_mean=("ellipse_angle_deg", "mean")
            ).reset_index()

            minute_data = pd.merge(charge_per_minute, angle_per_minute, on="minute_bin", how="inner")
            minute_data["material"] = material_name
            all_minute_data.append(minute_data)

    if not all_minute_data:
        print(f"❌ No valid data found for Prominence {prominence_level}.")
        return

    master_minute_df = pd.concat(all_minute_data, ignore_index=True)
    master_minute_df["motor_speed_rounded"] = master_minute_df["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
    final_df = master_minute_df.rename(columns={"motor_speed_rounded": "grouped_speed"})
    final_df.to_csv(master_csv_path, index=False)
    print(f"💾 Master data saved to: **{master_csv_path}**")

    # --- SETUP FOR PLOTTING ---
    speeds = sorted(final_df["grouped_speed"].unique())
    materials = final_df["material"].unique()

    # =========================================================
    # PART 2: PLOTTING - DUAL OUTPUT
    # =========================================================

    # 2a. Plots comparing all materials at one speed
    plot_by_speed_comparison(final_df, speeds, materials, run_name, prominence_level, output_dir, AXIS_BUFFER_PCT)
    
    # 2b. Plots showing all speeds for one material
    plot_by_material_grid(final_df, speeds, materials, run_name, prominence_level, output_dir, AXIS_BUFFER_PCT)

    print(f"\n✅ Analysis complete for Prominence {prominence_level}. Outputs saved to: **{output_dir}**")

# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 Particle_Analyzer.py <Top_Level_RunFolder_Path>")
        print("Example: python3 Particle_Analyzer.py 500/Dirty/12mins")
        sys.exit(1)

    rel_path = sys.argv[1]
    top_dir = os.path.join(BASE_DIR, rel_path)

    if not os.path.isdir(top_dir):
        print(f"❌ Error: {top_dir} is not a directory")
        sys.exit(1)

    run_name = os.path.basename(os.path.normpath(top_dir))

    print(f"--- 🚀 Starting Single-Prominence, Dual-Output Analysis for Group: {run_name} ---")

    # Call the single run function with the fixed prominence
    process_and_plot_single_run(top_dir, run_name, FIXED_PROMINENCE)

    print(f"\n--- ALL ANALYSES COMPLETE ---")