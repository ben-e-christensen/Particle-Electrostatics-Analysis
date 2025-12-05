#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import medfilt, find_peaks
from scipy.stats import binned_statistic_2d

# === CONFIG ===
BASE_DIR = "F:/particle-data"
SAMPLE_RATE = 100  # Hz
BASELINE_WINDOW_SEC = 4  # smoothing window duration

# !!! CHANGE THIS VALUE TO ADJUST THE GRAPH !!!
TOP_PERCENT_CHARGE = 20  # e.g., 20 for Top 20% of values

# --- Parse argument ---
if len(sys.argv) < 2:
    print("Usage: python3 main.py <Material/RunFolder>")
    sys.exit(1)

rel_path = sys.argv[1]
parent_dir = os.path.join(BASE_DIR, rel_path)

if not os.path.isdir(parent_dir):
    print(f"❌ Error: {parent_dir} is not a directory")
    sys.exit(1)

# === OUTPUT PATHS ===
output_dir = os.path.join(parent_dir, "aggregated_results")
os.makedirs(output_dir, exist_ok=True)
plot_dir = output_dir 

run_name = os.path.basename(os.path.normpath(parent_dir))
material_name = run_name.replace("-", " ")

print(f"📂 Analyzing Group: {run_name}")

# Find all subfolders containing experiment_log.csv
subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]

# Storage
all_summaries = []      # For Angle Sweep (Forward/Backward)
all_minute_data = []    # For Contour/Heatmaps (Combined Channels)

# === HELPER FUNCTIONS ===
def avg_top_percent(series):
    abs_series = series.abs()
    if len(abs_series) == 0: return 0
    q = 1.0 - (TOP_PERCENT_CHARGE / 100.0)
    cutoff = abs_series.quantile(q)
    return abs_series[abs_series >= cutoff].mean() if len(abs_series[abs_series >= cutoff]) > 0 else 0

# =========================================================
# 🔄 MASTER LOOP
# =========================================================

for trial_folder in subfolders:
    input_csv = os.path.join(trial_folder, "experiment_log.csv")
    if not os.path.isfile(input_csv): continue

    trial_name = os.path.basename(trial_folder)
    print(f"   running trial: {trial_name}...")

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]
    
    df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
    
    # --- 1. PRE-PROCESSING FOR ANGLE SWEEP ---
    peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=3.5)
    result_df = df.iloc[peak_indices].copy()
    result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

    if len(result_df) > 0:
        # Identify Speed Changes
        result_df["speed_change"] = result_df["motor_speed"].ne(result_df["motor_speed"].shift())
        result_df["run_id"] = result_df["speed_change"].cumsum()

        # Legacy 26 RPM split fix (if needed for your specific dataset)
        mask26 = (result_df["motor_speed"] == 26)
        count26 = mask26.sum()
        if count26 > 0:
            idxs_26 = result_df.index[mask26].to_list()
            half = count26 // 2
            run_forward = idxs_26[:half]
            run_backward = idxs_26[half:]
            if len(run_forward) > 0 and len(run_backward) > 0:
                fwd_id = result_df.loc[run_forward[0], "run_id"]
                result_df.loc[run_backward, "run_id"] = fwd_id + 1
                result_df.loc[result_df.index > run_backward[-1], "run_id"] += 1

        # Aggregate by Speed Step
        trial_summary = (
            result_df.groupby("run_id")
            .agg(
                motor_speed=("motor_speed", "first"),
                angle_mean=("ellipse_angle_deg", "mean"),
                angle_std=("ellipse_angle_deg", "std")
            )
            .reset_index(drop=True)
        )
        
        # Determine Direction (Forward/Backward)
        trial_summary["direction"] = "Backward"
        half_pt = len(trial_summary) // 2
        trial_summary.iloc[:half_pt, trial_summary.columns.get_loc("direction")] = "Forward"
        
        all_summaries.append(trial_summary)

    # --- 2. PRE-PROCESSING FOR CONTOURS (Minute Bins) ---
    kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE) | 1
    
    df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
    df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
    
    df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
    df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

    t0 = df["timestamp"].iloc[0]
    df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)
    
    # Angle data for minutes
    angle_minutes = result_df.copy()
    angle_minutes["minute_bin"] = ((angle_minutes["timestamp"] - t0) / 60).astype(int)

    # Melt (Combine) Channels
    melted_df = df.melt(
        id_vars=["minute_bin", "motor_speed"], 
        value_vars=["CH2_clean", "CH3_clean"], 
        value_name="combined_clean"
    )

    charge_per_minute = melted_df.groupby("minute_bin").agg(
        combined_std=("combined_clean", "std"),
        combined_top_pct=("combined_clean", avg_top_percent),
        motor_speed=("motor_speed", "mean")
    )
    
    angle_per_minute = angle_minutes.groupby("minute_bin").agg(
        angle_mean=("ellipse_angle_deg", "mean")
    )
    
    minute_data = pd.merge(charge_per_minute, angle_per_minute, on="minute_bin", how="inner")
    minute_data["trial"] = trial_name
    all_minute_data.append(minute_data)

# =========================================================
# 📊 PLOTTING SECTION
# =========================================================

if not all_summaries:
    print("❌ No valid data found.")
    sys.exit(0)

print("🎨 Generating Plots...")

# Combine Data
master_angle_df = pd.concat(all_summaries, ignore_index=True)
master_minute_df = pd.concat(all_minute_data, ignore_index=True)

# --- 1. EXPORT SUMMARY STATS (For Comparison Script) ---
# Calculate Angle Stats
global_stats = master_angle_df.groupby(["motor_speed", "direction"])["angle_mean"].agg(["mean", "std"]).reset_index()

# Calculate Charge Stats (Aggregated by Speed)
charge_agg = master_minute_df.groupby("motor_speed").agg(
    charge_std=("combined_std", "mean"),
    charge_mag=("combined_top_pct", "mean")
).reset_index()

# Merge and Save
export_df = pd.merge(global_stats, charge_agg, on="motor_speed", how="left")
export_path = os.path.join(output_dir, "summary_stats.csv")
export_df.to_csv(export_path, index=False)
print(f"💾 Saved summary stats to: {export_path}")

# --- 2. PLOT: ANGLE SWEEP (Hysteresis) ---
fwd = global_stats[global_stats["direction"] == "Forward"].sort_values("motor_speed")
bwd = global_stats[global_stats["direction"] == "Backward"].sort_values("motor_speed", ascending=False)

plt.figure(figsize=(10, 6))
plt.errorbar(fwd["motor_speed"], fwd["mean"], yerr=fwd["std"], fmt="-o", capsize=5, label="Forward Sweep (0->Max)")
plt.errorbar(bwd["motor_speed"], bwd["mean"], yerr=bwd["std"], fmt="--s", capsize=5, label="Backward Sweep (Max->0)")

plt.xlabel("Motor Speed (RPM)")
plt.ylabel("Angle of Repose (°)")
plt.title(f"{material_name} — Hysteresis Sweep")
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig(os.path.join(plot_dir, "Graph_1_Angle_Sweep.png"))
plt.close()

# --- 3. PLOT: CONTOUR MAPS (Smooth) ---
def plot_combined_contour(data, z_col, title_main, cbar_label, filename):
    if len(data) < 5: return
    plt.figure(figsize=(10, 8))
    
    x = data["motor_speed"]
    y = data["angle_mean"]
    z = data[z_col]
    
    cntr = plt.tricontourf(x, y, z, levels=30, cmap="inferno")
    plt.plot(x, y, 'ko', ms=3, alpha=0.3, label="Data Points")
    
    cbar = plt.colorbar(cntr)
    cbar.set_label(cbar_label, fontsize=12)
    
    plt.xlabel("Motor Speed (RPM)")
    plt.ylabel("Angle of Repose (°)")
    plt.title(f"{material_name}\n{title_main} (Contour)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

plot_combined_contour(master_minute_df, "combined_std", "Electrostatic Phase Diagram (Noise)", "Voltage Std Dev (V)", "Graph_2_Contour_Noise.png")
plot_combined_contour(master_minute_df, "combined_top_pct", f"Electrostatic Phase Diagram (Top {TOP_PERCENT_CHARGE}%)", "Voltage (V)", "Graph_3_Contour_Magnitude.png")

# --- 4. PLOT: BINNED HEATMAPS (Histogram) ---
def plot_binned_heatmap(data, z_col, title_main, cbar_label, filename):
    if len(data) < 5: return
    
    x = data["motor_speed"]
    y = data["angle_mean"]
    z = data[z_col]
    
    # Define grid size
    x_bins = np.linspace(x.min(), x.max(), 12)  # ~2 RPM per bin
    y_bins = np.linspace(y.min(), y.max(), 12)  # ~2 Degrees per bin
    
    # Calculate statistics in bins
    ret = binned_statistic_2d(x, y, z, statistic='mean', bins=[x_bins, y_bins])
    
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    # Rotated and flipped to match standard XY plot orientation
    plt.imshow(ret.statistic.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], 
               aspect='auto', cmap="inferno")
    
    cbar = plt.colorbar()
    cbar.set_label(f"Mean {cbar_label}", fontsize=12)
    
    plt.xlabel("Motor Speed (RPM)")
    plt.ylabel("Angle of Repose (°)")
    plt.title(f"{material_name}\n{title_main} (Binned Heatmap)")
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

plot_binned_heatmap(master_minute_df, "combined_std", "Electrostatic Distribution (Noise)", "Voltage Std Dev (V)", "Graph_4_Heatmap_Noise.png")
plot_binned_heatmap(master_minute_df, "combined_top_pct", f"Electrostatic Distribution (Top {TOP_PERCENT_CHARGE}%)", "Voltage (V)", "Graph_5_Heatmap_Magnitude.png")

print(f"🚀 Done! Saved stats CSV + 5 graphs to: {output_dir}")