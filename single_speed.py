#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import medfilt, find_peaks

# === CONFIG ===
BASE_DIR = "F:/"
SAMPLE_RATE = 100   # Hz
BASELINE_WINDOW_SEC = 4   # smoothing window duration in seconds
HOUR_BIN_SIZE = 1   # hours (For the CSV summary tables)

# --- NEW PLOT SETTING ---
PLOT_BIN_MINUTES = 5  # <--- CHANGE THIS to 5, 10, 15, etc. for the Dual-Axis Graph

# --- CONFIG FROM PREVIOUS CHAT ---
MIN_HEIGHT = 10       # Min angle to be considered a peak
PROMINENCE = 3.5      # How much it must drop to count (The "Avalanche" threshold)
NOISE_THRESHOLD = 0.005 # 5mV threshold for "Active Fraction"

if len(sys.argv) < 2:
    print("Usage: python3 main_hourly.py <Material/RunFolder>")
    sys.exit(1)

rel_path = sys.argv[1]
run_dir = os.path.join(BASE_DIR, rel_path)
input_csv = os.path.join(run_dir, "experiment_log.csv")

if not os.path.isfile(input_csv):
    print(f"❌ Error: {input_csv} not found")
    sys.exit(1)

# === OUTPUT PATHS ===
csv_dir = os.path.join(run_dir, "csv_files")
graphs_dir = os.path.join(run_dir, "graphs")

os.makedirs(csv_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

plot_dir = graphs_dir
run_name = os.path.basename(run_dir)
material_name = run_name.replace("-", " ")

# === LOAD DATA ===
cols = [
    "index","timestamp","seq","ms","motor_angle_deg","motor_speed",
    "CH0_volts","CH2_volts","CH3_volts","ellipse_angle_deg",
    "ellipse_area_px2","frame_name","ch2_dv/dt","ch3_dv/dt",
    "ch2_flag","ch3_flag"
]
print("⏳ Loading Data...")
df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
df["ellipse_angle_derivative"] = df["ellipse_angle_deg"].diff()

# Time Calculations
t0 = df["timestamp"].iloc[0]
df["elapsed_hours"] = (df["timestamp"] - t0) / 3600.0
df["elapsed_minutes"] = (df["timestamp"] - t0) / 60.0 # Needed for minute binning

# Create the "Hour Bucket" for the CSV stats
df["hour_bin"] = np.floor(df["elapsed_hours"] / HOUR_BIN_SIZE).astype(int)

# === STEP 1: PEAK DETECTION ===
print("🏔️ Detecting Peaks...")

peak_indices, properties = find_peaks(
    df["ellipse_angle_deg"], 
    height=MIN_HEIGHT, 
    prominence=PROMINENCE
)

result_df = df.iloc[peak_indices].copy()
result_df = result_df[result_df["ellipse_angle_deg"] <= 70] # Safety filter

# Save Peaks
peaks_csv = os.path.join(csv_dir, "fall_local_maxima_hourly.csv")
result_df.to_csv(peaks_csv, index=False)
print(f"✅ Saved → {peaks_csv} ({len(result_df)} local maxima)")

# === STEP 2: HOURLY ANGLE SUMMARY ===
if len(result_df) > 0:
    summary = (
        result_df.groupby("hour_bin")
        .agg(
            num_peaks=("ellipse_angle_deg", "size"),
            angle_mean=("ellipse_angle_deg", "mean"),
            angle_std=("ellipse_angle_deg", "std"),
            angle_max=("ellipse_angle_deg", "max"),
        )
        .reset_index()
    )
    summary_csv = os.path.join(csv_dir, "hourly_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"✅ Saved → {summary_csv}")

# === STEP 3: CHARGE ANALYSIS ===
print("⚡ Analyzing Charge Data...")

# 1. Establish Baseline
kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE)
if kernel_size % 2 == 0: kernel_size += 1

df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

# 2. Calculate Stats per Hour Bin
charge_stats = []
hour_bins = df["hour_bin"].unique()

for h_bin in sorted(hour_bins):
    segment = df[df["hour_bin"] == h_bin]
    if len(segment) == 0: continue
    
    ch2_std = segment["CH2_clean"].std()
    ch3_std = segment["CH3_clean"].std()

    ch2_p2p = np.percentile(segment["CH2_clean"], 99) - np.percentile(segment["CH2_clean"], 1)
    ch3_p2p = np.percentile(segment["CH3_clean"], 99) - np.percentile(segment["CH3_clean"], 1)

    ch2_active_pct = ((segment["CH2_clean"].abs() > NOISE_THRESHOLD).sum() / len(segment)) * 100
    ch3_active_pct = ((segment["CH3_clean"].abs() > NOISE_THRESHOLD).sum() / len(segment)) * 100

    charge_stats.append({
        "hour_bin": h_bin,
        "ch2_std": ch2_std,
        "ch3_std": ch3_std,
        "ch2_p2p": ch2_p2p,
        "ch3_p2p": ch3_p2p,
        "ch2_active_pct": ch2_active_pct,
        "ch3_active_pct": ch3_active_pct
    })

charge_df = pd.DataFrame(charge_stats)
charge_csv = os.path.join(csv_dir, "hourly_charge_stats.csv")
charge_df.to_csv(charge_csv, index=False)
print(f"✅ Saved → {charge_csv}")

# === STEP 4: DUAL AXIS PLOT (Angle & Charge vs Time) ===
print(f"📈 Generating Dual-Axis Plot (Bin: {PLOT_BIN_MINUTES} mins)...")

# 1. Create Bins for Plotting
df["plot_bin"] = np.floor(df["elapsed_minutes"] / PLOT_BIN_MINUTES).astype(int)
# Re-calculate elapsed minutes for result_df (peaks)
result_df["elapsed_minutes"] = (result_df["timestamp"] - t0) / 60.0
result_df["plot_bin"] = np.floor(result_df["elapsed_minutes"] / PLOT_BIN_MINUTES).astype(int)

# 2. Aggregate Charge (from Raw Data)
plot_charge = (
    df.groupby("plot_bin")
    .agg(ch2_std=("CH2_clean", "std"))
    .reset_index()
)

# 3. Aggregate Angle (from Peak Data)
plot_angle = (
    result_df.groupby("plot_bin")
    .agg(angle_mean=("ellipse_angle_deg", "mean"))
    .reset_index()
)

# 4. Merge
plot_data = pd.merge(plot_charge, plot_angle, on="plot_bin", how="outer").sort_values("plot_bin")
plot_data["time_axis_mins"] = plot_data["plot_bin"] * PLOT_BIN_MINUTES

# 5. Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# --- Left Axis (Angle) ---
color_angle = 'tab:red'
ax1.set_xlabel(f'Time (Minutes) - {PLOT_BIN_MINUTES}m Bins')
ax1.set_ylabel('Mean Angle of Repose (°)', color=color_angle, fontweight='bold')
ax1.plot(plot_data["time_axis_mins"], plot_data["angle_mean"], color=color_angle, marker='o', markersize=5, linestyle='-', linewidth=2, label="Angle")
ax1.tick_params(axis='y', labelcolor=color_angle)
ax1.grid(True, alpha=0.3)

# --- Right Axis (Charge) ---
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color_charge = 'tab:blue'
ax2.set_ylabel('Charge Signal (CH2 Std Dev)', color=color_charge, fontweight='bold')
ax2.plot(plot_data["time_axis_mins"], plot_data["ch2_std"], color=color_charge, marker='s', markersize=5, linestyle='--', linewidth=2, label="Charge (CH2)")
ax2.tick_params(axis='y', labelcolor=color_charge)

plt.title(f"{material_name}: Angle vs Charge Over Time")
fig.tight_layout()  # otherwise the right y-label is slightly clipped

dual_plot_path = os.path.join(plot_dir, f"{run_name}_dual_axis_trend.png")
plt.savefig(dual_plot_path)
plt.close()
print(f"📊 Saved Dual-Axis Plot → {dual_plot_path}")


# === STANDARD PLOTS (Originals) ===

# 1️⃣ Angle vs Time (Scatter of Peaks)
plt.figure(figsize=(10, 5))
plt.scatter(result_df["elapsed_hours"], result_df["ellipse_angle_deg"], color="red", s=15, alpha=0.6, label="Detected Peaks")
plt.xlabel("Elapsed Time (hours)")
plt.ylabel("Ellipse Angle (°)")
plt.title(f"{material_name} — Angle of Repose vs Time")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_angle_vs_time.png"))
plt.close()

# 2️⃣ Peak Count per Hour (Bar)
if 'summary' in locals():
    plt.figure(figsize=(8,5))
    plt.bar(summary["hour_bin"], summary["num_peaks"], color="tab:orange", alpha=0.8)
    plt.xlabel("Hour Bin")
    plt.ylabel("Peak Count")
    plt.title(f"{material_name} — Avalanche Frequency per Hour")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_name}_peak_counts_hourly.png"))
    plt.close()

# 3️⃣ Charge Analysis: RAW DATA
plt.figure(figsize=(10, 6))
plt.plot(charge_df["hour_bin"], charge_df["ch2_std"], 'o-', color='tab:blue', label="CH2 Std Dev")
plt.plot(charge_df["hour_bin"], charge_df["ch3_std"], 's-', color='tab:red', label="CH3 Std Dev")
plt.xlabel("Time (Hours)")
plt.ylabel("Signal Energy (Std Dev)")
plt.title(f"{material_name} — Charge Evolution over Time (Raw)")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_hourly_charge_raw.png"))
plt.close()

print("✅ Processing complete.")