#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import medfilt, find_peaks

# === CONFIG ===
BASE_DIR = "F:/particle-data"
SAMPLE_RATE = 100   # Hz
BASELINE_WINDOW_SEC = 4   # smoothing window duration in seconds
HOUR_BIN_SIZE = 1.0   # hours

# --- NEW CONFIG FROM PREVIOUS CHAT ---
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

run_name = os.path.basename(run_dir)
material_name = run_name.replace("-", " ")
plot_dir = run_dir

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

# Convert timestamp → elapsed hours
t0 = df["timestamp"].iloc[0]
df["elapsed_hours"] = (df["timestamp"] - t0) / 3600.0
# Create the "Bucket" column based on time
df["hour_bin"] = np.floor(df["elapsed_hours"] / HOUR_BIN_SIZE).astype(int)

# === STEP 1: PEAK DETECTION (Upgraded to find_peaks) ===
print("🏔️ Detecting Peaks...")

peak_indices, properties = find_peaks(
    df["ellipse_angle_deg"], 
    height=MIN_HEIGHT, 
    prominence=PROMINENCE
)

result_df = df.iloc[peak_indices].copy()
result_df = result_df[result_df["ellipse_angle_deg"] <= 70] # Safety filter

# Save Peaks
peaks_csv = os.path.join(run_dir, "fall_local_maxima_hourly.csv")
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
    summary_csv = os.path.join(run_dir, "hourly_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"✅ Saved → {summary_csv}")

# === STEP 3: CHARGE ANALYSIS (Upgraded Metrics) ===
print("⚡ Analyzing Charge Data...")

# 1. Establish Baseline (Median Filter)
kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE)
if kernel_size % 2 == 0: kernel_size += 1

# Apply filter to whole dataset
df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

# 2. Loop through Hourly Bins to calculate stats
charge_stats = []
# Get list of all hour bins present in the data
hour_bins = df["hour_bin"].unique()

for h_bin in sorted(hour_bins):
    # Slice data for this hour
    segment = df[df["hour_bin"] == h_bin]
    if len(segment) == 0: continue
    
    # --- METRIC 1: Standard Deviation (Energy) ---
    ch2_std = segment["CH2_clean"].std()
    ch3_std = segment["CH3_clean"].std()

    # --- METRIC 2: Robust Peak-to-Peak (99th - 1st Percentile) ---
    ch2_p2p = np.percentile(segment["CH2_clean"], 99) - np.percentile(segment["CH2_clean"], 1)
    ch3_p2p = np.percentile(segment["CH3_clean"], 99) - np.percentile(segment["CH3_clean"], 1)

    # --- METRIC 3: Active Fraction (Time Spent Reading Particles) ---
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
charge_csv = os.path.join(run_dir, "hourly_charge_stats.csv")
charge_df.to_csv(charge_csv, index=False)
print(f"✅ Saved → {charge_csv}")

# === PLOTS ===

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

# 4️⃣ Charge Analysis: FITS & EQUATIONS
plt.figure(figsize=(10, 6))

def plot_trend(x, y, color, label_name, ax):
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = m * x_fit + b
        sign = "+" if b >= 0 else "-"
        eq = f"{label_name}: y = {m:.5f}x {sign} {abs(b):.4f}"
        ax.plot(x_fit, y_fit, linewidth=2, color=color, label=eq)
        ax.scatter(x, y, color=color, alpha=0.2) # faint dots

plot_trend(charge_df["hour_bin"], charge_df["ch2_std"], 'tab:blue', "CH2 Trend", plt.gca())
plot_trend(charge_df["hour_bin"], charge_df["ch3_std"], 'tab:red', "CH3 Trend", plt.gca())

plt.xlabel("Time (Hours)")
plt.ylabel("Linear Trend (Std Dev)")
plt.title(f"{material_name} — Charge Trends over Time")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_hourly_charge_fits.png"))
plt.close()

print("✅ Processing complete.")