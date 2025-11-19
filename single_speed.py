#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import medfilt

# === CONFIG ===
BASE_DIR = "/media/ben/Extreme SSD/particle-data"
DERIVATIVE_THRESHOLD = -1
SAMPLE_RATE = 100  # Hz
BASELINE_WINDOW_SEC = 4  # smoothing window duration in seconds
HOUR_BIN_SIZE = 1.0  # hours

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

# === LOAD DATA ===
cols = [
    "index","timestamp","seq","ms","motor_angle_deg","motor_speed",
    "CH0_volts","CH2_volts","CH3_volts","ellipse_angle_deg",
    "ellipse_area_px2","frame_name","ch2_dv/dt","ch3_dv/dt",
    "ch2_flag","ch3_flag"
]
df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
df["ellipse_angle_derivative"] = df["ellipse_angle_deg"].diff()

# Convert timestamp → elapsed hours
t0 = df["timestamp"].iloc[0]
df["elapsed_hours"] = (df["timestamp"] - t0) / 3600.0
df["hour_bin"] = np.floor(df["elapsed_hours"] / HOUR_BIN_SIZE).astype(int)

# === PEAK DETECTION ===
peaks = []
i = 1
while i < len(df):
    if df.loc[i, "ellipse_angle_derivative"] <= DERIVATIVE_THRESHOLD:
        j = i - 1
        while j > 0 and df.loc[j - 1, "ellipse_angle_deg"] >= df.loc[j, "ellipse_angle_deg"]:
            j -= 1
        peaks.append(df.loc[j])
        while i < len(df) - 1 and df.loc[i + 1, "ellipse_angle_deg"] <= df.loc[i, "ellipse_angle_deg"]:
            i += 1
    i += 1

result_df = pd.DataFrame(peaks).drop_duplicates(subset="index")
result_df = result_df[result_df["ellipse_angle_deg"] <= 70]
result_df["hour_bin"] = np.floor((result_df["timestamp"] - t0) / 3600.0).astype(int)

peaks_csv = os.path.join(run_dir, "fall_local_maxima_hourly.csv")
result_df.to_csv(peaks_csv, index=False)
print(f"✅ Saved → {peaks_csv} ({len(result_df)} local maxima)")

# === HOURLY SUMMARY ===
if len(result_df) > 0:
    summary = (
        result_df.groupby("hour_bin")
        .agg(
            num_points=("ellipse_angle_deg", "size"),
            angle_max=("ellipse_angle_deg", "max"),
            angle_min=("ellipse_angle_deg", "min"),
            angle_median=("ellipse_angle_deg", "median"),
            angle_mean=("ellipse_angle_deg", "mean"),
        )
        .reset_index()
    )
    summary_csv = os.path.join(run_dir, "hourly_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"✅ Saved → {summary_csv}")

# === PLOTS ===
plot_dir = run_dir

# 1️⃣ Angle vs Time (peaks only)
plt.figure(figsize=(10, 5))
plt.scatter(result_df["elapsed_hours"], result_df["ellipse_angle_deg"], color="red", s=25, label="Detected Peaks")
plt.xlabel("Elapsed Time (hours)")
plt.ylabel("Ellipse Angle (°)")
plt.title(f"{material_name} — Angle of Repose vs Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_angle_vs_time.png"))
plt.close()

# 2️⃣ Peak Count per Hour
peak_counts = result_df.groupby("hour_bin")["ellipse_angle_deg"].count().reset_index(name="num_peaks")
plt.figure(figsize=(8,5))
plt.bar(peak_counts["hour_bin"], peak_counts["num_peaks"], color="tab:orange")
plt.xlabel("Hour Bin")
plt.ylabel("Peak Count")
plt.title(f"{material_name} — Peaks per Hour")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_peak_counts_hourly.png"))
plt.close()

# 3️⃣ Charge Variability (std deviation per Hour)
kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE)
if kernel_size % 2 == 0:
    kernel_size += 1
df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

# --- Compute std deviation per hour bin ---
charge_std = (
    df.groupby("hour_bin")
    .apply(lambda g: np.sqrt(g["CH2_clean"].std()**2 + g["CH3_clean"].std()**2))
    .reset_index(name="charge_std")
)

# --- Plot it ---
plt.figure(figsize=(8,5))
plt.bar(charge_std["hour_bin"], charge_std["charge_std"], color="tab:blue")
plt.xlabel("Hour Bin")
plt.ylabel("Combined Std Dev (volts)")
plt.title(f"{material_name} — Charge Variability per Hour (baseline corrected)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_charge_std_hourly.png"))
plt.close()

# 📈 Mean Angle ± Std Dev per Hour
# 📈 Mean Angle ± Std Dev per Hour (using peaks only)
angle_stats = (
    result_df.groupby("hour_bin")["ellipse_angle_deg"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(8,5))
plt.errorbar(
    angle_stats["hour_bin"],
    angle_stats["mean"],
    yerr=angle_stats["std"],
    fmt="o-", color="tab:red", ecolor="gray", elinewidth=1.2, capsize=4,
    label="Mean ± 1σ (peaks)"
)
plt.xlabel("Hour Bin")
plt.ylabel("Ellipse Angle (°)")
plt.title(f"{material_name} — Hourly Mean Angle of Repose (±1σ, peaks only)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_mean_angle_std_hourly.png"))
plt.close()

