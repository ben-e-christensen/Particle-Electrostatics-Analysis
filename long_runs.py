#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data"
DERIVATIVE_THRESHOLD = -1
SAMPLE_RATE = 100  # Hz
WINDOW_SECONDS = 60  # 1-minute windows

# --- Parse argument ---
if len(sys.argv) < 2:
    print("Usage: python3 analyze_time_means.py <Material/RunFolder>")
    print("Example: python3 analyze_time_means.py Acrylic/Acrylic-200")
    sys.exit(1)

rel_path = sys.argv[1]
run_dir = os.path.join(BASE_DIR, rel_path)
input_csv = os.path.join(run_dir, "experiment_log.csv")

if not os.path.isfile(input_csv):
    print(f"❌ Error: {input_csv} not found")
    sys.exit(1)

# --- Output directory ---
plot_dir = run_dir
run_name = os.path.basename(run_dir)
material_name = run_name.replace("-", " ")

# --- Load CSV ---
cols = [
    "index", "timestamp", "seq", "ms",
    "motor_angle_deg", "motor_speed", "CH0_volts", "CH2_volts", "CH3_volts",
    "ellipse_angle_deg", "ellipse_area_px2", "frame_name",
    "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"
]
df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")

# --- Compute derivative and filter for angles ---
df["ellipse_angle_derivative"] = df["ellipse_angle_deg"].diff()
df = df[df["ellipse_angle_deg"].notna()]
df = df[df["ellipse_angle_deg"] <= 70]

# --- Convert timestamp to relative seconds ---
t0 = df["timestamp"].iloc[0]
df["time_sec"] = df["timestamp"] - t0

# --- Bin data by minute windows ---
df["minute_bin"] = (df["time_sec"] // WINDOW_SECONDS).astype(int)

# --- Compute mean per minute ---
minute_means = (
    df.groupby("minute_bin")
    .agg(
        time_start=("time_sec", "min"),
        time_end=("time_sec", "max"),
        angle_mean=("ellipse_angle_deg", "mean"),
        motor_mean=("motor_speed", "mean"),
    )
    .reset_index(drop=True)
)

# === Plot 1: Mean Angle per Minute ===
plt.figure(figsize=(10, 5))
plt.plot(minute_means["time_start"] / 60, minute_means["angle_mean"], marker="o", lw=2)
plt.xlabel("Time (minutes)")
plt.ylabel("Mean Ellipse Angle (°)")
plt.title(f"{material_name} — Mean Angle of Repose per Minute")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_angle_mean_per_min.png"))
plt.close()

print(f"📊 Saved → {os.path.join(plot_dir, f'{run_name}_angle_mean_per_min.png')}")

#
