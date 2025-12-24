#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import find_peaks

# === CONFIG ===
BASE_DIR = "F:/particle-data"
SAMPLE_RATE = 100 
AXIS_BUFFER_PCT = 0.20 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5
TOP_PERCENT_CHARGE = 20

def avg_top_percent(series):
    abs_series = series.abs()
    if len(abs_series) == 0: return 0
    q = 1.0 - (TOP_PERCENT_CHARGE / 100.0)
    cutoff = abs_series.quantile(q)
    filtered = abs_series[abs_series >= cutoff]
    return filtered.mean() if len(filtered) > 0 else 0

def process_and_plot_single_run(top_dir, metadata):
    output_dir = os.path.join(top_dir, "analysis_comparative_metrics")
    os.makedirs(output_dir, exist_ok=True)
    master_csv_path = os.path.join(output_dir, "master_comparison_data.csv")
    
    all_minute_data = []
    material_folders = [f.path for f in os.scandir(top_dir) if f.is_dir() and "analysis" not in f.name]

    for material_folder in material_folders:
        material_name = os.path.basename(material_folder)
        trial_folders = [f.path for f in os.scandir(material_folder) if f.is_dir()]

        for trial_folder in trial_folders:
            input_csv = os.path.join(trial_folder, "experiment_log.csv")
            if not os.path.isfile(input_csv): continue

            cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
                    "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
                    "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]

            df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
            
            # 1. Peak Detection (Avalanches)
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
            angle_df = df.iloc[peak_indices].copy()
            angle_df = angle_df[angle_df["ellipse_angle_deg"] <= 70]

            # 2. Time Binning
            t0 = df["timestamp"].iloc[0]
            df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)
            angle_df["minute_bin"] = ((angle_df["timestamp"] - t0) / 60).astype(int)

            # 3. Aggregation
            melted_df = df.melt(id_vars=["minute_bin", "motor_speed"], value_vars=["CH2_volts", "CH3_volts"], value_name="raw_volts")
            charge_per_minute = melted_df.groupby("minute_bin").agg(
                voltage_std=("raw_volts", "std"),
                top_mag=("raw_volts", avg_top_percent),
                motor_speed=("motor_speed", "mean"),
                sample_count=("raw_volts", "count")
            ).reset_index()
            
            # Filtering and Collapse Counting
            min_samples = SAMPLE_RATE * MIN_SECONDS_PER_BIN * 2 
            charge_per_minute = charge_per_minute[charge_per_minute["sample_count"] >= min_samples]
            
            angle_per_minute = angle_df.groupby("minute_bin").agg(
                angle_mean=("ellipse_angle_deg", "mean"),
                collapse_count=("ellipse_angle_deg", "count") # <--- COUNTING COLLAPSES
            ).reset_index()

            # 4. Merge
            minute_data = pd.merge(charge_per_minute, angle_per_minute, on="minute_bin", how="left")
            minute_data["material"] = material_name
            minute_data["trial_id"] = os.path.basename(trial_folder)
            
            if not minute_data.empty:
                all_minute_data.append(minute_data)

    if not all_minute_data:
        print("Error: No data found.")
        return

    master_df = pd.concat(all_minute_data, ignore_index=True)
    master_df["grouped_speed"] = master_df["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
    master_df.to_csv(master_csv_path, index=False)
    print(f"SUCCESS: Master CSV with Collapse Counts saved to {master_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    rel_path = sys.argv[1].replace('\\', '/')
    parts = rel_path.strip('/').split('/')
    if len(parts) < 3: sys.exit(1)
    metadata = {'vol': parts[0], 'cond': parts[1], 'dur': parts[2]}
    process_and_plot_single_run(os.path.join(BASE_DIR, rel_path), metadata)