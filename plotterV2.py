#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import find_peaks

# === CONFIG ===
BASE_DIR = "F:/particle-data/"
SAMPLE_RATE = 100 
AXIS_BUFFER_PCT = 0.20 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5
TOP_PERCENT_CHARGE = 20
# ======================================================

def avg_top_percent(series):
    abs_series = series.abs()
    if len(abs_series) == 0: return 0
    q = 1.0 - (TOP_PERCENT_CHARGE / 100.0)
    cutoff = abs_series.quantile(q)
    filtered = abs_series[abs_series >= cutoff]
    return filtered.mean() if len(filtered) > 0 else 0

def get_dynamic_limits(df, x_col, y_col, buffer_pct):
    if df.empty: return 0, 1, 0, 1 
    min_y, max_y = df[y_col].min(), df[y_col].max()
    min_x, max_x = df[x_col].min(), df[x_col].max()
    range_y, range_x = max_y - min_y, max_x - min_x
    y_min = max(0, min_y - (range_y * buffer_pct))
    y_max = max_y + (range_y * buffer_pct)
    x_min = max(0, min_x - (range_x * buffer_pct))
    x_max = max_x + (range_x * buffer_pct)
    return x_min, x_max, y_min, y_max

def generate_plots(df, speeds, materials, metadata, output_dir, metric_col, metric_label):
    plot_subdir = os.path.join(output_dir, f"Plots_{metric_col}")
    os.makedirs(plot_subdir, exist_ok=True)
    m = metadata
    full_desc = f"Vol: {m['vol']} | {m['cond']} | {m['dur']}"
    
    try:
        colors = plt.colormaps["Set1"]
    except AttributeError:
        colors = plt.cm.get_cmap("Set1", max(len(materials), len(speeds)))

    # 1. Grid Plots (By Speed - Materials compared)
    for speed in speeds:
        speed_df = df[df["grouped_speed"] == speed]
        if speed_df.empty: continue
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Speed: {speed} RPM | {full_desc}\nMetric: {metric_label}", fontsize=16)
        axs_flat = axs.flatten()

        for j, material in enumerate(materials):
            if j >= 4: break 
            ax = axs_flat[j]
            mat_speed_df = speed_df[speed_df["material"] == material]
            if mat_speed_df.empty: continue

            scatter_data = mat_speed_df.groupby(["trial_id", "minute_bin"]).agg(
                angle_mean=("angle_mean", "mean"),
                metric=(metric_col, "mean")
            ).reset_index()

            if len(scatter_data) < 2: continue
            x, y = scatter_data["metric"], scatter_data["angle_mean"]
            x_min, x_max, y_min, y_max = get_dynamic_limits(scatter_data, "metric", "angle_mean", AXIS_BUFFER_PCT)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            color_val = colors(j) if callable(colors) else colors[j]
            ax.scatter(x, y, color=color_val, alpha=0.7, s=50, edgecolors='white')

            try:
                m_fit, b_fit = np.polyfit(x, y, 1)
                x_t = np.linspace(x.min(), x.max(), 10)
                ax.plot(x_t, m_fit*x_t + b_fit, "--", color=color_val, alpha=0.8)
            except: pass
            ax.set_title(f"{material} (n={len(scatter_data)})", fontweight='bold')
            ax.set_xlabel(metric_label); ax.set_ylabel("Angle of Repose (deg)"); ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(plot_subdir, f"Grid_{speed}RPM_{metric_col}.png")); plt.close()

    # 2. Summary Plots (By Material - Speeds compared)
    for material in materials:
        mat_df = df[df["material"] == material]
        if mat_df.empty: continue
        
        plt.figure(figsize=(10, 7))
        plt.title(f"Material: {material} | {full_desc}\nMetric: {metric_label}", fontsize=14)
        
        # Calculate global limits for this specific material plot
        x_min, x_max, y_min, y_max = get_dynamic_limits(mat_df, metric_col, "angle_mean", AXIS_BUFFER_PCT)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        for j, speed in enumerate(speeds):
            speed_df = mat_df[mat_df["grouped_speed"] == speed]
            if speed_df.empty: continue
            
            scatter_data = speed_df.groupby(["trial_id", "minute_bin"]).agg(
                angle_mean=("angle_mean", "mean"), metric=(metric_col, "mean")
            ).reset_index()
            
            if scatter_data.empty: continue
            
            color_val = colors(j) if callable(colors) else colors[j]
            x, y = scatter_data["metric"], scatter_data["angle_mean"]
            
            plt.scatter(x, y, label=f"{speed} RPM", color=color_val, alpha=0.6, s=40, edgecolors='white')

            # --- Added: Line of best fit for each speed in summary plot ---
            if len(scatter_data) >= 2:
                try:
                    m_fit, b_fit = np.polyfit(x, y, 1)
                    x_t = np.linspace(x.min(), x.max(), 10)
                    plt.plot(x_t, m_fit*x_t + b_fit, "--", color=color_val, alpha=0.4)
                except: pass

        plt.xlabel(metric_label)
        plt.ylabel("Angle of Repose (deg)")
        plt.legend(title="Motor Speed", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_subdir, f"Summary_{material}_{metric_col}.png"))
        plt.close()

def process_and_plot_single_run(top_dir, metadata):
    output_dir = os.path.join(top_dir, "analysis_comparative_metrics")
    os.makedirs(output_dir, exist_ok=True)
    all_minute_data = []
    material_folders = [f.path for f in os.scandir(top_dir) if f.is_dir() and "analysis" not in f.name]

    for material_folder in material_folders:
        material_name = os.path.basename(material_folder)
        trial_folders = [f.path for f in os.scandir(material_folder) if f.is_dir()]
        for trial_folder in trial_folders:
            trial_id = os.path.basename(trial_folder)
            input_csv = os.path.join(trial_folder, "experiment_log.csv")
            if not os.path.isfile(input_csv): continue
            cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
                    "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
                    "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]
            df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
            
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
            angle_df = df.iloc[peak_indices].copy()
            angle_df = angle_df[angle_df["ellipse_angle_deg"] <= 70]
            t0 = df["timestamp"].iloc[0]
            df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)
            angle_df["minute_bin"] = ((angle_df["timestamp"] - t0) / 60).astype(int)

            melted_df = df.melt(id_vars=["minute_bin", "motor_speed"], value_vars=["CH2_volts", "CH3_volts"], value_name="raw_volts")
            charge_per_minute = melted_df.groupby("minute_bin").agg(
                voltage_std=("raw_volts", "std"),
                top_mag=("raw_volts", avg_top_percent),
                motor_speed=("motor_speed", "mean"),
                sample_count=("raw_volts", "count")
            ).reset_index()
            
            min_samples = SAMPLE_RATE * MIN_SECONDS_PER_BIN * 2 
            charge_per_minute = charge_per_minute[charge_per_minute["sample_count"] >= min_samples]
            angle_per_minute = angle_df.groupby("minute_bin").agg(angle_mean=("ellipse_angle_deg", "mean")).reset_index()

            minute_data = pd.merge(charge_per_minute, angle_per_minute, on="minute_bin", how="left")
            minute_data["material"], minute_data["trial_id"] = material_name, trial_id
            all_minute_data.append(minute_data)

    if not all_minute_data: return
    master_df = pd.concat(all_minute_data, ignore_index=True)
    plotting_df = master_df.dropna(subset=["angle_mean"]).copy()
    plotting_df["grouped_speed"] = plotting_df["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
    speeds, materials = sorted(plotting_df["grouped_speed"].unique()), plotting_df["material"].unique()

    generate_plots(plotting_df, speeds, materials, metadata, output_dir, "voltage_std", "Voltage Std Dev")
    generate_plots(plotting_df, speeds, materials, metadata, output_dir, "top_mag", f"Top {TOP_PERCENT_CHARGE}% Mag")
    print(f"Done. Files in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    rel_path = sys.argv[1].replace('\\', '/')
    parts = rel_path.strip('/').split('/')
    if len(parts) < 3: sys.exit(1)
    metadata = {'vol': parts[0], 'cond': parts[1], 'dur': parts[2]}
    process_and_plot_single_run(os.path.join(BASE_DIR, rel_path), metadata)