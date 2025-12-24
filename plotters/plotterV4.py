#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import re
from scipy.signal import find_peaks

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data/"
SAMPLE_RATE = 100 
AXIS_BUFFER_PCT = 0.20 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5

# ======================================================

def parse_duration_minutes(dur_str: str, default: int = 60) -> int:
    if not dur_str: return default
    m = re.search(r"(\d+)", str(dur_str))
    return int(m.group(1)) if m else default

def get_dynamic_limits(df, x_col, y_col, buffer_pct):
    if df.empty: return 0, 1, 0, 1 
    min_y, max_y = df[y_col].min(), df[y_col].max()
    min_x, max_x = df[x_col].min(), df[x_col].max()
    range_y, range_x = max_y - min_y, max_x - min_x
    return (max(0, min_x - (range_x * buffer_pct)), max_x + (range_x * buffer_pct),
            max(0, min_y - (range_y * buffer_pct)), max_y + (range_y * buffer_pct))

# === PLOTTING FUNCTIONS ===

def generate_temporal_grids(df, speeds, materials, metadata, output_dir):
    """Produces the 2x2 grids per speed, time-colored."""
    plot_subdir = os.path.join(output_dir, "Plots_Grids_By_Speed")
    os.makedirs(plot_subdir, exist_ok=True)
    time_max = int(metadata.get("time_max_min", 60))
    
    for speed in speeds:
        speed_df = df[df["grouped_speed"] == speed]
        if speed_df.empty: continue
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        fig.suptitle(f"Speed: {speed} RPM | {metadata['vol']} {metadata['cond']}", fontsize=16)
        axs_flat = axs.flatten()
        sc = None

        for j, material in enumerate(materials):
            if j >= 4: break 
            ax = axs_flat[j]
            mat_speed_df = speed_df[speed_df["material"] == material]
            if mat_speed_df.empty: continue

            x, y = mat_speed_df["voltage_std"], mat_speed_df["angle_mean"]
            x_min, x_max, y_min, y_max = get_dynamic_limits(mat_speed_df, "voltage_std", "angle_mean", AXIS_BUFFER_PCT)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            
            sc = ax.scatter(x, y, c=mat_speed_df["minute_bin"], cmap="coolwarm", 
                            vmin=1, vmax=time_max, alpha=0.8, s=60, edgecolors='black')

            if len(mat_speed_df) >= 2:
                m_fit, b_fit = np.polyfit(x, y, 1)
                ax.plot(x, m_fit*x + b_fit, "--", color="black", alpha=0.4)
            
            ax.set_title(f"{material} (n={len(mat_speed_df)})")
            ax.set_xlabel("Voltage Std Dev"); ax.set_ylabel("Angle (deg)"); ax.grid(True, alpha=0.3)

        if sc is not None:
            fig.colorbar(sc, ax=axs, label=f"Time (min, 1-{time_max})", shrink=0.8)

        plt.savefig(os.path.join(plot_subdir, f"Grid_{speed}RPM_Temporal.png"))
        plt.close()

def generate_whole_run_summaries(df, materials, metadata, output_dir):
    """Produces a single summary plot per material containing all speeds."""
    plot_subdir = os.path.join(output_dir, "Plots_Whole_Run_Summaries")
    os.makedirs(plot_subdir, exist_ok=True)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for material in materials:
        mat_df = df[df["material"] == material]
        if mat_df.empty: continue
        
        plt.figure(figsize=(10, 7))
        speeds = sorted(mat_df["grouped_speed"].unique())
        
        for i, speed in enumerate(speeds):
            s_df = mat_df[mat_df["grouped_speed"] == speed]
            color = colors[i % len(colors)]
            
            plt.scatter(s_df["voltage_std"], s_df["angle_mean"], 
                        label=f"{speed} RPM", color=color, alpha=0.6, s=50, edgecolors='white')
            
            if len(s_df) >= 2:
                m, b = np.polyfit(s_df["voltage_std"], s_df["angle_mean"], 1)
                x_range = np.linspace(s_df["voltage_std"].min(), s_df["voltage_std"].max(), 10)
                plt.plot(x_range, m * x_range + b, color=color, linestyle="--", alpha=0.5)

        plt.title(f"Whole Run Summary: {material} | {metadata['vol']} {metadata['cond']}")
        plt.xlabel("Voltage Std Dev")
        plt.ylabel("Angle of Repose (deg)")
        plt.legend(title="Motor Speed (RPM)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_subdir, f"Summary_{material}.png"))
        plt.close()

def generate_hysteresis(df, materials, output_dir, y_col, y_label, filename_prefix):
    """Produces hysteresis line plots (Increasing vs Decreasing) for any given metric."""
    plot_subdir = os.path.join(output_dir, "Plots_Hysteresis")
    os.makedirs(plot_subdir, exist_ok=True)
    
    for material in materials:
        mat_df = df[df["material"] == material]
        if mat_df.empty: continue
        
        # Aggregate by speed and direction
        h_stats = mat_df.groupby(['grouped_speed', 'direction']).agg(
            y_avg=(y_col, 'mean'), y_std=(y_col, 'std')
        ).unstack()
        
        plt.figure(figsize=(10, 7))
        for direction, color in [('Increasing', 'blue'), ('Decreasing', 'red')]:
            if direction in h_stats['y_avg'].columns:
                data = h_stats.xs(direction, axis=1, level=1).dropna()
                plt.errorbar(data.index, data['y_avg'], yerr=data['y_std'], 
                             fmt='o-', color=color, label=direction, capsize=5, lw=2)
        
        plt.title(f"{y_label} Hysteresis: {material}")
        plt.xlabel("Motor Speed (RPM)"); plt.ylabel(y_label)
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_subdir, f"{filename_prefix}_{material}.png"))
        plt.close()

# === CORE PROCESSING ===

def process_and_plot_single_run(top_dir, metadata):
    # Dynamic output directory naming: e.g., 500-Dirty-12mins-graphs
    folder_name = f"{metadata['vol']}-{metadata['cond']}-{metadata['dur']}-graphs"
    output_dir = os.path.join(top_dir, folder_name)
    
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
            
            # Use columns from your "older version" for reliability
            cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
                    "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
                    "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]
            
            df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
            if df.empty: continue

            # Hysteresis Direction
            max_idx = df["motor_speed"].idxmax()
            df["direction"] = "Increasing"
            df.loc[max_idx:, "direction"] = "Decreasing"
            
            # Peak Detection
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
            angle_df = df.iloc[peak_indices].copy()
            angle_df = angle_df[angle_df["ellipse_angle_deg"] <= 70]
            
            # Time Binning
            t0 = df["timestamp"].iloc[0]
            df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int) + 1
            angle_df["minute_bin"] = ((angle_df["timestamp"] - t0) / 60).astype(int) + 1

            # Aggregation with Sample Guard
            melted_df = df.melt(id_vars=["minute_bin", "motor_speed", "direction"], 
                               value_vars=["CH2_volts", "CH3_volts"], value_name="raw_volts")
            
            charge_per_minute = melted_df.groupby(["minute_bin", "direction"]).agg(
                voltage_std=("raw_volts", "std"),
                motor_speed=("motor_speed", "mean"),
                sample_count=("raw_volts", "count")
            ).reset_index()
            
            # Quality Filter: 30s of data per bin
            min_samples = SAMPLE_RATE * MIN_SECONDS_PER_BIN * 2 
            charge_per_minute = charge_per_minute[charge_per_minute["sample_count"] >= min_samples]
            

            angle_per_minute = angle_df.groupby(["minute_bin", "direction"]).agg(
                angle_mean=("ellipse_angle_deg", "mean"),
                collapse_count=("ellipse_angle_deg", "count")  # <--- THIS IS THE NEW LINE
            ).reset_index()

            # Ensure that minutes with NO avalanches show 0 instead of NaN
            angle_per_minute["collapse_count"] = angle_per_minute["collapse_count"].fillna(0)

            # Merge
            minute_data = pd.merge(charge_per_minute, angle_per_minute, on=["minute_bin", "direction"], how="left")
            minute_data["material"], minute_data["trial_id"] = material_name, trial_id
            all_minute_data.append(minute_data)

    if not all_minute_data: return
    
    # 1. GENERATE MASTER CSV FOR CONTOUR SCRIPT
    master_df = pd.concat(all_minute_data, ignore_index=True).dropna(subset=["angle_mean"])
    master_df["grouped_speed"] = master_df["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
    master_df = master_df[master_df["grouped_speed"] >= 1]
    
    # Save to the specific path your Contour script expects
    master_csv_path = os.path.join(output_dir, "master_comparison_data.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"Master Data CSV exported to: {master_csv_path}")

    # 2. GENERATE PLOTS
    mats = master_df["material"].unique()
    speeds = sorted(master_df["grouped_speed"].unique())

    generate_temporal_grids(master_df, speeds, mats, metadata, output_dir)
    generate_whole_run_summaries(master_df, mats, metadata, output_dir)
    generate_hysteresis(master_df, mats, output_dir, "angle_mean", "Angle (deg)", "Hysteresis_Angle")
    generate_hysteresis(master_df, mats, output_dir, "voltage_std", "Voltage Std Dev", "Hysteresis_Charge")
    
    print(f"Analysis Finished. Graphs available in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    rel_path = sys.argv[1].replace('\\', '/')
    parts = rel_path.strip('/').split('/')
    if len(parts) < 3: sys.exit(1)
    
    meta = {'vol': parts[0], 'cond': parts[1], 'dur': parts[2]}
    meta["time_max_min"] = parse_duration_minutes(meta["dur"])
    process_and_plot_single_run(os.path.join(BASE_DIR, rel_path), meta)