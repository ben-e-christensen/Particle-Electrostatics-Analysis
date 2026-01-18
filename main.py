#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import re
from scipy.signal import find_peaks

# === CONFIG ===
BASE_DIR = "F:/particle-data/"
SAMPLE_RATE = 100 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def parse_duration_minutes(dur_str: str, default: int = 60) -> int:
    if not dur_str: return default
    m = re.search(r"(\d+)", str(dur_str))
    return int(m.group(1)) if m else default

# ======================================================
# PLOTTING FUNCTIONS
# ======================================================

def generate_temporal_grids(df, speeds, materials, metadata, output_dir):
    """Minute-by-minute bins for all speeds, color-coded by time."""
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
            mat_speed_df = speed_df[speed_df["material"] == material.lower()]
            if mat_speed_df.empty: continue
                
            # Plotting each minute-bin independently within this speed/material
            x, y = mat_speed_df["voltage_std"], mat_speed_df["angle_mean"]
            sc = ax.scatter(x, y, c=mat_speed_df["minute_bin"], cmap="coolwarm", 
                            vmin=1, vmax=time_max, alpha=0.8, s=60, edgecolors='black')
            
            if len(mat_speed_df) >= 2:
                m, b = np.polyfit(x, y, 1)
                ax.plot(x, m*x + b, "--", color="black", alpha=0.4)
                
            ax.set_title(f"{material.capitalize()} (Minutes: {len(mat_speed_df)})")
            ax.set_xlabel("Std Dev Voltage (V)", fontweight='bold')
            ax.set_ylabel("Angle of Repose (deg)", fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        if sc: 
            fig.colorbar(sc, ax=axs, label="Elapsed Time (min)", shrink=0.8)
            
        plt.savefig(os.path.join(plot_subdir, f"Grid_{speed}RPM.png"))
        plt.close()

def generate_whole_run_summaries(df, materials, output_dir):
    plot_subdir = os.path.join(output_dir, "Plots_Summaries")
    os.makedirs(plot_subdir, exist_ok=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for material in materials:
        mat_df = df[df["material"] == material.lower()]
        if mat_df.empty: continue
        
        plt.figure(figsize=(11, 8))
        speeds = sorted(mat_df["grouped_speed"].unique())
        for i, speed in enumerate(speeds):
            s_df = mat_df[mat_df["grouped_speed"] == speed]
            c = colors[i % len(colors)]
            plt.scatter(s_df["voltage_std"], s_df["angle_mean"], label=f"{speed} RPM", color=c, alpha=0.6, s=50)
            if len(s_df) >= 2:
                m, b = np.polyfit(s_df["voltage_std"], s_df["angle_mean"], 1)
                plt.plot(s_df["voltage_std"], m*s_df["voltage_std"]+b, color=c, ls="--", alpha=0.4)
                
        plt.title(f"Whole Run Summary: {material.capitalize()}", pad=20)
        plt.xlabel("Std Dev Voltage (V)", fontweight='bold')
        plt.ylabel("Angle of Repose (deg)", fontweight='bold')
        plt.legend(title="RPM", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_subdir, f"Summary_{material}.png"))
        plt.close()

def generate_hysteresis_grid(df, materials, output_dir, y_col, y_label, filename):
    plot_subdir = os.path.join(output_dir, "Plots_Hysteresis")
    os.makedirs(plot_subdir, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"{y_label} Hysteresis Analysis", fontsize=16)
    axs_flat = axs.flatten()
    
    for i, material in enumerate(materials):
        if i >= 4: break
        ax = axs_flat[i]
        mat_df = df[df["material"] == material.lower()]
        if mat_df.empty: continue

        # Hysteresis aggregates by speed and direction (average of all minutes in that phase)
        h_stats = mat_df.groupby(['grouped_speed', 'direction']).agg(y_avg=(y_col, 'mean'), y_std=(y_col, 'std')).unstack()
        
        for direction, color in [('Increasing', '#1f77b4'), ('Decreasing', '#d62728')]:
            if direction in h_stats['y_avg'].columns:
                data = h_stats.xs(direction, axis=1, level=1).dropna()
                ax.errorbar(data.index, data['y_avg'], yerr=data['y_std'], fmt='o-', 
                            color=color, label=direction, capsize=5, lw=2, markersize=8)
        
        ax.set_title(f"{material.capitalize()}", fontweight='bold')
        ax.set_xlabel("Motor Speed (RPM)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.savefig(os.path.join(plot_subdir, f"{filename}.png"))
    plt.close()

# ======================================================
# CORE PROCESSING
# ======================================================

def process_run(rel_path, metadata):
    top_dir = os.path.join(BASE_DIR, rel_path)
    folder_name = f"{metadata['vol']}-{metadata['cond']}-{metadata['dur']}-graphs"
    output_dir = os.path.join(top_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    all_minute_data = []
    material_folders = [f.path for f in os.scandir(top_dir) if f.is_dir() and "analysis" not in f.name and "-graphs" not in f.name]

    for material_folder in material_folders:
        material_name = os.path.basename(material_folder).lower()
        for trial_folder in [f.path for f in os.scandir(material_folder) if f.is_dir()]:
            input_csv = os.path.join(trial_folder, "experiment_log.csv")
            if not os.path.isfile(input_csv): continue
            
            cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]
            df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
            
            # 1. UNIVERSAL DIRECTION SPLIT
            max_val = df["motor_speed"].max()
            max_indices = df.index[df["motor_speed"] == max_val].tolist()
            mid_plateau_idx = max_indices[len(max_indices) // 2]
            
            df["direction"] = "Increasing"
            df.loc[mid_plateau_idx + 1:, "direction"] = "Decreasing" 
            
            # 2. CREATE MINUTE BINS
            t0 = df["timestamp"].iloc[0]
            df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int) + 1
            
            # 3. IDENTIFY PEAKS FOR ANGLE
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
            angle_df = df.iloc[peak_indices].copy()
            
            # 4. AGGREGATE BY MINUTE, DIRECTION, AND SPEED
            # This ensures Grids by Speed have one data point per minute per speed phase
            charge_agg = df.melt(id_vars=["minute_bin", "direction", "motor_speed"], 
                                 value_vars=["CH2_volts", "CH3_volts"]).groupby(["minute_bin", "direction", "motor_speed"]).agg(
                voltage_std=("value", "std"), 
                sample_count=("value", "count")).reset_index()
            
            angle_agg = angle_df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                angle_mean=("ellipse_angle_deg", "mean")).reset_index()

            # Filter for valid data density
            charge_agg = charge_agg[charge_agg["sample_count"] >= (SAMPLE_RATE * MIN_SECONDS_PER_BIN)]
            
            minute_data = pd.merge(charge_agg, angle_agg, on=["minute_bin", "direction", "motor_speed"], how="left")
            minute_data["material"] = material_name
            all_minute_data.append(minute_data)

    if not all_minute_data: return
        
    master_df = pd.concat(all_minute_data, ignore_index=True).dropna(subset=["angle_mean"])
    master_df["grouped_speed"] = master_df["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
    master_df = master_df[master_df["grouped_speed"] >= 1]
    
    master_df.to_csv(os.path.join(output_dir, "master_comparison_data.csv"), index=False)
    mats = sorted(master_df["material"].unique())
    speeds = sorted(master_df["grouped_speed"].unique())
    
    # GENERATE PLOTS
    generate_temporal_grids(master_df, speeds, mats, metadata, output_dir)
    generate_whole_run_summaries(master_df, mats, output_dir)
    generate_hysteresis_grid(master_df, mats, output_dir, "angle_mean", "Angle of Repose (deg)", "Hysteresis_Angle_2x2")
    generate_hysteresis_grid(master_df, mats, output_dir, "voltage_std", "Std Dev Voltage (V)", "Hysteresis_Charge_2x2")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    run_path = sys.argv[1].replace('\\', '/').strip('/')
    parts = run_path.split('/')
    meta = {'vol': parts[0], 'cond': parts[1], 'dur': parts[2], 'time_max_min': parse_duration_minutes(parts[2])}
    process_run(run_path, meta)