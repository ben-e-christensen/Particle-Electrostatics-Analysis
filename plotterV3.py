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
MIN_SECONDS_PER_BIN = 50 
FIXED_PROMINENCE = 1.5
FIXED_TIME_MAX = 60 

# ======================================================

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

def generate_temporal_speed_grids(df, speeds, materials, metadata, output_dir):
    plot_subdir = os.path.join(output_dir, "Plots_Speed_Grids_Temporal")
    os.makedirs(plot_subdir, exist_ok=True)
    
    for speed in speeds:
        speed_df = df[df["grouped_speed"] == speed]
        if speed_df.empty: continue
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Speed: {speed} RPM | Time-Colored Drift\nVol: {metadata['vol']} | {metadata['cond']}", fontsize=16)
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
                            vmin=0, vmax=FIXED_TIME_MAX, alpha=0.8, s=60, edgecolors='black')
            
            if len(mat_speed_df) >= 2:
                m_fit, b_fit = np.polyfit(x, y, 1)
                ax.plot(x, m_fit*x + b_fit, "--", color='black', alpha=0.5)
            
            ax.set_title(f"{material} (n={len(mat_speed_df)})")
            ax.set_xlabel("Voltage Std Dev"); ax.set_ylabel("Angle (deg)"); ax.grid(True, alpha=0.3)
        
        if sc is not None:
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
            fig.colorbar(sc, cax=cbar_ax, label=f"Time (Scale Fixed 0-{FIXED_TIME_MAX})")
            
        plt.savefig(os.path.join(plot_subdir, f"Grid_{speed}RPM_Temporal.png"))
        plt.close()

def generate_hysteresis_plots(df, materials, metadata, output_dir, y_col, y_label, filename_prefix):
    plot_subdir = os.path.join(output_dir, "Plots_Hysteresis")
    os.makedirs(plot_subdir, exist_ok=True)

    for material in materials:
        mat_df = df[df["material"] == material]
        if mat_df.empty: continue
        
        h_stats = mat_df.groupby(['grouped_speed', 'direction']).agg(
            y_avg=(y_col, 'mean'),
            y_std=(y_col, 'std')
        ).unstack()
        
        plt.figure(figsize=(10, 7))
        max_rpm, min_rpm = int(mat_df['grouped_speed'].max()), int(mat_df['grouped_speed'].min())

        for direction, color, label_text in [('Increasing', 'blue', f'{min_rpm}→{max_rpm}'), 
                                             ('Decreasing', 'red', f'{max_rpm}→{min_rpm}')]:
            if direction in h_stats['y_avg'].columns:
                data = h_stats.xs(direction, axis=1, level=1).dropna()
                plt.errorbar(data.index, data['y_avg'], yerr=data['y_std'], 
                             fmt='o-', color=color, label=f'{direction} ({label_text})', capsize=5, lw=2)
            
        plt.title(f"{y_label} Hysteresis: {material}")
        plt.xlabel("Motor Speed (RPM)"); plt.ylabel(y_label)
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_subdir, f"{filename_prefix}_{material}.png"))
        plt.close()

def generate_whole_run_summaries(df, materials, metadata, output_dir):
    plot_subdir = os.path.join(output_dir, "Plots_Whole_Run")
    os.makedirs(plot_subdir, exist_ok=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for material in materials:
        mat_df = df[df["material"] == material]
        plt.figure(figsize=(10, 7))
        speeds = sorted(mat_df["grouped_speed"].unique())
        
        for i, speed in enumerate(speeds):
            s_df = mat_df[mat_df["grouped_speed"] == speed]
            color = colors[i % len(colors)]
            plt.scatter(s_df["voltage_std"], s_df["angle_mean"], label=f"{speed} RPM", color=color, alpha=0.6)
            
            if len(s_df) >= 2:
                m, b = np.polyfit(s_df["voltage_std"], s_df["angle_mean"], 1)
                x_vals = np.linspace(s_df["voltage_std"].min(), s_df["voltage_std"].max(), 10)
                plt.plot(x_vals, m*x_vals + b, color=color, linestyle="--", alpha=0.8)

        plt.title(f"Whole Run Summary: {material} (Per-Speed Trends)")
        plt.xlabel("Voltage Std Dev"); plt.ylabel("Angle (deg)")
        plt.legend(title="Speed (RPM)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plot_subdir, f"Summary_{material}.png"))
        plt.close()

def process_and_plot_single_run(top_dir, metadata):
    output_dir = os.path.join(top_dir, "analysis_comparative_metrics")
    os.makedirs(output_dir, exist_ok=True)
    all_minute_data = []
    
    cols_15 = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
               "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
               "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag"]
    
    material_folders = [f.path for f in os.scandir(top_dir) if f.is_dir() and "analysis" not in f.name]

    for material_folder in material_folders:
        material_name = os.path.basename(material_folder)
        trial_folders = [f.path for f in os.scandir(material_folder) if f.is_dir()]
        for trial_folder in trial_folders:
            input_csv = os.path.join(trial_folder, "experiment_log.csv")
            if not os.path.isfile(input_csv): continue
            
            df = pd.read_csv(input_csv, header=0, on_bad_lines="skip", engine="python")
            if len(df.columns) == 15: df.columns = cols_15
            elif len(df.columns) == 16: df.columns = cols_15 + ["ch3_flag"]
            
            # 1. Tag Directions strictly to prevent double points
            peak_speed_val = df["motor_speed"].max()
            mid_peak_idx = df[df["motor_speed"] == peak_speed_val].index[len(df[df["motor_speed"] == peak_speed_val])//2]
            df["raw_direction"] = "Increasing"
            df.loc[mid_peak_idx:, "raw_direction"] = "Decreasing"

            # 2. Assign Time Bins
            t0 = df["timestamp"].iloc[0]
            df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)

            # 3. FIX: Ensure each minute has only ONE direction (the majority direction)
            # This eliminates the "10% extra" ghost points.
            direction_map = df.groupby("minute_bin")["raw_direction"].agg(lambda x: x.value_counts().index[0]).to_dict()
            df["direction"] = df["minute_bin"].map(direction_map)

            # 4. Avalanche Detection
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
            angle_df = df.iloc[peak_indices].copy()
            angle_df = angle_df[angle_df["ellipse_angle_deg"] <= 70]
            
            # 5. Aggregate
            ch_min = df.groupby(["minute_bin", "direction"]).agg(
                voltage_std=("CH2_volts", "std"), motor_speed=("motor_speed", "mean")
            ).reset_index()
            
            ang_min = angle_df.groupby(["minute_bin", "direction"]).agg(
                angle_mean=("ellipse_angle_deg", "mean")
            ).reset_index()
            
            minute_data = pd.merge(ch_min, ang_min, on=["minute_bin", "direction"], how="left")
            minute_data["material"] = material_name
            all_minute_data.append(minute_data)

    if not all_minute_data: return
    master_df = pd.concat(all_minute_data, ignore_index=True).dropna(subset=["angle_mean"])
    master_df["grouped_speed"] = master_df["motor_speed"].round(0).astype(int)
    
    materials = master_df["material"].unique()
    speeds = sorted(master_df["grouped_speed"].unique())

    generate_temporal_speed_grids(master_df, speeds, materials, metadata, output_dir)
    generate_hysteresis_plots(master_df, materials, metadata, output_dir, "angle_mean", "Angle (deg)", "Hysteresis_Angle")
    generate_hysteresis_plots(master_df, materials, metadata, output_dir, "voltage_std", "Charge Std Dev", "Hysteresis_Charge")
    generate_whole_run_summaries(master_df, materials, metadata, output_dir)
    
    print(f"Analysis Complete. Results in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    rel_path = sys.argv[1].replace('\\', '/')
    parts = rel_path.strip('/').split('/')
    if len(parts) < 3: sys.exit(1)
    metadata = {'vol': parts[0], 'cond': parts[1], 'dur': parts[2]}
    process_and_plot_single_run(os.path.join(BASE_DIR, rel_path), metadata)