#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys, os
import re
from scipy.signal import find_peaks

# === CONFIG ===
BASE_DIR = "F:/particle-data/"
SAMPLE_RATE = 100 
AXIS_BUFFER_PCT = 0.20 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5

# UPDATED PHYSICS DATABASE
PHYSICS_DB = {
    "acetal":   {"density": 1.41, "tribo_rank": -143.33, "resistivity": 15},
    "acrylic":  {"density": 1.18, "tribo_rank": -48.73,  "resistivity": 14},
    "nylon":    {"density": 1.14, "tribo_rank": -18.35,  "resistivity": 12},
    "teflon":   {"density": 2.20, "tribo_rank": -113.06, "resistivity": 18}
}

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
            x, y = mat_speed_df["voltage_std"], mat_speed_df["angle_mean"]
            sc = ax.scatter(x, y, c=mat_speed_df["minute_bin"], cmap="coolwarm", vmin=1, vmax=time_max, alpha=0.8, s=60, edgecolors='black')
            if len(mat_speed_df) >= 2:
                m, b = np.polyfit(x, y, 1); ax.plot(x, m*x + b, "--", color="black", alpha=0.4)
            ax.set_title(f"{material.capitalize()} (n={len(mat_speed_df)})")
            ax.set_xlabel("Std Dev Voltage (V)", fontweight='bold', labelpad=10)
            ax.set_ylabel("Angle of Repose (deg)", fontweight='bold', labelpad=10)
            ax.grid(True, alpha=0.3)
        if sc: fig.colorbar(sc, ax=axs, label="Time (min)", shrink=0.8)
        plt.savefig(os.path.join(plot_subdir, f"Grid_{speed}RPM.png")); plt.close()

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
        plt.xlabel("Std Dev Voltage (V)", fontweight='bold', labelpad=15)
        plt.ylabel("Angle of Repose (deg)", fontweight='bold', labelpad=15)
        plt.legend(title="RPM", bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True, alpha=0.3); plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(plot_subdir, f"Summary_{material}.png")); plt.close()

def generate_hysteresis(df, materials, output_dir, y_col, y_label, filename_prefix):
    plot_subdir = os.path.join(output_dir, "Plots_Hysteresis")
    os.makedirs(plot_subdir, exist_ok=True)
    for material in materials:
        mat_df = df[df["material"] == material.lower()]
        if mat_df.empty: continue
        h_stats = mat_df.groupby(['grouped_speed', 'direction']).agg(y_avg=(y_col, 'mean'), y_std=(y_col, 'std')).unstack()
        plt.figure(figsize=(10, 7))
        for direction, color in [('Increasing', 'blue'), ('Decreasing', 'red')]:
            if direction in h_stats['y_avg'].columns:
                data = h_stats.xs(direction, axis=1, level=1).dropna()
                plt.errorbar(data.index, data['y_avg'], yerr=data['y_std'], fmt='o-', color=color, label=direction, capsize=5, lw=2)
        final_y_label = "Std Dev Voltage (V)" if "Charge" in y_label or "voltage" in y_col else "Angle of Repose (deg)"
        plt.title(f"{final_y_label} Hysteresis: {material.capitalize()}", pad=15)
        plt.xlabel("Motor Speed (RPM)", fontweight='bold', labelpad=12)
        plt.ylabel(final_y_label, fontweight='bold', labelpad=12); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plot_subdir, f"{filename_prefix}_{material}.png")); plt.close()

def generate_physics_contours(df, speeds, output_dir):
    base_contour_dir = os.path.join(output_dir, "Plots_Contours")
    os.makedirs(base_contour_dir, exist_ok=True)
    # RAW STRING r"" added below to fix SyntaxWarning
    properties = [
        ("density", "Density (g/cm3)"), 
        ("tribo_rank", r"Tribo Charge Density ($\mu C \cdot m^{-2}$)"), 
        ("resistivity", r"Log Resistivity ($\Omega \cdot cm$)"), 
        ("collapse_count", "Collapses / Min")
    ]
    for prop_col, prop_label in properties:
        prop_dir = os.path.join(base_contour_dir, prop_col.capitalize()); os.makedirs(prop_dir, exist_ok=True)
        for speed in speeds:
            speed_df = df[df['grouped_speed'] == speed].dropna(subset=[prop_col, 'voltage_std', 'angle_mean'])
            if speed_df['material'].nunique() < 3: continue
            plt.figure(figsize=(10, 7))
            x, y, z = speed_df[prop_col], speed_df['voltage_std'], speed_df['angle_mean']
            try:
                cntr = plt.tricontourf(x, y, z, levels=20, cmap="viridis", alpha=0.9)
                plt.colorbar(cntr).set_label("Angle of Repose (deg)", fontweight='bold')
                plt.scatter(x, y, c=z, cmap="viridis", edgecolors='white', s=110, zorder=5)
            except: plt.close(); continue
            plt.title(f"{prop_label} vs Charge Intensity | {speed} RPM", pad=15)
            plt.xlabel(prop_label, fontweight='bold', labelpad=12)
            plt.ylabel("Std Dev Voltage (V)", fontweight='bold', labelpad=12); plt.grid(True, linestyle=':', alpha=0.3)
            plt.savefig(os.path.join(prop_dir, f"RPM_{speed}.png")); plt.close()

def generate_4d_ridges(df, output_dir):
    ridge_dir = os.path.join(output_dir, "Plots_4D_Ridge_Analysis")
    os.makedirs(ridge_dir, exist_ok=True)
    # RAW STRING r"" added below
    properties = [
        ("density", "Density (g/cm3)"), 
        ("tribo_rank", r"Tribo Charge Density ($\mu C \cdot m^{-2}$)"), 
        ("resistivity", r"Log Resistivity ($\Omega \cdot cm$)"), 
        ("collapse_count", "Collapses")
    ]
    for prop_col, prop_label in properties:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        x_data, y_data, z_data, angles = df[prop_col], df['grouped_speed'], df['voltage_std'], df['angle_mean']
        ridge_df = df.sort_values('angle_mean', ascending=False).drop_duplicates('grouped_speed').sort_values('grouped_speed')
        rx, ry, rz = ridge_df[prop_col].values, ridge_df['grouped_speed'].values, ridge_df['voltage_std'].values
        try:
            z_coef = np.polyfit(ry, rz, 1); x_coef = np.polyfit(ry, rx, 1); y_line = np.linspace(ry.min(), ry.max(), 100)
            ax.plot(np.polyval(x_coef, y_line), y_line, np.polyval(z_coef, y_line), color='red', linewidth=3, label='Ridge Line', zorder=100)
            eq_text = f"Ridge Eqs:\nCharge = {z_coef[0]:.3f}*RPM + {z_coef[1]:.2f}\n{prop_col} = {x_coef[0]:.3f}*RPM + {x_coef[1]:.2f}"
            ax.text2D(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=11, color='red', bbox=dict(facecolor='white', alpha=0.8))
        except: pass
        img = ax.scatter(x_data, y_data, z_data, c=angles, cmap='viridis', s=80, alpha=0.5, edgecolors='w')
        ax.scatter(rx, ry, rz, color='red', s=150, edgecolors='black', label='Peaks', zorder=101)
        ax.set_xlabel(prop_label, fontweight='bold', labelpad=10)
        ax.set_ylabel('Motor Speed (RPM)', fontweight='bold', labelpad=10)
        ax.set_zlabel('Std Dev Voltage (V)', fontweight='bold', labelpad=10)
        plt.colorbar(img, ax=ax, shrink=0.6).set_label("Angle of Repose (deg)", fontweight='bold')
        plt.title(f"4D Ridge Analysis: {prop_label}", fontsize=15, pad=20)
        plt.savefig(os.path.join(ridge_dir, f"Ridge_{prop_col}.png")); plt.close()

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
            max_idx = df["motor_speed"].idxmax(); df["direction"] = "Increasing"; df.loc[max_idx:, "direction"] = "Decreasing"
            t0 = df["timestamp"].iloc[0]; df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int) + 1
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
            angle_df = df.iloc[peak_indices].copy(); angle_df["minute_bin"] = ((angle_df["timestamp"] - t0) / 60).astype(int) + 1
            charge_agg = df.melt(id_vars=["minute_bin", "direction", "motor_speed"], value_vars=["CH2_volts", "CH3_volts"]).groupby(["minute_bin", "direction"]).agg(
                voltage_std=("value", "std"), motor_speed=("motor_speed", "mean"), sample_count=("value", "count")).reset_index()
            angle_agg = angle_df.groupby(["minute_bin", "direction"]).agg(
                angle_mean=("ellipse_angle_deg", "mean"), collapse_count=("ellipse_angle_deg", "count")).reset_index()
            charge_agg = charge_agg[charge_agg["sample_count"] >= (SAMPLE_RATE * MIN_SECONDS_PER_BIN * 2)]
            minute_data = pd.merge(charge_agg, angle_agg, on=["minute_bin", "direction"], how="left")
            minute_data["collapse_count"] = minute_data["collapse_count"].fillna(0); minute_data["material"] = material_name
            for prop in ["density", "tribo_rank", "resistivity"]:
                minute_data[prop] = PHYSICS_DB.get(material_name, {}).get(prop, np.nan)
            all_minute_data.append(minute_data)

    if not all_minute_data: return
    master_df = pd.concat(all_minute_data, ignore_index=True).dropna(subset=["angle_mean"])
    master_df["grouped_speed"] = master_df["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
    master_df = master_df[master_df["grouped_speed"] >= 1]
    
    master_df.to_csv(os.path.join(output_dir, "master_comparison_data.csv"), index=False)
    mats, speeds = master_df["material"].unique(), sorted(master_df["grouped_speed"].unique())
    
    # RESTORED GRID PLOTS
    generate_temporal_grids(master_df, speeds, mats, metadata, output_dir)
    generate_whole_run_summaries(master_df, mats, output_dir)
    generate_hysteresis(master_df, mats, output_dir, "angle_mean", "Angle", "Hyst_Angle")
    generate_hysteresis(master_df, mats, output_dir, "voltage_std", "Charge", "Hyst_Charge")
    generate_physics_contours(master_df, speeds, output_dir)
    generate_4d_ridges(master_df, output_dir)
    print(f"Unified Analysis Complete. Graphs: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    run_path = sys.argv[1].replace('\\', '/').strip('/')
    parts = run_path.split('/')
    meta = {'vol': parts[0], 'cond': parts[1], 'dur': parts[2], 'time_max_min': parse_duration_minutes(parts[2])}
    process_run(run_path, meta)