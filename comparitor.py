#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import re
from scipy.signal import find_peaks

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data"
SAMPLE_RATE = 100 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5

# ======================================================
# DATA LOADING & PROCESSING
# ======================================================

def get_processed_df(root_path):
    """Processes all materials in a specific condition folder."""
    all_minute_data = []
    if not os.path.exists(root_path):
        return pd.DataFrame()

    material_folders = [f.path for f in os.scandir(root_path) if f.is_dir() and "-graphs" not in f.name]

    for material_folder in material_folders:
        material_name = os.path.basename(material_folder).lower()
        for trial_folder in [f.path for f in os.scandir(material_folder) if f.is_dir()]:
            input_csv = os.path.join(trial_folder, "experiment_log.csv")
            if not os.path.isfile(input_csv): continue
            
            df = pd.read_csv(input_csv, names=["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
                                               "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
                                               "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
                                               "ch2_flag", "ch3_flag"], header=0, on_bad_lines="skip", engine="python")
            
            # Universal Direction Split (Mid-plateau)
            max_val = df["motor_speed"].max()
            max_indices = df.index[df["motor_speed"] == max_val].tolist()
            mid_plateau_idx = max_indices[len(max_indices) // 2]
            df["direction"] = "Increasing"
            df.loc[mid_plateau_idx + 1:, "direction"] = "Decreasing" 
            
            t0 = df["timestamp"].iloc[0]
            df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int) + 1
            
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
            angle_df = df.iloc[peak_indices].copy()
            
            charge_agg = df.melt(id_vars=["minute_bin", "direction", "motor_speed"], 
                                 value_vars=["CH2_volts", "CH3_volts"]).groupby(["minute_bin", "direction", "motor_speed"]).agg(
                voltage_std=("value", "std"), sample_count=("value", "count")).reset_index()
            
            angle_agg = angle_df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                angle_mean=("ellipse_angle_deg", "mean")).reset_index()

            charge_agg = charge_agg[charge_agg["sample_count"] >= (SAMPLE_RATE * MIN_SECONDS_PER_BIN)]
            minute_data = pd.merge(charge_agg, angle_agg, on=["minute_bin", "direction", "motor_speed"], how="left")
            minute_data["material"] = material_name
            all_minute_data.append(minute_data)

    if not all_minute_data: return pd.DataFrame()
    master = pd.concat(all_minute_data, ignore_index=True).dropna(subset=["angle_mean"])
    master["grouped_speed"] = master["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
    return master[master["grouped_speed"] >= 1]

# ======================================================
# HELPER FOR AXIS LIMITS
# ======================================================

def get_padded_limits(series1, series2, padding=0.05):
    """Calculates min/max across two series with padding."""
    # Combine series and drop NaNs
    combined = pd.concat([series1, series2]).dropna()
    
    if combined.empty:
        return None, None
        
    val_min = combined.min()
    val_max = combined.max()
    
    if val_min == val_max:
        # Handle case where all values are the same (avoid singular matrix or zero range)
        margin = abs(val_min) * 0.1 if val_min != 0 else 1.0
        return val_min - margin, val_max + margin
        
    span = val_max - val_min
    return val_min - (span * padding), val_max + (span * padding)

# ======================================================
# DYNAMIC COMPARISON PLOTTING
# ======================================================

def generate_comparison_hysteresis(df_dirty, df_clean, materials, output_dir, y_col, y_label, filename):
    """Generates a dynamic Nx2 grid for side-by-side Hysteresis comparison with fixed axes per material."""
    plot_subdir = os.path.join(output_dir, "Comparison_Hysteresis")
    os.makedirs(plot_subdir, exist_ok=True)
    
    num_mats = len(materials)
    fig, axs = plt.subplots(num_mats, 2, figsize=(16, 5 * num_mats), squeeze=False, constrained_layout=True)
    fig.suptitle(f"{y_label} Comparison: Dirty vs Clean", fontsize=20, fontweight='bold', y=1.02)
    
    for i, mat in enumerate(materials):
        # 1. Isolate data for this material
        mat_dirty = df_dirty[df_dirty["material"] == mat]
        mat_clean = df_clean[df_clean["material"] == mat]

        # 2. Calculate shared limits for this row (Material)
        # Hysteresis X is Motor Speed (grouped_speed), Y is the variable passed (y_col)
        
        # We aggregate first to get the actual plotted points (mean + std) to ensure limits cover error bars
        # However, for simplicity and robustness, we calculate limits on the raw data points used for the plot
        # Or better: calculate on the aggregated means + std dev to ensure error bars fit.
        
        # Quick aggregation to find ranges including error bars
        d_stats = mat_dirty.groupby(['grouped_speed', 'direction'])[y_col].agg(['mean', 'std']).reset_index() if not mat_dirty.empty else pd.DataFrame()
        c_stats = mat_clean.groupby(['grouped_speed', 'direction'])[y_col].agg(['mean', 'std']).reset_index() if not mat_clean.empty else pd.DataFrame()
        
        # Calculate Y limits (Mean + Std Dev to ensure full visibility)
        y_vals = []
        if not d_stats.empty: 
            y_vals.extend((d_stats['mean'] + d_stats['std'].fillna(0)).tolist())
            y_vals.extend((d_stats['mean'] - d_stats['std'].fillna(0)).tolist())
        if not c_stats.empty:
            y_vals.extend((c_stats['mean'] + c_stats['std'].fillna(0)).tolist())
            y_vals.extend((c_stats['mean'] - c_stats['std'].fillna(0)).tolist())
            
        y_series = pd.Series(y_vals)
        y_min, y_max = get_padded_limits(y_series, pd.Series([])) # Padding helper handles single series too

        # Calculate X limits (Motor Speed)
        x_min, x_max = get_padded_limits(mat_dirty["grouped_speed"], mat_clean["grouped_speed"])

        for col, (df_type, label) in enumerate([(mat_dirty, "Dirty"), (mat_clean, "Clean")]):
            ax = axs[i, col]
            
            if not df_type.empty:
                h_stats = df_type.groupby(['grouped_speed', 'direction']).agg(y_avg=(y_col, 'mean'), y_std=(y_col, 'std')).unstack()
                for direction, color in [('Increasing', '#1f77b4'), ('Decreasing', '#d62728')]:
                    if direction in h_stats['y_avg'].columns:
                        data = h_stats.xs(direction, axis=1, level=1).dropna()
                        ax.errorbar(data.index, data['y_avg'], yerr=data['y_std'], fmt='o-', 
                                    color=color, label=direction, capsize=5, lw=2, markersize=8)
            
            # Apply the shared limits
            if y_min is not None: ax.set_ylim(y_min, y_max)
            if x_min is not None: ax.set_xlim(x_min, x_max)

            ax.set_title(f"{mat.capitalize()} ({label})", fontweight='bold', fontsize=14)
            ax.set_xlabel("Motor Speed (RPM)")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.savefig(os.path.join(plot_subdir, f"{filename}.png"), bbox_inches='tight')
    plt.close()

def generate_comparison_grids(df_dirty, df_clean, materials, speeds, output_dir):
    """Generates a dynamic Nx2 grid for side-by-side Minute-by-Minute analysis with fixed axes per material."""
    plot_subdir = os.path.join(output_dir, "Comparison_Grids_By_Speed")
    os.makedirs(plot_subdir, exist_ok=True)

    for speed in speeds:
        num_mats = len(materials)
        fig, axs = plt.subplots(num_mats, 2, figsize=(16, 5 * num_mats), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Speed: {speed} RPM | Dirty vs Clean Comparison", fontsize=20, fontweight='bold', y=1.02)
        
        for i, mat in enumerate(materials):
            # 1. Filter data for this Material AND Speed
            mat_speed_dirty = df_dirty[(df_dirty["material"] == mat) & (df_dirty["grouped_speed"] == speed)]
            mat_speed_clean = df_clean[(df_clean["material"] == mat) & (df_clean["grouped_speed"] == speed)]

            # 2. Find shared Colorbar scale (Minute Bins)
            t_max = max(
                mat_speed_dirty["minute_bin"].max() if not mat_speed_dirty.empty else 1, 
                mat_speed_clean["minute_bin"].max() if not mat_speed_clean.empty else 1
            )

            # 3. Find shared X/Y limits for scatter plot
            # X = Voltage Std, Y = Angle Mean
            x_min, x_max = get_padded_limits(mat_speed_dirty["voltage_std"], mat_speed_clean["voltage_std"])
            y_min, y_max = get_padded_limits(mat_speed_dirty["angle_mean"], mat_speed_clean["angle_mean"])

            for col, (df_subset, label) in enumerate([(mat_speed_dirty, "Dirty"), (mat_speed_clean, "Clean")]):
                ax = axs[i, col]
                
                if not df_subset.empty:
                    sc = ax.scatter(df_subset["voltage_std"], df_subset["angle_mean"], 
                                    c=df_subset["minute_bin"], cmap="coolwarm", vmin=1, vmax=t_max,
                                    alpha=0.8, s=80, edgecolors='black')
                    
                    if len(df_subset) >= 2:
                        m, b = np.polyfit(df_subset["voltage_std"], df_subset["angle_mean"], 1)
                        # Plot trendline across the full visible X range for aesthetics
                        x_vals = np.array([x_min, x_max]) if x_min is not None else df_subset["voltage_std"]
                        ax.plot(x_vals, m*x_vals + b, "--", color="black", alpha=0.4)
                    
                    if col == 1: 
                        plt.colorbar(sc, ax=ax, label="Minute Bin")
                
                # Apply the shared limits
                if x_min is not None: ax.set_xlim(x_min, x_max)
                if y_min is not None: ax.set_ylim(y_min, y_max)

                ax.set_title(f"{mat.capitalize()} ({label})", fontweight='bold', fontsize=14)
                ax.set_xlabel("Std Dev Voltage (V)")
                ax.set_ylabel("Angle of Repose (deg)")
                ax.grid(True, alpha=0.3)

        plt.savefig(os.path.join(plot_subdir, f"Comparison_{speed}RPM.png"), bbox_inches='tight')
        plt.close()

# ======================================================
# EXECUTION
# ======================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python comparison_script.py <Volume> <Duration>")
        sys.exit(1)
    
    vol_arg, dur_arg = sys.argv[1], sys.argv[2]
    
    dirty_path = os.path.join(BASE_DIR, vol_arg, "Dirty", dur_arg).replace('\\', '/')
    clean_path = os.path.join(BASE_DIR, vol_arg, "Clean", dur_arg).replace('\\', '/')
    
    df_dirty = get_processed_df(dirty_path)
    df_clean = get_processed_df(clean_path)
    
    common_mats = sorted(list(set(df_dirty["material"].unique()) & set(df_clean["material"].unique())))
    
    if not common_mats:
        print(f"No overlapping materials found between {dirty_path} and {clean_path}")
        sys.exit(1)
    
    output_dir = os.path.join(BASE_DIR, vol_arg, f"Comparison_{dur_arg}_Analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    all_speeds = sorted(list(set(df_dirty["grouped_speed"].unique()) | set(df_clean["grouped_speed"].unique())))

    # Generate Side-by-Side Plots
    generate_comparison_hysteresis(df_dirty, df_clean, common_mats, output_dir, "angle_mean", "Angle of Repose (deg)", "Comp_Hysteresis_Angle")
    generate_comparison_hysteresis(df_dirty, df_clean, common_mats, output_dir, "voltage_std", "Std Dev Voltage (V)", "Comp_Hysteresis_Charge")
    generate_comparison_grids(df_dirty, df_clean, common_mats, all_speeds, output_dir)
    
    print(f"Dynamic Comparison Complete. Files saved to: {output_dir}")