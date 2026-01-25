#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import find_peaks

# ======================================================
# CONFIGURATION
# ======================================================
DEFAULT_VOL = "500"
BASE_DIR_TEMPLATE = "/media/ben/SANDISK/particle-data/{}/time_lapse"
TIME_POINTS = ["t0", "day-1", "day-7"]

SAMPLE_RATE = 100 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5

# ======================================================
# DATA LOADING (Depth Agnostic)
# ======================================================

def get_material_name(csv_path, time_point_root):
    """
    Determines material name regardless of folder depth.
    """
    trial_dir = os.path.dirname(csv_path)
    parent_dir = os.path.dirname(trial_dir)
    parent_name = os.path.basename(parent_dir)
    
    if parent_name == os.path.basename(time_point_root) or parent_name in TIME_POINTS:
        trial_name = os.path.basename(trial_dir)
        candidate = trial_name.split("-")[0].split("_")[0]
        return candidate.lower()
    
    return parent_name.lower()

def load_hardcoded_data(root_path):
    all_data = []
    
    if not os.path.exists(root_path):
        print(f"CRITICAL ERROR: Root path does not exist: {root_path}")
        sys.exit(1)

    print(f"\n--- SCANNING ROOT: {root_path} ---")

    for tp in TIME_POINTS:
        tp_path = os.path.join(root_path, tp)
        
        # 1. Locate Time Point Folder
        if not os.path.exists(tp_path):
            found = False
            for actual_name in os.listdir(root_path):
                if actual_name.lower() == tp.lower():
                    tp_path = os.path.join(root_path, actual_name)
                    found = True
                    break
            if not found:
                print(f"  > WARNING: Could not find folder for '{tp}'. Skipping.")
                continue
        
        print(f"  > Scanning: {tp}...")

        # 2. Walk recursively
        csv_files = []
        for root, dirs, files in os.walk(tp_path):
            if "experiment_log.csv" in files:
                csv_files.append(os.path.join(root, "experiment_log.csv"))

        if not csv_files:
            print(f"    [!] No CSVs found in {tp}")
            continue

        for input_csv in csv_files:
            try:
                # 3. Smart Material Detection
                material_name = get_material_name(input_csv, tp_path)
                
                if "images" in input_csv.split(os.sep): continue

                df = pd.read_csv(input_csv, names=["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
                                                   "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
                                                   "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
                                                   "ch2_flag", "ch3_flag"], header=0, on_bad_lines="skip", engine="python")
                
                if df.empty: continue

                max_val = df["motor_speed"].max()
                max_indices = df.index[df["motor_speed"] == max_val].tolist()
                mid_plateau_idx = max_indices[len(max_indices) // 2]
                df["direction"] = "Increasing"
                df.loc[mid_plateau_idx + 1:, "direction"] = "Decreasing" 
                
                t0_ts = df["timestamp"].iloc[0]
                df["minute_bin"] = ((df["timestamp"] - t0_ts) / 60).astype(int) + 1
                
                peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
                if len(peak_indices) == 0: continue
                
                angle_df = df.iloc[peak_indices].copy()
                
                charge_agg = df.melt(id_vars=["minute_bin", "direction", "motor_speed"], 
                                     value_vars=["CH2_volts", "CH3_volts"]).groupby(["minute_bin", "direction", "motor_speed"]).agg(
                    voltage_std=("value", "std"), sample_count=("value", "count")).reset_index()
                
                angle_agg = angle_df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                    angle_mean=("ellipse_angle_deg", "mean")).reset_index()

                valid_charge_agg = charge_agg[charge_agg["sample_count"] >= (SAMPLE_RATE * MIN_SECONDS_PER_BIN)]
                if valid_charge_agg.empty: continue

                minute_data = pd.merge(valid_charge_agg, angle_agg, on=["minute_bin", "direction", "motor_speed"], how="left")
                minute_data["material"] = material_name
                minute_data["time_point"] = tp 
                
                all_data.append(minute_data)
                print(f"    [OK] Found Material: '{material_name}' | Trial: {os.path.basename(os.path.dirname(input_csv))}")

            except Exception as e:
                print(f"    [ERROR] processing {input_csv}: {e}")

    if not all_data: 
        return pd.DataFrame()
        
    master = pd.concat(all_data, ignore_index=True).dropna(subset=["angle_mean"])
    master["grouped_speed"] = master["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
    return master[master["grouped_speed"] >= 1]

# ======================================================
# PLOTTING
# ======================================================

def generate_side_by_side_hysteresis(df, output_dir):
    save_dir = os.path.join(output_dir, "Hysteresis_Comparisons")
    os.makedirs(save_dir, exist_ok=True)
    
    materials = sorted(df["material"].unique())
    metrics = [("angle_mean", "Angle of Repose (deg)"), ("voltage_std", "Std Dev Voltage (V)")]

    for mat in materials:
        for y_col, y_label in metrics:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True) # sharey is already True
            fig.suptitle(f"{mat.capitalize()} - {y_label}", fontsize=20, fontweight='bold', y=1.05)
            
            mat_subset = df[df["material"] == mat]
            if mat_subset.empty: continue
            
            # --- CALCULATE GLOBAL LIMITS FOR THIS MATERIAL (Across all time points) ---
            # Y-Axis Limits
            y_min, y_max = mat_subset[y_col].min(), mat_subset[y_col].max()
            y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
            
            # X-Axis Limits (NEW)
            x_min, x_max = mat_subset["grouped_speed"].min(), mat_subset["grouped_speed"].max()
            x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 5
            
            for i, tp in enumerate(TIME_POINTS):
                ax = axs[i]
                subset = mat_subset[mat_subset["time_point"] == tp]
                
                if not subset.empty:
                    h_stats = subset.groupby(['grouped_speed', 'direction']).agg(
                        y_avg=(y_col, 'mean'), 
                        y_std=(y_col, 'std')
                    ).unstack()
                    
                    for direction, color in [('Increasing', '#1f77b4'), ('Decreasing', '#d62728')]:
                        if direction in h_stats['y_avg'].columns:
                            data = h_stats.xs(direction, axis=1, level=1).dropna()
                            ax.errorbar(data.index, data['y_avg'], yerr=data['y_std'], fmt='o-', 
                                        color=color, label=direction, capsize=3, lw=2)
                
                ax.set_title(tp, fontweight='bold', fontsize=16, color="#333333")
                ax.set_xlabel("Speed (RPM)")
                if i == 0: ax.set_ylabel(y_label)
                
                # --- APPLY SHARED LIMITS ---
                if not pd.isna(y_min): ax.set_ylim(y_min - y_margin, y_max + y_margin)
                if not pd.isna(x_min): ax.set_xlim(x_min - x_margin, x_max + x_margin)
                
                ax.grid(True, alpha=0.3)
                if i == 0: ax.legend()

            safe_mat = mat.replace(" ", "_").replace("/", "-")
            safe_metric = y_col.split("_")[0]
            plt.savefig(os.path.join(save_dir, f"{safe_mat}_{safe_metric}_SideBySide.png"), bbox_inches='tight')
            plt.close()

def generate_side_by_side_scatter(df, output_dir):
    save_dir = os.path.join(output_dir, "Scatter_Comparisons")
    os.makedirs(save_dir, exist_ok=True)
    
    materials = sorted(df["material"].unique())
    speeds = sorted(df["grouped_speed"].unique())

    for mat in materials:
        for speed in speeds:
            mat_speed_df = df[(df["material"] == mat) & (df["grouped_speed"] == speed)]
            if mat_speed_df.empty: continue

            fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True, sharex=True)
            fig.suptitle(f"{mat.capitalize()} @ {speed} RPM - Charge vs Angle", fontsize=20, fontweight='bold', y=1.05)
            
            t_max = mat_speed_df["minute_bin"].max()
            
            # Global Limits for this Material + Speed combo
            x_min, x_max = mat_speed_df["voltage_std"].min(), mat_speed_df["voltage_std"].max()
            y_min, y_max = mat_speed_df["angle_mean"].min(), mat_speed_df["angle_mean"].max()
            
            x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 0.1
            y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1

            for i, tp in enumerate(TIME_POINTS):
                ax = axs[i]
                subset = mat_speed_df[mat_speed_df["time_point"] == tp]
                
                if not subset.empty:
                    sc = ax.scatter(subset["voltage_std"], subset["angle_mean"], 
                                    c=subset["minute_bin"], cmap="coolwarm", vmin=1, vmax=t_max,
                                    alpha=0.8, s=80, edgecolors='black')
                    
                    if len(subset) >= 2:
                        try:
                            m, b = np.polyfit(subset["voltage_std"], subset["angle_mean"], 1)
                            ax.plot(subset["voltage_std"], m*subset["voltage_std"] + b, "--", color="black", alpha=0.4)
                        except: pass
                    
                    if i == 2: 
                        plt.colorbar(sc, ax=ax, label="Minute Bin")

                ax.set_title(tp, fontweight='bold', fontsize=16, color="#333333")
                ax.set_xlabel("Std Dev Voltage (V)")
                if i == 0: ax.set_ylabel("Angle of Repose (deg)")
                
                # Apply Shared Limits
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                ax.grid(True, alpha=0.3)

            safe_mat = mat.replace(" ", "_").replace("/", "-")
            plt.savefig(os.path.join(save_dir, f"{safe_mat}_{speed}RPM_Scatter_SideBySide.png"), bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    vol_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VOL
    root_data_path = BASE_DIR_TEMPLATE.format(vol_arg)
    output_dir = os.path.join(os.path.dirname(root_data_path), f"Time_Lapse_SideBySide_{vol_arg}")
    
    print(f"--- 1x3 Side-by-Side (Consistent Axis Scales) ---")
    df = load_hardcoded_data(root_data_path)
    
    if df.empty:
        print("No valid data found.")
        sys.exit(1)
        
    print(f"\nData loaded successfully. Found materials: {df['material'].unique()}")
    generate_side_by_side_hysteresis(df, output_dir)
    generate_side_by_side_scatter(df, output_dir)
    print(f"Graphs saved to: {output_dir}")