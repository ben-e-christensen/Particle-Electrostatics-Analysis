import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Comparisons_Dual_Material_Per_RPM"
VOLTAGE_COL = "CH2_volts"
HARD_CUTOFF_MIN = 60.0  
SAMPLE_RATE = 100
MIN_SECONDS_PER_BIN = 30
FIXED_PROMINENCE = 1.5

# Formatting
FONT_TICK = 14
FONT_LABEL = 16
FONT_TITLE = 18

# Trial Markers
TRIAL_MARKERS = ['o', 's', '^', 'D', 'X', 'P', '*', 'v']
# =================================================

def load_and_aggregate_trials(target_dir):
    """
    Scans folder, loads CSVs, and AGGREGATES them into 1-minute bins.
    Returns list of dataframes (one per trial) with columns: 
    [grouped_speed, minute_bin, voltage_std, angle_mean]
    """
    aggregated_trials = []

    if not os.path.exists(target_dir):
        print(f"  [!] Folder not found: {target_dir}")
        return aggregated_trials

    trial_folders = []
    for root, dirs, files in os.walk(target_dir):
        if "experiment_log.csv" in files:
            trial_folders.append(root)
    
    trial_folders.sort() 
    
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    for i, trial_path in enumerate(trial_folders):
        try:
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            if df.empty: continue

            # 1. Clean & Filter
            df = df.dropna(subset=["motor_speed", "timestamp"])
            start_ms = df["ms"].min()
            df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
            df = df[df["rel_time_min"] <= HARD_CUTOFF_MIN]
            if df.empty: continue

            # 2. Minute Bins
            t0 = df["timestamp"].iloc[0]
            df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int) + 1
            
            # 3. Identify Peaks (for Angle Mean)
            peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
            angle_df = df.iloc[peak_indices].copy()

            # 4. Group & Aggregate (Match Master Script Logic)
            # We group by Speed too, to separate RPM steps
            df["grouped_speed"] = df["motor_speed"].round(0).astype(int)
            angle_df["grouped_speed"] = angle_df["motor_speed"].round(0).astype(int)

            charge_agg = df.groupby(["minute_bin", "grouped_speed"]).agg(
                voltage_std=("CH2_volts", "std"), 
                count=("CH2_volts", "count")
            ).reset_index()
            
            angle_agg = angle_df.groupby(["minute_bin", "grouped_speed"]).agg(
                angle_mean=("ellipse_angle_deg", "mean")
            ).reset_index()

            # Merge
            merged = pd.merge(charge_agg, angle_agg, on=["minute_bin", "grouped_speed"])
            
            # Filter low density bins
            merged = merged[merged["count"] > (SAMPLE_RATE * MIN_SECONDS_PER_BIN)]
            
            if not merged.empty:
                merged["trial_idx"] = i
                aggregated_trials.append(merged)

        except Exception as e:
            print(f"Skipping trial {os.path.basename(trial_path)}: {e}")
            
    return aggregated_trials

def get_global_limits(all_dfs):
    """Finds min/max of the AGGREGATED metrics (Std Dev & Mean Angle)."""
    if not all_dfs: return (0,1), (0,1)
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    x_min, x_max = master_df["voltage_std"].min(), master_df["voltage_std"].max()
    y_min, y_max = master_df["angle_mean"].min(), master_df["angle_mean"].max()
    
    x_buff = (x_max - x_min) * 0.1 if x_max != x_min else 0.01
    y_buff = (y_max - y_min) * 0.1 if y_max != y_min else 1
    
    return (x_min - x_buff, x_max + x_buff), (y_min - y_buff, y_max + y_buff)

def process_dual_comparison(parent_dir, target_vol, mat1, mat2):
    print(f"--- Starting Comparison (Binned): {mat1} vs {mat2} ({target_vol}) ---")
    
    scenarios = [
        ("Dirty", mat1), ("Clean", mat1),
        ("Dirty", mat2), ("Clean", mat2)
    ]
    
    data_store = {}
    all_dfs_for_limits = []

    # 1. Load & Aggregate Data
    for cond, mat in scenarios:
        path = os.path.join(parent_dir, cond, mat, target_vol)
        print(f"Loading {cond} {mat} from: {path}")
        trials = load_and_aggregate_trials(path)
        data_store[(cond, mat)] = trials
        all_dfs_for_limits.extend(trials)

    if not all_dfs_for_limits:
        print("No valid data found!")
        return

    # 2. Global Limits
    xlim, ylim = get_global_limits(all_dfs_for_limits)
    
    # 3. Find Unique Speeds
    master_df = pd.concat(all_dfs_for_limits, ignore_index=True)
    unique_speeds = sorted(master_df["grouped_speed"].unique())
    unique_speeds = [s for s in unique_speeds if s >= 1]

    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    layout_map = [
        (0, 0, "Dirty", mat1), (0, 1, "Clean", mat1),
        (1, 0, "Dirty", mat2), (1, 1, "Clean", mat2)
    ]

    # 4. Generate Plots
    for speed in unique_speeds:
        print(f"  Generating plot for {speed} RPM...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        fig.suptitle(f"{speed} RPM Comparison: {mat1} vs {mat2} (Volume {target_vol})", fontsize=22, fontweight='bold')
        
        global_sc = None

        for r, c, cond, mat in layout_map:
            ax = axes[r, c]
            trials = data_store[(cond, mat)]
            
            ax.set_title(f"{mat} ({cond})", fontsize=FONT_TITLE, fontweight='bold')
            
            has_data = False
            if trials:
                for df in trials:
                    # Filter for speed
                    speed_df = df[df["grouped_speed"] == speed]
                    if speed_df.empty: continue
                    has_data = True
                    
                    t_idx = df["trial_idx"].iloc[0]
                    marker = TRIAL_MARKERS[t_idx % len(TRIAL_MARKERS)]
                    
                    # PLOT: Voltage Std vs Angle Mean
                    sc = ax.scatter(
                        speed_df["voltage_std"], speed_df["angle_mean"],
                        c=speed_df["minute_bin"], cmap="coolwarm",
                        marker=marker, s=80, edgecolors='black', alpha=0.8,
                        vmin=0, vmax=60 
                    )
                    global_sc = sc
                    
                    # Trendline
                    if len(speed_df) > 2:
                        z = np.polyfit(speed_df["voltage_std"], speed_df["angle_mean"], 1)
                        p = np.poly1d(z)
                        xr = np.linspace(speed_df["voltage_std"].min(), speed_df["voltage_std"].max(), 10)
                        ax.plot(xr, p(xr), color='black', linestyle='--', alpha=0.5)

            if not has_data:
                ax.text(0.5, 0.5, "No Data", ha='center', fontsize=FONT_LABEL)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
            
            if r == 1: ax.set_xlabel("Voltage Std Dev (V)", fontsize=FONT_LABEL)
            if c == 0: ax.set_ylabel("Angle of Repose (deg)", fontsize=FONT_LABEL)
            else: ax.tick_params(labelleft=False)

        # Legends
        if global_sc:
            cbar = fig.colorbar(global_sc, ax=axes, shrink=0.7, location='right', pad=0.03)
            cbar.set_label("Elapsed Time (min)", fontsize=14)
        
        shape_handles = []
        for i in range(3):
            shape_handles.append(Line2D([0], [0], marker=TRIAL_MARKERS[i], color='w', 
                                        label=f"Trial {i+1}", markerfacecolor='grey', markersize=10, markeredgecolor='k'))
        
        fig.legend(handles=shape_handles, loc='upper right', bbox_to_anchor=(1.08, 0.95), title="Trials", fontsize=12)

        save_name = f"Compare_{mat1}_vs_{mat2}_{target_vol}_{speed}RPM.png"
        plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Done! Saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python compare_dual_materials_per_rpm_v2.py <Path_Folder> <Volume> <Mat1> <Mat2>")
    else:
        parent_dir = sys.argv[1]
        vol_arg = sys.argv[2]
        mat1_arg = sys.argv[3]
        mat2_arg = sys.argv[4]
        process_dual_comparison(parent_dir, vol_arg, mat1_arg, mat2_arg)