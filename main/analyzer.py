import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.signal import medfilt, find_peaks
from scipy.stats import binned_statistic_2d

# === CONFIG ===
SAMPLE_RATE = 100  
BASELINE_WINDOW_SEC = 4 
TOP_PERCENT_CHARGE = 20  

def avg_top_percent(series):
    abs_series = series.abs()
    if len(abs_series) == 0: return 0
    q = 1.0 - (TOP_PERCENT_CHARGE / 100.0)
    cutoff = abs_series.quantile(q)
    return abs_series[abs_series >= cutoff].mean() if len(abs_series[abs_series >= cutoff]) > 0 else 0

def run_analysis(target_dir):
    """
    Runs analysis on a specific Duration folder (e.g. '12mins').
    Strictly looks for subfolders ending in '-T#' to avoid processing output folders.
    """
    if not os.path.isdir(target_dir): return

    # Setup Output
    output_dir = os.path.join(target_dir, "aggregated_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Naming
    run_name = os.path.basename(os.path.normpath(target_dir)) # e.g. "12mins"
    rpm_name = os.path.basename(os.path.dirname(os.path.normpath(target_dir))) # e.g. "1000"
    mat_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(target_dir)))) # e.g. "Nylon-Dirty"
    full_title = f"{mat_name} {rpm_name} {run_name}"

    # Find Subfolders (Strict Filter for -T#)
    # This prevents reading "aggregated_results" as a data folder
    subfolders = [f.path for f in os.scandir(target_dir) if f.is_dir() and re.search(r'-T\d+$', f.name)]
    
    # Storage
    all_summaries = []      
    all_minute_data = []    

    # --- 1. PROCESS TRIALS ---
    for trial_folder in subfolders:
        input_csv = os.path.join(trial_folder, "experiment_log.csv")
        if not os.path.isfile(input_csv): continue

        cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
                "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
                "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]
        
        try:
            df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
        except:
            continue
        
        # Angle Processing
        peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=3.5)
        result_df = df.iloc[peak_indices].copy()
        result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

        if len(result_df) > 0:
            result_df["speed_change"] = result_df["motor_speed"].ne(result_df["motor_speed"].shift())
            result_df["run_id"] = result_df["speed_change"].cumsum()

            # (Insert your legacy 26 RPM fix here if needed)

            trial_summary = (
                result_df.groupby("run_id")
                .agg(motor_speed=("motor_speed", "first"), angle_mean=("ellipse_angle_deg", "mean"), angle_std=("ellipse_angle_deg", "std"))
                .reset_index(drop=True)
            )
            trial_summary["direction"] = "Backward"
            half_pt = len(trial_summary) // 2
            trial_summary.iloc[:half_pt, trial_summary.columns.get_loc("direction")] = "Forward"
            all_summaries.append(trial_summary)

        # Charge Processing
        kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE) | 1
        df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
        df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
        df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
        df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

        t0 = df["timestamp"].iloc[0]
        df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)
        
        # Melt Channels
        melted_df = df.melt(id_vars=["minute_bin", "motor_speed"], value_vars=["CH2_clean", "CH3_clean"], value_name="combined_clean")

        charge_per_minute = melted_df.groupby("minute_bin").agg(
            combined_std=("combined_clean", "std"),
            combined_top_pct=("combined_clean", avg_top_percent),
            motor_speed=("motor_speed", "mean")
        )
        
        angle_minutes = result_df.copy()
        angle_minutes["minute_bin"] = ((angle_minutes["timestamp"] - t0) / 60).astype(int)
        angle_per_minute = angle_minutes.groupby("minute_bin").agg(angle_mean=("ellipse_angle_deg", "mean"))
        
        minute_data = pd.merge(charge_per_minute, angle_per_minute, on="minute_bin", how="inner")
        minute_data["trial"] = os.path.basename(trial_folder)
        all_minute_data.append(minute_data)

    if not all_summaries: return

    # --- 2. GENERATE PLOTS ---
    master_angle_df = pd.concat(all_summaries, ignore_index=True)
    master_minute_df = pd.concat(all_minute_data, ignore_index=True)

    # Export Stats
    global_stats = master_angle_df.groupby(["motor_speed", "direction"])["angle_mean"].agg(["mean", "std"]).reset_index()
    charge_agg = master_minute_df.groupby("motor_speed").agg(charge_std=("combined_std", "mean"), charge_mag=("combined_top_pct", "mean")).reset_index()
    export_df = pd.merge(global_stats, charge_agg, on="motor_speed", how="left")
    export_df.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)

    # Plot 1: Angle Sweep
    plt.figure(figsize=(10, 6))
    fwd = global_stats[global_stats["direction"] == "Forward"].sort_values("motor_speed")
    bwd = global_stats[global_stats["direction"] == "Backward"].sort_values("motor_speed", ascending=False)
    plt.errorbar(fwd["motor_speed"], fwd["mean"], yerr=fwd["std"], fmt="-o", label="Forward")
    plt.errorbar(bwd["motor_speed"], bwd["mean"], yerr=bwd["std"], fmt="--s", label="Backward")
    plt.title(f"{full_title} — Hysteresis")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Graph_1_Angle_Sweep.png"))
    plt.close()

    # Plots 2 & 3: Contour Maps
    def plot_contour(z_col, label, fname):
        if len(master_minute_df) < 5: return
        plt.figure(figsize=(10, 8))
        plt.tricontourf(master_minute_df["motor_speed"], master_minute_df["angle_mean"], master_minute_df[z_col], levels=30, cmap="inferno")
        plt.colorbar(label=label)
        plt.title(f"{full_title}\n{fname}")
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    plot_contour("combined_std", "Std Dev (V)", "Graph_2_Contour_Noise.png")
    plot_contour("combined_top_pct", "Top 20% (V)", "Graph_3_Contour_Magnitude.png")

    # Plots 4 & 5: Heatmaps
    def plot_heatmap(z_col, label, fname):
        if len(master_minute_df) < 5: return
        x, y, z = master_minute_df["motor_speed"], master_minute_df["angle_mean"], master_minute_df[z_col]
        # Safety check for dimensions
        if x.nunique() < 2 or y.nunique() < 2: return
        
        try:
            ret = binned_statistic_2d(x, y, z, statistic='mean', bins=[12, 12])
            plt.figure(figsize=(10, 8))
            plt.imshow(ret.statistic.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto', cmap="inferno")
            plt.colorbar(label=label)
            plt.title(f"{full_title}\n{fname}")
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()
        except:
            pass

    plot_heatmap("combined_std", "Mean Std Dev", "Graph_4_Heatmap_Noise.png")
    plot_heatmap("combined_top_pct", "Mean Top 20%", "Graph_5_Heatmap_Magnitude.png")
    
# Allow standalone execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: run_analysis(sys.argv[1])