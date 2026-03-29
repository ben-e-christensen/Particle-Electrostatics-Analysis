import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import rcParams
import collections

# ================= CONFIGURATION & PATHS =================
SHOW_LEGEND = True
SHOW_TITLES = True

# --- SIGNAL PROCESSING ---
VOLTAGE_COL = "CH2_volts"
FLIP_CH2 = True  

# --- ACADEMIC PLOT SETTINGS ---
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
FONT_TICK = 18
FONT_LABEL = 18
FONT_TITLE = 20

# Expected target speeds to snap the raw motor data to
SPEEDS = [1, 6, 11, 16, 21, 26]

TRIAL_COLORS = [
    ('#c5a3d4', '#9944ff'), ('#81aad4', '#083b70'), ('#a3d4a3', '#087008'),
    ('#d4a3a3', '#700808'), ('#d4c5a3', '#705c08'), ('#a3d4d4', '#087070'),
    ('#d4a3c5', '#70085c')
]
DIR_COLORS = {'Increasing': ('#d3d3d3', '#555555'), 'Decreasing': ('#888888', '#000000')}
# ========================================================

def apply_clean_axes(ax):
    ax.xaxis.set_tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline, NO interpolation."""
    if len(chunk) < 1:
        return None
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    return chunk[local_max_idx]

def plot_1x3_hysteresis(df, material_name, y_col_mean, y_col_std, y_label, output_dir, filename_suffix):
    if df.empty: return
    
    volumes = ["500", "750", "1000"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98, bottom=0.15, top=0.85 if SHOW_TITLES else 0.95)

    #if SHOW_TITLES: fig.suptitle(f"{material_name} - {y_label} Hysteresis", fontsize=FONT_TITLE, fontweight='bold', y=0.98)

    for i, vol in enumerate(volumes):
        ax = axes[i]
        vol_df = df[df['Volume'] == vol]
        if SHOW_TITLES: ax.set_title(f"Volume: {vol}", fontsize=FONT_TITLE, fontweight='bold')
        if vol_df.empty:
            apply_clean_axes(ax)
            continue

        for direct, (face_c, edge_c) in DIR_COLORS.items():
            d_data = vol_df[vol_df['Direction'] == direct].sort_values('Motor Speed (RPM)')
            if not d_data.empty:
                ax.errorbar(d_data['Motor Speed (RPM)'], d_data[y_col_mean], yerr=d_data[y_col_std], 
                            fmt='o-', color=edge_c, markerfacecolor=face_c, markeredgecolor=edge_c, 
                            label="Accelerating" if direct == "Increasing" else "Decelerating", 
                                        capsize=4, lw=2, markersize=10, elinewidth=2)

        apply_clean_axes(ax)
        if i == 0: ax.set_ylabel(y_label, fontsize=FONT_LABEL)
        if i == 1: ax.set_xlabel("Speed (RPM)", fontsize=FONT_LABEL)

    if SHOW_LEGEND:
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles: axes[-1].legend(handles, labels, loc='lower right', fontsize=12, framealpha=0.9)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = f"{material_name.lower()}_{filename_suffix}.png"
    plt.savefig(os.path.join(output_dir, safe_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {safe_name}")

def plot_1x3_timeseries(df, material_name, output_dir):
    if df.empty: return
    
    volumes = ["500", "750", "1000"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98, bottom=0.15, top=0.85 if SHOW_TITLES else 0.95)

    if SHOW_TITLES: fig.suptitle(f"{material_name} - Angle Over Time", fontsize=FONT_TITLE, fontweight='bold', y=0.98)

    for i, vol in enumerate(volumes):
        ax = axes[i]
        if SHOW_TITLES: ax.set_title(f"Volume: {vol}", fontsize=FONT_TITLE, fontweight='bold')
            
        vol_df = df[df['Volume'] == vol]
        if not vol_df.empty:
            ax.scatter(vol_df['Time (min)'], vol_df['Angle (deg)'], color='black', s=0.5, alpha=0.3, edgecolors='none')
        
        apply_clean_axes(ax)
        ax.set_xlim(0, 60) 
        ax.set_ylim(-5, 95) 
        if i == 0: ax.set_ylabel("Angle (deg)", fontsize=FONT_LABEL)
        if i == 1: ax.set_xlabel("Time (min)", fontsize=FONT_LABEL)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = f"{material_name.lower()}_angle_timeseries.png"
    plt.savefig(os.path.join(output_dir, safe_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {safe_name}")

def plot_1x3_colored_timeseries(df, material_name, y_col, y_label, output_dir, filename_suffix):
    if df.empty: return
    
    volumes = ["500", "750", "1000"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98, bottom=0.15, top=0.85 if SHOW_TITLES else 0.95)

    if SHOW_TITLES: fig.suptitle(f"{material_name} - {y_label} Over Time", fontsize=FONT_TITLE, fontweight='bold', y=0.98)

    max_trials = 0

    for i, vol in enumerate(volumes):
        ax = axes[i]
        if SHOW_TITLES: ax.set_title(f"Volume: {vol}", fontsize=FONT_TITLE, fontweight='bold')
            
        vol_df = df[df['Volume'] == vol]
        if not vol_df.empty:
            trials = sorted(vol_df['Trial_ID'].unique())
            max_trials = max(max_trials, len(trials))
            
            for j, trial in enumerate(trials):
                trial_df = vol_df[vol_df['Trial_ID'] == trial]
                face_c, edge_c = TRIAL_COLORS[j % len(TRIAL_COLORS)]
                
                # Matches your exact scatter styling
                ax.scatter(trial_df['Time (min)'], trial_df[y_col], 
                           s=10, facecolors=face_c, edgecolors=edge_c, linewidth=0.5, alpha=0.9)
        
        apply_clean_axes(ax)
        ax.set_xlim(0, 60) 
        if i == 0: ax.set_ylabel(y_label, fontsize=FONT_LABEL)
        if i == 1: ax.set_xlabel("Time (min)", fontsize=FONT_LABEL)

    if SHOW_LEGEND and max_trials > 0:
        legend_handles = [Line2D([0], [0], marker='o', color='w', label=f"Trial {t+1}", 
                                 markerfacecolor=TRIAL_COLORS[t%len(TRIAL_COLORS)][0], 
                                 markeredgecolor=TRIAL_COLORS[t%len(TRIAL_COLORS)][1], markersize=10) 
                          for t in range(max_trials)]
        axes[-1].legend(handles=legend_handles, loc='lower right', fontsize=12)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = f"{material_name.lower()}_{filename_suffix}.png"
    plt.savefig(os.path.join(output_dir, safe_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {safe_name}")

# ==========================================
# DATA WRANGLING & EXECUTION
# ==========================================
def main(parent_dir):
    normalized_path = parent_dir.replace('\\', '/').rstrip('/')
    material = os.path.basename(normalized_path)
    output_dir = os.path.join(normalized_path, "Volume_Master_Results")
    
    print(f"=== Starting Volume Grid Generation for: {material} ===")
    
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]

    all_raw_data = []
    all_peak_data = []

    for root, dirs, files in os.walk(normalized_path):
        if "experiment_log.csv" in files:
            vol = next((v for v in ["500", "750", "1000"] if v in root.split(os.sep)), None)
            if not vol: continue
                
            csv_path = os.path.join(root, "experiment_log.csv")
            try:
                df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
                df = df.dropna(subset=["motor_speed", "ellipse_angle_deg", "CH2_volts"])
                if df.empty: continue
                
                start_ms = df["ms"].min()
                df["Time (min)"] = (df["ms"] - start_ms) / 60000.0
                df["Motor Speed (RPM)"] = df["motor_speed"].apply(lambda x: min(SPEEDS, key=lambda s: abs(s - x)) if x > 0.5 else 0)
                
                max_val = df["motor_speed"].max()
                max_indices = df.index[df["motor_speed"] == max_val].tolist()
                if not max_indices: continue
                
                mid_idx = max_indices[len(max_indices) // 2]
                df["Direction"] = "Increasing"
                df.loc[mid_idx + 1:, "Direction"] = "Decreasing"
                
                df["Volume"] = vol
                df["Trial_ID"] = root
                df["Angle (deg)"] = df["ellipse_angle_deg"]
                
                # Flip the voltage!
                multiplier = -1.0 if FLIP_CH2 else 1.0
                df["Voltage (V)"] = pd.to_numeric(df[VOLTAGE_COL], errors='coerce') * multiplier
                df = df.dropna(subset=["Voltage (V)"])
                
                # 1. Save the raw data for Angle and Hysteresis calculations
                all_raw_data.append(df[["Trial_ID", "Volume", "Time (min)", "Motor Speed (RPM)", "Direction", "Angle (deg)", "Voltage (V)"]])

                # 2. Extract ONLY the peaks for the Voltage Time Series
                times = df["Time (min)"].values
                speeds = df["motor_speed"].values 
                volts = df["Voltage (V)"].values
                
                t_peaks, v_peaks = [], []
                current_idx = 0
                n_samples = len(times)
                
                # Create a rolling memory of the last 5 valid peaks to track the dominant trend
                recent_peaks = collections.deque(maxlen=5)

                while current_idx < n_samples:
                    curr_t = times[current_idx]
                    curr_speed = speeds[current_idx]
                    effective_speed = 1.0 if curr_speed < 0.5 else curr_speed
                    window_min = 1.0 / effective_speed
                    window_end_t = curr_t + window_min

                    end_idx = current_idx
                    while end_idx < n_samples and times[end_idx] < window_end_t:
                        end_idx += 1

                    chunk_v = volts[current_idx:end_idx]
                    
                    if len(chunk_v) > 0:
                        baseline = np.median(chunk_v)
                        c_chunk = chunk_v - baseline
                        
                        # Find both the local max and local min deviation
                        local_max = np.max(c_chunk)
                        local_min = np.min(c_chunk)
                        
                        # The absolute strongest signal in this window is our candidate
                        candidate = local_max if abs(local_max) > abs(local_min) else local_min
                        
                        # --- SMART RC BOUNCE FILTER ---
                        is_valid = True
                        if len(recent_peaks) == 5:
                            # What is the normal physical behavior right now?
                            recent_median = np.median(recent_peaks)
                            recent_mag = abs(recent_median)
                            
                            # Check 1: Is the candidate the OPPOSITE polarity of our recent trend?
                            if (candidate * recent_median) < 0:
                                # Check 2: Is it weaker than a real avalanche? (Threshold set to 75%)
                                if abs(candidate) < (recent_mag * 0.75):
                                    is_valid = False # It is a ghost tail. Drop it.
                                    
                        if is_valid:
                            # Log the true peak
                            local_idx = np.where(c_chunk == candidate)[0][0]
                            t_peaks.append(times[current_idx + local_idx])
                            v_peaks.append(candidate)
                            recent_peaks.append(candidate) # Add to memory

                    current_idx = end_idx
                    if current_idx >= n_samples: break
                
                peaks_df = pd.DataFrame({
                    "Trial_ID": root,
                    "Volume": vol,
                    "Time (min)": t_peaks,
                    "Voltage Peak (V)": v_peaks
                })
                all_peak_data.append(peaks_df)

            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")

    if not all_raw_data:
        print("No valid data found to plot!")
        return

    # Combine Data
    master_raw_df = pd.concat(all_raw_data, ignore_index=True)
    master_peak_df = pd.concat(all_peak_data, ignore_index=True)

    ts_raw_df = master_raw_df[master_raw_df["Motor Speed (RPM)"] > 0]

    # Aggregation for Hysteresis
    trial_df = ts_raw_df.groupby(["Trial_ID", "Volume", "Motor Speed (RPM)", "Direction"]).agg(
        Trial_Angle_Mean=("Angle (deg)", "mean"),
        Trial_Voltage_Std=("Voltage (V)", "std") 
    ).reset_index()

    hyst_df = trial_df.groupby(["Volume", "Motor Speed (RPM)", "Direction"]).agg(
        Angle_Mean=("Trial_Angle_Mean", "mean"),
        Angle_Std=("Trial_Angle_Mean", "std"),
        Voltage_Std_Mean=("Trial_Voltage_Std", "mean"), 
        Voltage_Std_Err=("Trial_Voltage_Std", "std")    
    ).reset_index()

    print("  -> Data aggregated successfully. Generating plots...")

    plot_1x3_timeseries(ts_raw_df, material, output_dir=output_dir)
    
    plot_1x3_colored_timeseries(master_peak_df, material, 
                                y_col="Voltage Peak (V)", y_label="Voltage (V)", 
                                output_dir=output_dir, filename_suffix="voltage_timeseries")
    
    plot_1x3_hysteresis(hyst_df, material, "Angle_Mean", "Angle_Std", 
                        "Angle of Repose (deg)", output_dir, "angle_mean_Hysteresis")
                        
    plot_1x3_hysteresis(hyst_df, material, "Voltage_Std_Mean", "Voltage_Std_Err", 
                        "Std Dev Voltage (V)", output_dir, "voltage_std_Hysteresis")

    print(f"=== All reports generated for {material} ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python volume_comparison_plots.py <path_to_material_folder>")
    else:
        main(sys.argv[1])