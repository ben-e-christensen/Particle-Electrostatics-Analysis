import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
import sys
import os

# ================= CONFIGURATION =================
TIME_POINTS = ["t0", "day-1", "day-7"]

# Analysis Settings
SAMPLE_RATE = 100 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5
HARD_CUTOFF_MIN = 60.0  # Max duration in minutes to plot

# --- SIGNAL PROCESSING (NEW HYBRID TRACKER SETTINGS) ---
VOLTAGE_COL = "CH2_volts"
GLOBAL_BASELINE_MINUTES = 5.0 
SMOOTHING_WINDOW = 7 
FLIP_CH2 = True  

# Formatting Constants
FONT_TICK = 14
FONT_LABEL = 16
FONT_TITLE = 18

# Colors & Markers
TRIAL_COLORS = ['grey', 'skyblue', 'navy', 'mediumpurple', 'indigo', 'black']
DIR_COLORS = {'Increasing': 'grey', 'Decreasing': 'black'}
TRIAL_MARKERS = ['o', 's', '^', 'D', 'X', 'P', '*', 'v']
# =================================================

def get_material_name(csv_path, time_point_root):
    """Determines material name regardless of folder depth."""
    trial_dir = os.path.dirname(csv_path)
    parent_dir = os.path.dirname(trial_dir)
    parent_name = os.path.basename(parent_dir)
    
    if parent_name == os.path.basename(time_point_root) or parent_name in TIME_POINTS:
        trial_name = os.path.basename(trial_dir)
        candidate = trial_name.split("-")[0].split("_")[0]
        return candidate.lower()
    return parent_name.lower()

def extract_tracked_peaks(times, volts, speeds, global_zero):
    """
    Hybrid Phase-Locked Tracker: 
    1. Uses local median to FIND the peak (ignoring global float and RC bounds).
    2. Uses global_zero to MEASURE the peak (preserving accumulated charge).
    """
    n_samples = len(times)
    if n_samples < 2: return [], []
        
    peak_indices = []
    
    # 1. FIND INITIAL PEAK (Using local median)
    first_eff_speed = speeds[0] if speeds[0] > 0.5 else 1.0
    first_rot_time = 1.0 / first_eff_speed
    end_idx = np.searchsorted(times, times[0] + (first_rot_time * 1.5))
    
    window_v = volts[0:max(10, end_idx)]
    local_base = np.median(window_v)
    peak_indices.append(np.argmax(np.abs(window_v - local_base)))
    
    # 2. TRACK SUBSEQUENT PEAKS
    curr_p = peak_indices[0]
    while True:
        curr_t = times[curr_p]
        eff_speed = speeds[curr_p] if speeds[curr_p] > 0.5 else 1.0
        rot_time = 1.0 / eff_speed
        
        # Phase-locked search window
        search_start_t = curr_t + (rot_time * 0.5)
        search_end_t = curr_t + (rot_time * 1.5)
        
        s_idx = np.searchsorted(times, search_start_t)
        e_idx = np.searchsorted(times, search_end_t)
        
        if s_idx >= n_samples: break
        if s_idx == e_idx:
            curr_p = min(s_idx + 1, n_samples - 1)
            continue
            
        # FIND relative to LOCAL median
        window_v = volts[s_idx:e_idx]
        local_base = np.median(window_v)
        local_max_offset = np.argmax(np.abs(window_v - local_base))
        
        next_p = s_idx + local_max_offset
        peak_indices.append(next_p)
        curr_p = next_p

    # 3. CALCULATE MEASUREMENTS (Relative to GLOBAL zero)
    t_peaks, v_adj_peaks = [], []
    for p_idx in peak_indices:
        t_peaks.append(times[p_idx])
        v_adj_peaks.append(volts[p_idx] - global_zero)
        
    return np.array(t_peaks), np.array(v_adj_peaks)

def load_data(root_path):
    print(f"--- SCANNING ROOT: {root_path} ---")
    
    minute_data_list = []
    raw_trials_list = []

    for tp in TIME_POINTS:
        # Fuzzy match for time point folder
        tp_path = os.path.join(root_path, tp)
        if not os.path.exists(tp_path):
            found = False
            for actual in os.listdir(root_path):
                if actual.lower() == tp.lower():
                    tp_path = os.path.join(root_path, actual)
                    found = True; break
            if not found:
                print(f"  > Skipping {tp} (Folder not found)")
                continue

        csv_files = []
        for root, dirs, files in os.walk(tp_path):
            if "experiment_log.csv" in files:
                csv_files.append(os.path.join(root, "experiment_log.csv"))
        
        csv_files.sort()

        for trial_idx, input_csv in enumerate(csv_files):
            try:
                if "images" in input_csv.split(os.sep): continue
                material_name = get_material_name(input_csv, tp_path)
                trial_name_folder = os.path.basename(os.path.dirname(input_csv))

                # Load CSV
                cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
                        "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
                        "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
                        "ch2_flag", "ch3_flag"]
                df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
                if df.empty: continue

                # Basic Calc
                start_ms = df["ms"].min()
                df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
                df = df[df["rel_time_min"] <= HARD_CUTOFF_MIN]
                if df.empty: continue

                # --- PART A: AGGREGATE DATA (For Hysteresis) ---
                max_val = df["motor_speed"].max()
                max_indices = df.index[df["motor_speed"] == max_val].tolist()
                
                if max_indices:
                    mid_idx = max_indices[len(max_indices)//2]
                    df["direction"] = "Increasing"
                    df.loc[mid_idx+1:, "direction"] = "Decreasing"
                    
                    t0 = df["timestamp"].iloc[0]
                    df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int) + 1
                    
                    p_idxs, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
                    angle_df = df.iloc[p_idxs].copy()
                    
                    charge_agg = df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                        voltage_std=("CH2_volts", "std"), count=("CH2_volts", "count")).reset_index()
                    angle_agg = angle_df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                        angle_mean=("ellipse_angle_deg", "mean")).reset_index()
                    
                    merged = pd.merge(charge_agg, angle_agg, on=["minute_bin", "direction", "motor_speed"])
                    merged = merged[merged["count"] > (SAMPLE_RATE * MIN_SECONDS_PER_BIN)]
                    
                    if not merged.empty:
                        merged["material"] = material_name
                        merged["time_point"] = tp
                        merged["trial_id"] = trial_name_folder # Store trial ID for scatter shapes
                        merged["grouped_speed"] = merged["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
                        minute_data_list.append(merged)

                # --- PART B: RAW DATA (For Time Series) ---
                raw_angle = df.dropna(subset=["ellipse_angle_deg"])
                angle_pts = None
                if not raw_angle.empty:
                    angle_pts = (raw_angle["rel_time_min"].values, raw_angle["ellipse_angle_deg"].values)

                # Peak Tracking (Flip, Smooth, Global Zero)
                volt_pts = None
                df_v = df.dropna(subset=["motor_speed", VOLTAGE_COL]).copy()
                
                multiplier = -1.0 if FLIP_CH2 else 1.0
                df_v[VOLTAGE_COL] = pd.to_numeric(df_v[VOLTAGE_COL], errors='coerce') * multiplier
                df_v = df_v.dropna(subset=[VOLTAGE_COL])
                
                if not df_v.empty:
                    if SMOOTHING_WINDOW > 1:
                        df_v[VOLTAGE_COL] = df_v[VOLTAGE_COL].rolling(window=SMOOTHING_WINDOW, center=True).mean()
                        df_v = df_v.dropna(subset=[VOLTAGE_COL])
                    
                    times_full = df_v["rel_time_min"].values
                    speeds_full = df_v["motor_speed"].values
                    volts_full = df_v[VOLTAGE_COL].values
                    
                    # Calculate Global Zero
                    mask = times_full <= GLOBAL_BASELINE_MINUTES
                    global_zero = np.median(volts_full[mask]) if mask.any() else np.median(volts_full)

                    t_peaks, v_peaks = extract_tracked_peaks(times_full, volts_full, speeds_full, global_zero)
                    
                    if len(t_peaks) > 0:
                        volt_pts = (t_peaks, v_peaks)

                raw_trials_list.append({
                    "material": material_name,
                    "time_point": tp,
                    "trial_idx": trial_idx, # Used for COLOR logic
                    "angle_data": angle_pts,
                    "voltage_data": volt_pts
                })
                
                print(f"    Loaded: {material_name} ({tp})")

            except Exception as e:
                print(f"    Error: {e}")

    return pd.concat(minute_data_list, ignore_index=True) if minute_data_list else pd.DataFrame(), raw_trials_list

# ======================================================
# PLOTTING FUNCTIONS
# ======================================================

def plot_1x3_hysteresis(df, output_dir):
    save_dir = os.path.join(output_dir, "Hysteresis_Comparisons")
    os.makedirs(save_dir, exist_ok=True)
    
    materials = sorted(df["material"].unique())
    metrics = [("angle_mean", "Angle of Repose (deg)"), ("voltage_std", "Std Dev Voltage (V)")]

    for mat in materials:
        for y_col, y_label in metrics:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True)
            fig.suptitle(f"{mat.title()} - {y_label}", fontsize=FONT_TITLE, fontweight='bold')
            
            mat_df = df[df["material"] == mat]
            y_min, y_max = mat_df[y_col].min(), mat_df[y_col].max()
            buff = (y_max - y_min) * 0.1 if y_max != y_min else 1
            FIXED_YLIM = (y_min - buff, y_max + buff)
            
            x_min, x_max = mat_df["grouped_speed"].min(), mat_df["grouped_speed"].max()
            FIXED_XLIM = (x_min - 2, x_max + 2)

            for i, tp in enumerate(TIME_POINTS):
                ax = axs[i]
                subset = mat_df[mat_df["time_point"] == tp]
                
                if not subset.empty:
                    stats = subset.groupby(['grouped_speed', 'direction'])[y_col].agg(['mean', 'std']).reset_index()
                    for direct, color in DIR_COLORS.items():
                        d_data = stats[stats['direction'] == direct]
                        if not d_data.empty:
                            ax.errorbar(d_data['grouped_speed'], d_data['mean'], yerr=d_data['std'], 
                                        fmt='o-', color=color, label=direct, capsize=3, lw=2)
                
                ax.set_title(tp, fontweight='bold', fontsize=FONT_LABEL)
                ax.set_ylim(FIXED_YLIM)
                ax.set_xlim(FIXED_XLIM)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=FONT_TICK)
                
                if i == 1: ax.set_xlabel("Speed (RPM)", fontsize=FONT_LABEL)
                if i == 0: 
                    ax.set_ylabel(y_label, fontsize=FONT_LABEL)
                    ax.legend()
            
            plt.savefig(os.path.join(save_dir, f"{mat}_{y_col}_Hysteresis.png"))
            plt.close()

def plot_1x3_scatter(df, output_dir):
    save_dir = os.path.join(output_dir, "Scatter_Comparisons")
    os.makedirs(save_dir, exist_ok=True)
    
    materials = sorted(df["material"].unique())
    speeds = sorted(df["grouped_speed"].unique())

    for mat in materials:
        for speed in speeds:
            s_df = df[(df["material"] == mat) & (df["grouped_speed"] == speed)]
            if s_df.empty: continue

            fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True, sharex=True)
            fig.suptitle(f"{mat.title()} @ {speed} RPM - Charge vs Angle", fontsize=FONT_TITLE, fontweight='bold')
            
            x_min, x_max = s_df["voltage_std"].min(), s_df["voltage_std"].max()
            y_min, y_max = s_df["angle_mean"].min(), s_df["angle_mean"].max()
            
            x_buff = (x_max - x_min)*0.1 if x_max!=x_min else 0.1
            y_buff = (y_max - y_min)*0.1 if y_max!=y_min else 1
            
            global_sc = None

            for i, tp in enumerate(TIME_POINTS):
                ax = axs[i]
                subset = s_df[s_df["time_point"] == tp]
                
                if not subset.empty:
                    # Iterate through unique trials to assign SHAPES
                    unique_trials = sorted(subset["trial_id"].unique())
                    for t_idx, t_id in enumerate(unique_trials):
                        t_subset = subset[subset["trial_id"] == t_id]
                        marker = TRIAL_MARKERS[t_idx % len(TRIAL_MARKERS)]
                        
                        sc = ax.scatter(t_subset["voltage_std"], t_subset["angle_mean"], 
                                        c=t_subset["minute_bin"], cmap="coolwarm", vmin=1, vmax=subset["minute_bin"].max(),
                                        alpha=0.8, s=80, edgecolors='black', marker=marker)
                        global_sc = sc
                        
                        # Add trendline for this trial
                        if len(t_subset) > 2:
                            try:
                                z = np.polyfit(t_subset["voltage_std"], t_subset["angle_mean"], 1)
                                p = np.poly1d(z)
                                xr = np.linspace(t_subset["voltage_std"].min(), t_subset["voltage_std"].max(), 10)
                                ax.plot(xr, p(xr), "--k", alpha=0.4)
                            except: pass

                ax.set_title(tp, fontweight='bold', fontsize=FONT_LABEL)
                ax.set_xlim(x_min - x_buff, x_max + x_buff)
                ax.set_ylim(y_min - y_buff, y_max + y_buff)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=FONT_TICK)
                
                if i == 1: ax.set_xlabel("Voltage Std (V)", fontsize=FONT_LABEL)
                if i == 0: ax.set_ylabel("Angle (deg)", fontsize=FONT_LABEL)

            if global_sc:
                cbar = plt.colorbar(global_sc, ax=axs, label="Minute Bin")
                
                # Add Legend for Shapes
                shape_handles = [Line2D([0],[0], marker=TRIAL_MARKERS[t], color='w', label=f"Trial {t+1}", 
                                        markerfacecolor='grey', markersize=10, markeredgecolor='k') for t in range(3)]
                fig.legend(handles=shape_handles, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=10)

            plt.savefig(os.path.join(save_dir, f"{mat}_{speed}RPM_Scatter.png"))
            plt.close()

def plot_1x3_timeseries(raw_list, output_dir):
    save_dir_ang = os.path.join(output_dir, "TimeSeries_Angle")
    save_dir_vol = os.path.join(output_dir, "TimeSeries_Voltage")
    os.makedirs(save_dir_ang, exist_ok=True)
    os.makedirs(save_dir_vol, exist_ok=True)
    
    materials = sorted(list(set(r['material'] for r in raw_list)))
    
    for mat in materials:
        mat_data = [r for r in raw_list if r['material'] == mat]
        
        all_ang_y = []
        all_vol_y = []
        # Find max trial index to determine legend size
        max_trial_idx = 0

        for r in mat_data:
            if r['angle_data']: all_ang_y.extend(r['angle_data'][1])
            if r['voltage_data']: all_vol_y.extend(r['voltage_data'][1])
            max_trial_idx = max(max_trial_idx, r['trial_idx'])
            
        if all_ang_y:
            y_min, y_max = min(all_ang_y), max(all_ang_y)
            buff = (y_max - y_min) * 0.1 if y_max!=y_min else 5
            LIM_ANG = (y_min - buff, y_max + buff)
        else: LIM_ANG = (0, 90)
        
        if all_vol_y:
            y_min, y_max = min(all_vol_y), max(all_vol_y)
            buff = (y_max - y_min) * 0.1 if y_max!=y_min else 0.1
            LIM_VOL = (y_min - buff, y_max + buff)
        else: LIM_VOL = (0, 1)

        # === PLOT 1: ANGLE ===
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True, sharex=True)
        fig.suptitle(f"{mat.title()} - Angle Time Series", fontsize=FONT_TITLE, fontweight='bold')
        
        for i, tp in enumerate(TIME_POINTS):
            ax = axs[i]
            tp_data = [r for r in mat_data if r['time_point'] == tp]
            
            for item in tp_data:
                if item['angle_data']:
                    x, y = item['angle_data']
                    ax.scatter(x, y, s=0.5, color='black', alpha=0.05, rasterized=True)
            
            ax.set_title(tp, fontweight='bold', fontsize=FONT_LABEL)
            ax.set_ylim(LIM_ANG)
            ax.set_xlim(0, HARD_CUTOFF_MIN)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=FONT_TICK)
            
            if i == 1: ax.set_xlabel("Time (min)", fontsize=FONT_LABEL)
            if i == 0: ax.set_ylabel("Angle (deg)", fontsize=FONT_LABEL)
            
        plt.savefig(os.path.join(save_dir_ang, f"{mat}_Angle_TimeSeries.png"), dpi=300)
        plt.close()

        # === PLOT 2: VOLTAGE ===
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True, sharex=True)
        # Updated Title to reflect Global Zero tracking
        fig.suptitle(f"{mat.title()} - Peak Charge Time Series (Flipped & Adjusted)", fontsize=FONT_TITLE, fontweight='bold')
        
        for i, tp in enumerate(TIME_POINTS):
            ax = axs[i]
            tp_data = [r for r in mat_data if r['time_point'] == tp]
            tp_data.sort(key=lambda x: x['trial_idx'])
            
            for local_idx, item in enumerate(tp_data):
                if item['voltage_data']:
                    x, y = item['voltage_data']
                    # Use GLOBAL trial index for color to keep T1 grey across all plots
                    c = TRIAL_COLORS[item['trial_idx'] % len(TRIAL_COLORS)]
                    ax.scatter(x, y, s=15, color=c, alpha=0.6)
            
            ax.set_title(tp, fontweight='bold', fontsize=FONT_LABEL)
            ax.set_ylim(LIM_VOL)
            ax.set_xlim(0, HARD_CUTOFF_MIN)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=FONT_TICK)
            
            if i == 1: ax.set_xlabel("Time (min)", fontsize=FONT_LABEL)
            # Updated Y-label
            if i == 0: ax.set_ylabel("Adjusted Voltage (V)", fontsize=FONT_LABEL)
            
        # FORCE LEGEND ON LAST PLOT (Or Outside)
        # We manually construct the legend based on the max trials we found
        legend_handles = []
        for t_idx in range(max_trial_idx + 1):
            c = TRIAL_COLORS[t_idx % len(TRIAL_COLORS)]
            legend_handles.append(Line2D([0], [0], marker='o', color='w', label=f"Trial {t_idx+1}", 
                                         markerfacecolor=c, markersize=10))
        
        axs[2].legend(handles=legend_handles, loc='upper right', fontsize=10)

        plt.savefig(os.path.join(save_dir_vol, f"{mat}_Voltage_TimeSeries.png"), dpi=300)
        plt.close()

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python time_lapse_master_analysis_v3.py <PATH_TO_ROOT_FOLDER>")
        sys.exit(1)
        
    root_path = sys.argv[1]
    output_dir = os.path.join(root_path, "Time_Lapse_Master_Results")
    
    agg_df, raw_list = load_data(root_path)
    
    if agg_df.empty:
        print("No valid data found.")
        sys.exit(1)
        
    print(f"\nData loaded. Materials found: {agg_df['material'].unique()}")
    
    print("Generating Hysteresis Grids...")
    plot_1x3_hysteresis(agg_df, output_dir)
    
    print("Generating Scatter Grids...")
    plot_1x3_scatter(agg_df, output_dir)
    
    print("Generating Time Series Grids...")
    plot_1x3_timeseries(raw_list, output_dir)
    
    print(f"Done! Results saved to: {output_dir}")