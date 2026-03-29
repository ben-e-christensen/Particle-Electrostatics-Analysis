import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
import sys
import os

# ================= CONFIGURATION =================
TIME_POINTS = ["t0", "day-1", "day-7"]

# Title Mapping for LaTeX Rendering
TITLE_MAP = {
    "t0": r"$t_0$",
    "day-1": "Day 1",
    "day-7": "Day 7"
}

# --- TOGGLES ---
SHOW_LEGEND = False  
SHOW_TITLES = False  

# Analysis Settings
SAMPLE_RATE = 100 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5
HARD_CUTOFF_MIN = 60.0  

# --- SIGNAL PROCESSING ---
VOLTAGE_COL = "CH2_volts"
SMOOTHING_WINDOW = 7 
FLIP_CH2 = True  

# --- ACADEMIC PLOT SETTINGS ---
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"

FONT_TICK = 18
FONT_LABEL = 18
FONT_TITLE = 20

TRIAL_COLORS = [
    ('#c5a3d4', '#9944ff'), ('#81aad4', '#083b70'), ('#a3d4a3', '#087008'),
    ('#d4a3a3', '#700808'), ('#d4c5a3', '#705c08'), ('#a3d4d4', '#087070'),
    ('#d4a3c5', '#70085c')
]

DIR_COLORS = {'Increasing': ('#d3d3d3', '#555555'), 'Decreasing': ('#888888', '#000000')}
# =================================================

def apply_academic_axes(ax, xlabel="", ylabel=""):
    ax.xaxis.set_tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_tick_params(labelsize=FONT_TICK)
    ax.tick_params('both', length=7, width=1, which='major')
    if xlabel: ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel: ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.grid(True, alpha=0.3)

def get_material_name(csv_path, time_point_root):
    trial_dir = os.path.dirname(csv_path)
    parent_dir = os.path.dirname(trial_dir)
    parent_name = os.path.basename(parent_dir)
    
    if parent_name == os.path.basename(time_point_root) or parent_name in TIME_POINTS:
        trial_name = os.path.basename(trial_dir)
        candidate = trial_name.split("-")[0].split("_")[0]
        return candidate.lower()
    return parent_name.lower()

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline, NO interpolation."""
    if len(chunk) < 1:
        return None
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    return chunk[local_max_idx]

def load_data(root_path):
    print(f"--- SCANNING ROOT: {root_path} ---")
    minute_data_list = []
    raw_trials_list = []

    for tp in TIME_POINTS:
        tp_path = os.path.join(root_path, tp)
        if not os.path.exists(tp_path):
            found = False
            for actual in os.listdir(root_path):
                if actual.lower() == tp.lower():
                    tp_path = os.path.join(root_path, actual)
                    found = True; break
            if not found:
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

                cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
                        "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
                        "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
                        "ch2_flag", "ch3_flag"]
                df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
                if df.empty: continue

                start_ms = df["ms"].min()
                df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
                df = df[df["rel_time_min"] <= HARD_CUTOFF_MIN]
                if df.empty: continue

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
                        merged["trial_id"] = trial_name_folder 
                        merged["grouped_speed"] = merged["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
                        minute_data_list.append(merged)

                raw_angle = df.dropna(subset=["ellipse_angle_deg"])
                angle_pts = None
                if not raw_angle.empty:
                    angle_pts = (raw_angle["rel_time_min"].values, raw_angle["ellipse_angle_deg"].values)

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
                    
                    t_peaks, v_peaks = [], []
                    current_idx = 0
                    n_samples = len(times_full)

                    while current_idx < n_samples:
                        curr_t = times_full[current_idx]
                        curr_speed = speeds_full[current_idx]
                        effective_speed = 1.0 if curr_speed < 0.5 else curr_speed
                        window_min = 1.0 / effective_speed
                        window_end_t = curr_t + window_min

                        end_idx = current_idx
                        while end_idx < n_samples and times_full[end_idx] < window_end_t:
                            end_idx += 1

                        chunk_v = volts_full[current_idx:end_idx]
                        val = get_peak(chunk_v)
                        if val is not None:
                            t_peaks.append(curr_t)
                            v_peaks.append(val)

                        current_idx = end_idx
                        if current_idx >= n_samples: break
                    
                    if len(t_peaks) > 0:
                        volt_pts = (np.array(t_peaks), np.array(v_peaks))

                raw_trials_list.append({
                    "material": material_name,
                    "time_point": tp,
                    "trial_idx": trial_idx, 
                    "angle_data": angle_pts,
                    "voltage_data": volt_pts
                })
                print(f"    Loaded: {material_name} ({tp})")

            except Exception as e:
                print(f"    Error: {e}")

    return pd.concat(minute_data_list, ignore_index=True) if minute_data_list else pd.DataFrame(), raw_trials_list

def plot_1x3_hysteresis(df, output_dir):
    save_dir = os.path.join(output_dir, "Hysteresis_Comparisons")
    os.makedirs(save_dir, exist_ok=True)
    materials = sorted(df["material"].unique())
    metrics = [("angle_mean", "Angle of Repose (deg)"), ("voltage_std", "Std Dev Voltage (V)")]

    for mat in materials:
        for y_col, y_label in metrics:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True)
            if SHOW_TITLES:
                fig.suptitle(f"{mat.title()} - {y_label}", fontsize=FONT_TITLE)
            
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
                    for direct, (face_c, edge_c) in DIR_COLORS.items():
                        d_data = stats[stats['direction'] == direct]
                        if not d_data.empty:
                            ax.errorbar(d_data['grouped_speed'], d_data['mean'], yerr=d_data['std'], 
                                        fmt='o-', color=edge_c, markerfacecolor=face_c, markeredgecolor=edge_c, 
                                        label=direct, capsize=4, lw=2, markersize=10, elinewidth=2)
                
                # LaTeX mapped here
                if SHOW_TITLES: ax.set_title(TITLE_MAP.get(tp, tp), fontsize=FONT_LABEL)
                ax.set_ylim(FIXED_YLIM)
                ax.set_xlim(FIXED_XLIM)
                apply_academic_axes(ax, "Speed (RPM)" if i == 1 else "", y_label if i == 0 else "")
                if i == 0 and SHOW_LEGEND: ax.legend(fontsize=12)
            
            plt.savefig(os.path.join(save_dir, f"{mat}_{y_col}_Hysteresis.png"))
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
        max_trial_idx = 0

        for r in mat_data:
            if r['angle_data']: all_ang_y.extend(r['angle_data'][1])
            if r['voltage_data']: all_vol_y.extend(r['voltage_data'][1])
            max_trial_idx = max(max_trial_idx, r['trial_idx'])
            
        LIM_ANG = (min(all_ang_y) - 5, max(all_ang_y) + 5) if all_ang_y else (0, 90)
        LIM_VOL = (min(all_vol_y) - 0.025, max(all_vol_y) + 0.025) if all_vol_y else (0, 1)

        # === PLOT 1: ANGLE ===
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True, sharex=True)
        if SHOW_TITLES: fig.suptitle(f"{mat.title()} - Angle Time Series", fontsize=FONT_TITLE, fontweight='bold')
            
        for i, tp in enumerate(TIME_POINTS):
            ax = axs[i]
            tp_data = [r for r in mat_data if r['time_point'] == tp]
            for item in tp_data:
                if item['angle_data']:
                    x, y = item['angle_data']
                    ax.scatter(x, y, s=0.5, color='black', alpha=0.05, rasterized=True)
            
            # LaTeX mapped here
            if SHOW_TITLES: ax.set_title(TITLE_MAP.get(tp, tp),  fontsize=FONT_LABEL)
            ax.set_ylim(LIM_ANG)
            ax.set_xlim(0, HARD_CUTOFF_MIN)
            apply_academic_axes(ax, "Time (min)" if i == 1 else "", "Angle (deg)" if i == 0 else "")
            
        plt.savefig(os.path.join(save_dir_ang, f"{mat}_Angle_TimeSeries.png"), dpi=300)
        plt.close()

        # === PLOT 2: VOLTAGE ===
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True, sharex=True)
        
        # Uncommented and restored the Suptitle!
        #if SHOW_TITLES: fig.suptitle(f"{mat.title()} - Peak Charge Time Series", fontsize=FONT_TITLE, fontweight='bold')
            
        for i, tp in enumerate(TIME_POINTS):
            ax = axs[i]
            tp_data = [r for r in mat_data if r['time_point'] == tp]
            tp_data.sort(key=lambda x: x['trial_idx'])
            for local_idx, item in enumerate(tp_data):
                if item['voltage_data']:
                    x, y = item['voltage_data']
                    face_c, edge_c = TRIAL_COLORS[item['trial_idx'] % len(TRIAL_COLORS)]
                    ax.scatter(x, y, s=10, facecolors=face_c, edgecolors=edge_c, linewidth=0.5, alpha=0.9)
            
            # LaTeX mapped here
            if SHOW_TITLES: ax.set_title(TITLE_MAP.get(tp, tp),  fontsize=FONT_LABEL)
            ax.set_ylim(LIM_VOL)
            ax.set_xlim(0, HARD_CUTOFF_MIN)
            apply_academic_axes(ax, "Time (min)" if i == 1 else "", "Voltage (V)" if i == 0 else "")
            
        if SHOW_LEGEND:
            legend_handles = [Line2D([0], [0], marker='o', color='w', label=f"Trial {t+1}", 
                                     markerfacecolor=TRIAL_COLORS[t%len(TRIAL_COLORS)][0], 
                                     markeredgecolor=TRIAL_COLORS[t%len(TRIAL_COLORS)][1], markersize=10) 
                              for t in range(len(TRIAL_COLORS))] # <--- CHANGED TO 7 ALWAYS
            axs[2].legend(handles=legend_handles, loc='lower right', fontsize=12)

        plt.savefig(os.path.join(save_dir_vol, f"{mat}_Voltage_TimeSeries.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python time_lapse_master_analysis_v3.py <PATH_TO_ROOT_FOLDER>")
        sys.exit(1)
    root_path = sys.argv[1]
    output_dir = os.path.join(root_path, "Time_Lapse_Master_Results")
    agg_df, raw_list = load_data(root_path)
    
    if agg_df.empty: sys.exit("No valid data found.")
    
    print("Generating Hysteresis Grids...")
    plot_1x3_hysteresis(agg_df, output_dir)
    print("Generating Time Series Grids...")
    plot_1x3_timeseries(raw_list, output_dir)
    print(f"Done! Results saved to: {output_dir}")