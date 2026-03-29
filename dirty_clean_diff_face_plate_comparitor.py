import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# ================= CONFIGURATION & PATHS =================
MAIN_PARENT_DIR = r"D:\particle-data" 
OUTPUT_FOLDER_NAME = "Grand_Voltage_Comparisons"

# --- TOGGLES ---
SHOW_LEGEND = False  
SHOW_TITLES = False  

TIME_CUTOFF_MIN = 60.0  
RESAMPLE_INTERVAL_MIN = 0.5 

ATTENUATION_FACTORS = {
    'Acetal': -30.63, 'Nylon': -22.33 , 'Teflon': -40.98, 'Acrylic': 0.0 
}

# --- ACADEMIC PLOT SETTINGS ---
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
FONT_TICK = 18
FONT_LABEL = 18
FONT_TITLE = 20

LINE_STYLES = {
    'Dirty':        {'facecolor': '#d4b38c', 'edgecolor': '#8c510a', 'marker': 'o', 'ms': 9, 'label': 'Dirty'},
    'Clean':        {'facecolor': '#bfbfbf', 'edgecolor': '#404040', 'marker': 's', 'ms': 9, 'label': 'Clean (Standard)'},
    'Clean_Atten':  {'facecolor': '#80cdc1', 'edgecolor': '#018571', 'marker': '^', 'ms': 9, 'ls': '--', 'label': 'Clean (Attenuated)'},
    'Faceplates':   {'facecolor': '#f4a582', 'edgecolor': '#d7191c', 'marker': 'D', 'ms': 9, 'label': 'Diff Faceplates'}
}
# ========================================================

def apply_academic_axes(ax, xlabel="", ylabel=""):
    ax.xaxis.set_tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_tick_params(labelsize=FONT_TICK)
    ax.tick_params('both', length=7, width=1, which='major')
    if xlabel: ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel: ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.grid(True, alpha=0.3)

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline, NO interpolation."""
    if len(chunk) < 1:
        return None
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    return chunk[local_max_idx]

def process_grand_voltage_comparison(base_dir, material):
    print(f"\n--- Starting Grand Voltage Analysis (Max Local Peak Averaged) for {material} ---")
    
    num_samples = int(TIME_CUTOFF_MIN / RESAMPLE_INTERVAL_MIN) + 1
    common_time_grid = np.linspace(0.0, TIME_CUTOFF_MIN, num_samples)
    ensemble_averaged_data = {}

    condition_roots = {
        'Dirty': os.path.join(base_dir, 'Dirty', material),
        'Clean': os.path.join(base_dir, 'Clean', material),
        'Faceplates': os.path.join(base_dir, 'diff_face_plates', f"{material}-In-{material}")
    }

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    for condition, search_dir in condition_roots.items():
        if not os.path.exists(search_dir): continue

        trial_csvs = []
        for root, dirs, files in os.walk(search_dir):
            if '500' in root.split(os.sep) and 'experiment_log.csv' in files:
                trial_csvs.append(os.path.join(root, 'experiment_log.csv'))

        if not trial_csvs: continue
        all_resampled_peaks = []
        
        for csv_path in trial_csvs:
            try:
                df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", 
                                 dtype={"ms": "float32", "CH2_volts": "float32", "CH3_volts": "float32"})
                if df.empty: continue

                df = df.dropna(subset=["motor_speed", "CH2_volts", "CH3_volts"])
                if df.empty: continue

                start_ms = df["ms"].min()
                df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
                df = df[df["rel_time_min"] <= TIME_CUTOFF_MIN]
                df.sort_values(by="ms", inplace=True)

                df.loc[:, "CH2_volts"] = pd.to_numeric(df["CH2_volts"], errors='coerce') * -1.0
                df.loc[:, "CH3_volts"] = pd.to_numeric(df["CH3_volts"], errors='coerce')
                df = df.dropna(subset=["CH2_volts", "CH3_volts"])
                if df.empty: continue

                times = df["rel_time_min"].values
                speeds = df["motor_speed"].values
                volts_ch2 = df["CH2_volts"].values
                volts_ch3 = df["CH3_volts"].values

                t_peaks = []
                v_peaks_combined = []
                current_idx = 0
                n_samples_len = len(times)

                while current_idx < n_samples_len:
                    curr_t = times[current_idx]
                    curr_speed = speeds[current_idx]
                    effective_speed = 1.0 if curr_speed < 0.5 else curr_speed
                    window_min = 1.0 / effective_speed
                    window_end_t = curr_t + window_min

                    end_idx = current_idx
                    while end_idx < n_samples_len and times[end_idx] < window_end_t:
                        end_idx += 1

                    chunk_ch2 = volts_ch2[current_idx:end_idx]
                    chunk_ch3 = volts_ch3[current_idx:end_idx]
                    
                    val_ch2 = get_peak(chunk_ch2)
                    val_ch3 = get_peak(chunk_ch3)
                    
                    if val_ch2 is not None and val_ch3 is not None:
                        avg_peak = (val_ch2 + val_ch3) / 2.0
                        t_peaks.append(curr_t)
                        v_peaks_combined.append(avg_peak)

                    current_idx = end_idx
                    if current_idx >= n_samples_len: break

                if len(t_peaks) > 1:
                    resampled_v = np.interp(common_time_grid, t_peaks, v_peaks_combined)
                    all_resampled_peaks.append(resampled_v)

            except Exception as e:
                print(f"  Error reading {os.path.basename(os.path.dirname(csv_path))}: {e}")

        if all_resampled_peaks:
            ensemble_avg = np.nanmean(np.array(all_resampled_peaks), axis=0)
            ensemble_averaged_data[condition] = ensemble_avg

    if 'Clean' in ensemble_averaged_data:
        atten_pct = ATTENUATION_FACTORS.get(material, 0.0)
        multiplier = 1.0 + (atten_pct / 100.0)
        ensemble_averaged_data['Clean_Atten'] = ensemble_averaged_data['Clean'] * multiplier

    if not ensemble_averaged_data: return

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    # --- THE INJECTED FIX ---
    for condition, style in LINE_STYLES.items():
        if condition in ensemble_averaged_data:
            y_data = ensemble_averaged_data[condition]
            mark_every = int(5.0 / RESAMPLE_INTERVAL_MIN)
            
            # Line and scatter combined into one plot command
            ax.plot(common_time_grid, y_data, 
                    lw=2, 
                    ls=style.get('ls', '-'), 
                    color=style['edgecolor'], 
                    alpha=0.8, 
                    label=style['label'],
                    marker=style['marker'],
                    markerfacecolor=style['facecolor'],
                    markeredgecolor=style['edgecolor'],
                    markersize=style.get('ms', 9), # Bumped slightly to match the old scatter look
                    markevery=mark_every,
                    zorder=5)

    if SHOW_TITLES:
        ax.set_title(f"Averaged Peak Voltage: {material}", fontsize=FONT_TITLE, fontweight='bold')

    apply_academic_axes(ax, "Time (min)", "Voltage (V)")
    ax.set_xlim(0, TIME_CUTOFF_MIN)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    
    if SHOW_LEGEND: 
        # Added framealpha so it doesn't totally block your data if it overlaps
        ax.legend(fontsize=12, loc='best', framealpha=0.9)

    out_dir = os.path.join(base_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, f"MaxPeak_Voltage_{material}.png"), dpi=300)
    plt.close()
    print(f"Saved: {out_dir}\\MaxPeak_Voltage_{material}.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python grand_voltage_compare.py <MaterialFolderName>")
    process_grand_voltage_comparison(MAIN_PARENT_DIR, sys.argv[1])