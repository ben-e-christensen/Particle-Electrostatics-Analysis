import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ================= CONFIGURATION =================
VOLTAGE_COL = "CH2_volts"

# Set your viewing window (in minutes)
START_MIN = 0.0
END_MIN = 60.0

# Use the first X minutes for true zero
GLOBAL_BASELINE_MINUTES = 5.0 

# Smoothing window (7-11 is usually the sweet spot)
SMOOTHING_WINDOW = 7 
FLIP_CH2 = True  
# =================================================

def extract_tracked_peaks(times, volts, speeds, global_zero):
    """
    Hybrid Tracker: 
    1. Uses local median to FIND the peak (ignoring global float).
    2. Uses global_zero to MEASURE the peak (preserving accumulated charge).
    """
    n_samples = len(times)
    if n_samples < 2: return [], [], [], [], []
        
    peak_indices = []
    local_baselines = []
    
    # 1. FIND INITIAL PEAK (Using local median)
    first_eff_speed = speeds[0] if speeds[0] > 0.5 else 1.0
    first_rot_time = 1.0 / first_eff_speed
    end_idx = np.searchsorted(times, times[0] + (first_rot_time * 1.5))
    
    window_v = volts[0:max(10, end_idx)]
    local_base = np.median(window_v)
    peak_indices.append(np.argmax(np.abs(window_v - local_base)))
    local_baselines.append(local_base)
    
    # 2. TRACK SUBSEQUENT PEAKS
    curr_p = peak_indices[0]
    while True:
        curr_t = times[curr_p]
        eff_speed = speeds[curr_p] if speeds[curr_p] > 0.5 else 1.0
        rot_time = 1.0 / eff_speed
        
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
        local_baselines.append(local_base)
        curr_p = next_p

    # 3. CALCULATE MEASUREMENTS (Relative to GLOBAL zero)
    t_peaks, v_adj_peaks, v_raw_peaks = [], [], []
    t_starts = []
    
    for i in range(len(peak_indices)):
        p_idx = peak_indices[i]
        start_idx = 0 if i == 0 else peak_indices[i-1]
            
        t_peaks.append(times[p_idx])
        # MEASURE against Global Zero
        v_adj_peaks.append(volts[p_idx] - global_zero)
        v_raw_peaks.append(volts[p_idx])
        t_starts.append(times[start_idx])
        
    return t_peaks, v_adj_peaks, v_raw_peaks, local_baselines, t_starts

def test_baseline_logic(parent_dir):
    print(f"--- Starting Hybrid Baseline Test for: {parent_dir} ---")
    
    trial_folder = None
    for root, dirs, files in os.walk(parent_dir):
        if "experiment_log.csv" in files:
            trial_folder = root
            break
    if not trial_folder: return

    csv_path = os.path.join(trial_folder, "experiment_log.csv")
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", 
                     dtype={"ms": "float32", "CH2_volts": "float32"})
    
    if df.empty: return

    # --- PRE-PROCESSING & SMOOTHING ---
    start_ms = df["ms"].min() 
    df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
    df_volt = df.dropna(subset=["motor_speed", VOLTAGE_COL]).copy()
    
    multiplier = -1.0 if FLIP_CH2 else 1.0
    df_volt[VOLTAGE_COL] = pd.to_numeric(df_volt[VOLTAGE_COL], errors='coerce') * multiplier
    df_volt.sort_values(by="rel_time_min", inplace=True)
    
    if SMOOTHING_WINDOW > 1:
        df_volt[VOLTAGE_COL] = df_volt[VOLTAGE_COL].rolling(window=SMOOTHING_WINDOW, center=True).mean()
        df_volt = df_volt.dropna(subset=[VOLTAGE_COL])

    times_full = df_volt["rel_time_min"].values
    speeds_full = df_volt["motor_speed"].values
    volts_full = df_volt[VOLTAGE_COL].values

    # --- CALCULATE GLOBAL ZERO ---
    mask = times_full <= GLOBAL_BASELINE_MINUTES
    global_zero = np.median(volts_full[mask]) if mask.any() else np.median(volts_full)

    # --- TRACKING ---
    t_peaks, v_adj, v_raw, local_bases, t_starts = extract_tracked_peaks(times_full, volts_full, speeds_full, global_zero)

    # --- SLICING ---
    raw_mask = (times_full >= START_MIN) & (times_full <= END_MIN)
    w_times = times_full[raw_mask]
    w_volts = volts_full[raw_mask]
    
    t_peaks_arr = np.array(t_peaks)
    peak_mask = (t_peaks_arr >= START_MIN) & (t_peaks_arr <= END_MIN)
    
    w_tp = t_peaks_arr[peak_mask]
    w_vp_raw = np.array(v_raw)[peak_mask]
    w_v_adj = np.array(v_adj)[peak_mask]
    w_local_bases = np.array(local_bases)[peak_mask]
    w_t_starts = np.array(t_starts)[peak_mask]

    # --- PLOTTING ---
    plt.figure(figsize=(15, 8))
    plt.plot(w_times, w_volts, color='grey', alpha=0.5, label="Smoothed Signal")
    plt.scatter(w_tp, w_vp_raw, color='red', s=30, zorder=5, label="Detected Peak")
    
    # 1. Draw Global Zero (Measurement Floor)
    plt.axhline(global_zero, color='blue', linewidth=2, label="Global Zero (Measurement Base)")
    
    for i in range(len(w_tp)):
        # 2. Draw Local Baseline (Detection Floor)
        line_start = max(START_MIN, w_t_starts[i])
        plt.hlines(w_local_bases[i], line_start, w_tp[i], color='deepskyblue', linestyle='--', alpha=0.7, 
                   label="Local Baseline (Search Base)" if i == 0 else "")
        
        # 3. Draw Adjustment Line (From Global Zero to Peak)
        plt.vlines(w_tp[i], global_zero, w_vp_raw[i], color='green', alpha=0.4, 
                   label="Accumulated Charge Height" if i == 0 else "")
        
        # Labeling
        if len(w_tp) < 60: # Only draw text if not too crowded
            text_offset = -0.0005 if w_vp_raw[i] < global_zero else 0.0005
            plt.text(w_tp[i], w_vp_raw[i] + text_offset, f"{w_v_adj[i]:.4f}V", 
                     color='black', fontsize=8, ha='center', va='top' if w_vp_raw[i] < global_zero else 'bottom')

    plt.title(f"Hybrid Tracker: Local Find, Global Measure\nWindow: {START_MIN} to {END_MIN} min", fontsize=16, fontweight='bold')
    plt.xlabel("Time (min)", fontsize=14); plt.ylabel("Voltage (V)", fontsize=14)
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3); plt.tight_layout()
    
    output_path = "Baseline_Debug_Plot_Targeted.png"
    plt.savefig(output_path, dpi=300)
    print(f"Done! Saved debug graph to: {output_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python baseline_test.py <path_to_PARENT_folder>")
    else: test_baseline_logic(sys.argv[1])