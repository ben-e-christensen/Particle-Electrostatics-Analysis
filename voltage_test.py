import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Ensemble_Results"
VOLTAGE_COL_CH2 = "CH2_volts"
VOLTAGE_COL_CH3 = "CH3_volts"
HARD_CUTOFF_MIN = 60.0  # Full duration

# --- PLOT SETTINGS ---
FIXED_VOLT_YMIN = -0.025
FIXED_VOLT_YMAX = 0.025

# --- GLOBAL BASELINE & FILTER SETTINGS ---
# Use the first X minutes to calculate the true "zero" state
GLOBAL_BASELINE_MINUTES = 5.0 

# Moving average window to kill static. 1 = No smoothing. 7 is a good default.
SMOOTHING_WINDOW = 7 

# Color Palette for distinct trials
TRIAL_COLORS = [
    'grey',           # Trial 1
    'skyblue',        # Trial 2
    'navy',           # Trial 3
    'mediumpurple',   # Trial 4 (Extra)
    'indigo',         # Trial 5 (Extra)
    'black'           # Fallback
]
# =================================================

def extract_tracked_peaks(times, volts, speeds, global_zero):
    """
    Hybrid Tracker: 
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
        
    return t_peaks, v_adj_peaks

def process_separate_graphs(parent_dir):
    print(f"--- Starting Hybrid Analysis for: {parent_dir} ---")

    trial_folders = []
    for root, dirs, files in os.walk(parent_dir):
        if "experiment_log.csv" in files:
            trial_folders.append(root)

    if not trial_folders:
        print("No trial folders found!")
        return
    
    trial_folders.sort()
    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    angle_data_grouped = []
    ch2_voltage_data_grouped = []
    ch3_voltage_data_grouped = []

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    print("Processing trials...")

    for i, trial_path in enumerate(trial_folders):
        try:
            trial_name = os.path.basename(trial_path)
            print(f"  Reading Trial {i+1}: {trial_name}")
            
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", 
                             dtype={"ms": "float32", "CH2_volts": "float32", "CH3_volts": "float32", "ellipse_angle_deg": "float32"})
            
            if df.empty: continue

            # --- PRE-PROCESSING ---
            start_ms = df["ms"].min() 
            df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
            df = df[df["rel_time_min"] <= HARD_CUTOFF_MIN]
            if df.empty: continue
            df.sort_values(by="ms", inplace=True)

            # --- COLLECTION 1: ANGLE DATA ---
            df_angle = df.dropna(subset=["ellipse_angle_deg"])
            if len(df_angle) > 100000: 
                step = len(df_angle) // 100000
                df_angle = df_angle.iloc[::step]

            if not df_angle.empty:
                angle_data_grouped.append((i, df_angle["rel_time_min"].values, df_angle["ellipse_angle_deg"].values))

            # --- COLLECTION 2: VOLTAGE PEAKS (CH2 Flipped & CH3 Standard) ---
            df_volt = df.dropna(subset=["motor_speed", VOLTAGE_COL_CH2, VOLTAGE_COL_CH3]).copy()
            
            # Convert and FLIP CH2 ONLY
            df_volt[VOLTAGE_COL_CH2] = pd.to_numeric(df_volt[VOLTAGE_COL_CH2], errors='coerce') * -1.0
            df_volt[VOLTAGE_COL_CH3] = pd.to_numeric(df_volt[VOLTAGE_COL_CH3], errors='coerce')
            df_volt = df_volt.dropna(subset=[VOLTAGE_COL_CH2, VOLTAGE_COL_CH3])

            if not df_volt.empty:
                # APPLY GENTLE MOVING AVERAGE TO KILL STATIC
                if SMOOTHING_WINDOW > 1:
                    df_volt[VOLTAGE_COL_CH2] = df_volt[VOLTAGE_COL_CH2].rolling(window=SMOOTHING_WINDOW, center=True).mean()
                    df_volt[VOLTAGE_COL_CH3] = df_volt[VOLTAGE_COL_CH3].rolling(window=SMOOTHING_WINDOW, center=True).mean()
                    df_volt = df_volt.dropna(subset=[VOLTAGE_COL_CH2, VOLTAGE_COL_CH3])

                times_full = df_volt["rel_time_min"].values
                speeds_full = df_volt["motor_speed"].values
                volts_ch2 = df_volt[VOLTAGE_COL_CH2].values
                volts_ch3 = df_volt[VOLTAGE_COL_CH3].values
                
                # --- CALCULATE GLOBAL ZERO (First 5 Mins) ---
                baseline_mask = times_full <= GLOBAL_BASELINE_MINUTES
                
                if not baseline_mask.any():
                    global_zero_ch2 = np.median(volts_ch2)
                    global_zero_ch3 = np.median(volts_ch3)
                else:
                    global_zero_ch2 = np.median(volts_ch2[baseline_mask])
                    global_zero_ch3 = np.median(volts_ch3[baseline_mask])

                # Process CH2
                t_p2, v_adj2 = extract_tracked_peaks(times_full, volts_ch2, speeds_full, global_zero_ch2)
                if t_p2:
                    ch2_voltage_data_grouped.append((i, np.array(t_p2), np.array(v_adj2)))

                # Process CH3
                t_p3, v_adj3 = extract_tracked_peaks(times_full, volts_ch3, speeds_full, global_zero_ch3)
                if t_p3:
                    ch3_voltage_data_grouped.append((i, np.array(t_p3), np.array(v_adj3)))

        except Exception as e:
            print(f"Skipping {os.path.basename(trial_path)}: {e}")

    # --- PLOTTING HELPER FUNCTION ---
    def save_plot(data_grouped, title, ylabel, filename, is_angle=False, y_limits=None):
        print(f"Generating {title} Plot...")
        plt.figure(figsize=(12, 6))
        
        if data_grouped:
            for (trial_idx, x, y) in data_grouped:
                c = 'black' if is_angle else TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
                alpha = 0.05 if is_angle else 0.6
                s = 0.5 if is_angle else 25
                label = None if is_angle else f'Trial {trial_idx+1}'
                
                plt.scatter(x, y, s=s, color=c, alpha=alpha, label=label, rasterized=True)
            
            if y_limits:
                plt.ylim(y_limits)
            elif is_angle:
                all_y = np.concatenate([y for _, _, y in data_grouped])
                y_min, y_max = all_y.min(), all_y.max()
                buff = (y_max - y_min) * 0.1 if y_max != y_min else 5
                plt.ylim(y_min - buff, y_max + buff)
                
            plt.title(title, fontsize=20, fontweight='bold')
        else:
            plt.text(0.5, 0.5, f"No Data Found", ha='center', transform=plt.gca().transAxes)
            plt.title(f"{title} (Missing Data)")

        plt.xlabel("Time (min)", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xlim(0, HARD_CUTOFF_MIN)
        plt.grid(True, alpha=0.3)
        if not is_angle and data_grouped:
            plt.legend(fontsize=10, loc='upper right')
        plt.tick_params(axis='both', which='major', labelsize=14) 
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

    # Generate the 3 graphs
    save_plot(angle_data_grouped, "Angle Sampled At 100Hz", "Angle (deg)", "Global_Angle_Continuous.png", is_angle=True)
    save_plot(ch2_voltage_data_grouped, "CH2: Global Zero Adjusted Peak Charge (Flipped)", "Adjusted Peak Voltage (V)", "Global_Voltage_CH2_Flipped.png", y_limits=(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX))
    save_plot(ch3_voltage_data_grouped, "CH3: Global Zero Adjusted Peak Charge", "Adjusted Peak Voltage (V)", "Global_Voltage_CH3.png", y_limits=(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_separate_graphs.py <path_to_PARENT_folder>")
    else:
        process_separate_graphs(sys.argv[1])