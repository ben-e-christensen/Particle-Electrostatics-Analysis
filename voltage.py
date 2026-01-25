import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Ensemble_Macro_Results"
VOLTAGE_COL = "CH2_volts" 
HARD_CUTOFF_MIN = 30.0

# --- PEAK DETECTION SETTINGS ---
ROLLING_WINDOW_SEC = 1.0 
PEAK_HEIGHT_THRESH = 0.015 
PEAK_MIN_DIST_SEC = 2.0
# =================================================

def process_macro_peaks(parent_dir):
    print(f"--- Starting Macro Peak Analysis (Side-by-Side) for: {parent_dir} ---")
    
    # 1. DISCOVER TRIALS
    trial_folders = []
    for root, dirs, files in os.walk(parent_dir):
        if "experiment_log.csv" in files:
            trial_folders.append(root)
    
    if not trial_folders:
        print("No trial folders found!")
        return

    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 2. DETECT & COLLECT PEAKS
    print("Scanning trials and detecting peaks...")
    
    # We store simple (time, value) tuples
    all_increasing_peaks = {"time": [], "value": []}
    all_decreasing_peaks = {"time": [], "value": []}

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]

    for trial_path in trial_folders:
        try:
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            
            # Basic cleaning
            df = df.dropna(subset=["motor_speed", "ms", VOLTAGE_COL])
            df[VOLTAGE_COL] = pd.to_numeric(df[VOLTAGE_COL], errors='coerce')
            df = df.dropna(subset=[VOLTAGE_COL])
            if df.empty: continue

            # Determine split point
            max_val = df["motor_speed"].max()
            max_indices = df.index[df["motor_speed"] == max_val].tolist()
            if not max_indices: continue
            mid_idx = max_indices[len(max_indices) // 2]
            
            # --- PHASE 1: INCREASING ---
            df_inc = df.loc[:mid_idx].copy()
            if not df_inc.empty:
                start_ms = df_inc["ms"].iloc[0]
                df_inc["rel_time_min"] = (df_inc["ms"] - start_ms) / 60000.0
                
                # Sample rate estimation
                sr_ms = (df_inc["ms"].iloc[-1] - df_inc["ms"].iloc[0]) / len(df_inc)
                if sr_ms <= 0: sr_ms = 10
                
                # Signal Processing
                window_samples = int((ROLLING_WINDOW_SEC * 1000) / sr_ms)
                dist_samples = int((PEAK_MIN_DIST_SEC * 1000) / sr_ms)
                
                signal = df_inc[VOLTAGE_COL].rolling(window=window_samples, center=True).std()
                valid_sig = signal.dropna()
                
                if not valid_sig.empty:
                    p_idxs, _ = find_peaks(valid_sig, height=PEAK_HEIGHT_THRESH, distance=dist_samples)
                    
                    # Extract and Filter
                    times = df_inc.loc[valid_sig.index[p_idxs], "rel_time_min"].values
                    vals = valid_sig.iloc[p_idxs].values
                    
                    # Hard Cutoff Filter
                    mask = times <= HARD_CUTOFF_MIN
                    all_increasing_peaks["time"].extend(times[mask])
                    all_increasing_peaks["value"].extend(vals[mask])

            # --- PHASE 2: DECREASING ---
            df_dec = df.loc[mid_idx+1:].copy()
            if not df_dec.empty:
                start_dec_ms = df_dec["ms"].iloc[0]
                df_dec["rel_time_min"] = (df_dec["ms"] - start_dec_ms) / 60000.0
                
                sr_ms = (df_dec["ms"].iloc[-1] - df_dec["ms"].iloc[0]) / len(df_dec)
                if sr_ms <= 0: sr_ms = 10
                
                window_samples = int((ROLLING_WINDOW_SEC * 1000) / sr_ms)
                dist_samples = int((PEAK_MIN_DIST_SEC * 1000) / sr_ms)
                
                signal = df_dec[VOLTAGE_COL].rolling(window=window_samples, center=True).std()
                valid_sig = signal.dropna()
                
                if not valid_sig.empty:
                    p_idxs, _ = find_peaks(valid_sig, height=PEAK_HEIGHT_THRESH, distance=dist_samples)
                    
                    times = df_dec.loc[valid_sig.index[p_idxs], "rel_time_min"].values
                    vals = valid_sig.iloc[p_idxs].values
                    
                    mask = times <= HARD_CUTOFF_MIN
                    all_decreasing_peaks["time"].extend(times[mask])
                    all_decreasing_peaks["value"].extend(vals[mask])

        except Exception as e:
            print(f"Skipping {os.path.basename(trial_path)}: {e}")

    # 3. PLOTTING
    phases = [
        ("INCREASING", all_increasing_peaks),
        ("DECREASING", all_decreasing_peaks)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Global Y Limits
    all_vals = all_increasing_peaks["value"] + all_decreasing_peaks["value"]
    if all_vals:
        y_min, y_max = min(all_vals), max(all_vals)
        buff = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
        ylim = (max(0, y_min - buff), y_max + buff)
    else:
        ylim = (0, 0.1)

    for i, (name, data) in enumerate(phases):
        ax = axes[i]
        
        x = np.array(data["time"])
        y = np.array(data["value"])
        
        if len(x) == 0:
            ax.text(0.5, 0.5, "No Peaks Found", ha='center')
            continue

        # Plot Peaks
        ax.scatter(x, y, s=30, color='tab:blue', alpha=0.6, label='Detected Peaks')

        # Linear Trend Line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            fit_x = np.linspace(0, HARD_CUTOFF_MIN, 10)
            fit_y = p(fit_x)
            
            c = 'tab:cyan'
            ax.plot(fit_x, fit_y, color=c, linewidth=3, linestyle='--', label='Peak Trend')
            print(f"{name} Trend Slope: {z[0]:.5f}")

        ax.set_title(f"{name} Phase Peaks (0-{int(HARD_CUTOFF_MIN)} min)", fontsize=18, fontweight='bold')
        ax.set_xlabel("Time (min)", fontsize=14)
        ax.set_ylabel("Peak Voltage Std Dev (V)", fontsize=14)
        ax.set_ylim(ylim)
        ax.set_xlim(0, HARD_CUTOFF_MIN)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper right')

    plt.suptitle("Global Hysteresis: Voltage Peak Magnitude", fontsize=22, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "Global_Macro_Peaks.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_macro_peaks.py <path_to_PARENT_folder>")
    else:
        process_macro_peaks(sys.argv[1])