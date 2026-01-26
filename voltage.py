import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Ensemble_Macro_Results"
VOLTAGE_COL = "CH2_volts" 
# =================================================

def process_60min_interpolated(parent_dir):
    print(f"--- Starting 60-Minute Sub-Sample Interpolation Analysis for: {parent_dir} ---")
    
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

    all_peaks_time = []
    all_peaks_val = []

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]

    for trial_path in trial_folders:
        try:
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            
            df = df.dropna(subset=["motor_speed", "ms", VOLTAGE_COL])
            df[VOLTAGE_COL] = pd.to_numeric(df[VOLTAGE_COL], errors='coerce')
            df = df.dropna(subset=[VOLTAGE_COL])
            
            if df.empty: continue
            
            df = df.sort_values(by="ms")
            
            start_trial_ms = df["ms"].iloc[0]
            df["rel_time_min"] = (df["ms"] - start_trial_ms) / 60000.0
            
            times = df["rel_time_min"].values
            speeds = df["motor_speed"].values
            volts = df[VOLTAGE_COL].values
            
            current_idx = 0
            n_samples = len(times)
            
            # --- STRICT SLIDING WINDOW ---
            while current_idx < n_samples:
                curr_t = times[current_idx]
                curr_speed = speeds[current_idx]
                
                if curr_speed < 0.5: effective_speed = 1.0 
                else: effective_speed = curr_speed
                
                window_min = 1.0 / effective_speed
                window_end_t = curr_t + window_min
                
                end_idx = current_idx
                while end_idx < n_samples and times[end_idx] < window_end_t:
                    end_idx += 1
                
                chunk_volts = volts[current_idx:end_idx]
                
                if len(chunk_volts) >= 3:
                    # 1. Find the index of the max value in this chunk
                    local_max_idx = np.argmax(chunk_volts)
                    
                    # 2. Check bounds (cannot interpolate if max is at the very edge)
                    if 0 < local_max_idx < len(chunk_volts) - 1:
                        # Get the three points (y1, y2, y3)
                        y1 = chunk_volts[local_max_idx - 1]
                        y2 = chunk_volts[local_max_idx]     # The Max
                        y3 = chunk_volts[local_max_idx + 1]
                        
                        # 3. Quadratic Interpolation Formula
                        # Calculates the peak of the parabola fitting these 3 points
                        denom = (2 * (y1 - 2*y2 + y3))
                        if denom != 0:
                            # Offset from the center index (-0.5 to +0.5 typically)
                            delta = (y1 - y3) / denom
                            interpolated_val = y2 - (0.25 * (y1 - y3) * delta)
                        else:
                            interpolated_val = y2
                    else:
                        # Fallback to raw max if at edge
                        interpolated_val = np.max(chunk_volts)

                    all_peaks_time.append(curr_t)
                    all_peaks_val.append(interpolated_val)
                
                elif len(chunk_volts) > 0:
                    # Fallback for tiny chunks
                    all_peaks_time.append(curr_t)
                    all_peaks_val.append(np.max(chunk_volts))
                
                current_idx = end_idx
                if current_idx >= n_samples: break

        except Exception as e:
            print(f"Skipping {os.path.basename(trial_path)}: {e}")

    # 3. PLOTTING
    if not all_peaks_time:
        print("No peaks found.")
        return

    plt.figure(figsize=(12, 6))
    
    x = np.array(all_peaks_time)
    y = np.array(all_peaks_val)
    
    # Scatter Plot 
    plt.scatter(x, y, s=25, color='tab:blue', alpha=0.5, label='Interpolated Peak Voltage')

    plt.title("Global Hysteresis: Sub-Sample Interpolated Peaks", fontsize=20, fontweight='bold')
    plt.xlabel("Time (min)", fontsize=14)
    plt.ylabel("Peak Voltage (V)", fontsize=14)
    
    y_min, y_max = min(y), max(y)
    buff = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
    plt.ylim(y_min - buff, y_max + buff)
    
    x_max = max(x.max(), 60)
    plt.xlim(0, x_max + 2) 
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "Global_60Min_Interpolated.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_60min_interpolated_peaks.py <path_to_PARENT_folder>")
    else:
        process_60min_interpolated(sys.argv[1])