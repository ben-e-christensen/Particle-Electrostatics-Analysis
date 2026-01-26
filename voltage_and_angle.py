import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Ensemble_Results"
VOLTAGE_COL = "CH2_volts"
HARD_CUTOFF_MIN = 60.0  # Full duration
# =================================================

def process_separate_graphs(parent_dir):
    print(f"--- Starting Dual Analysis (Separate Graphs) for: {parent_dir} ---")

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

    # Containers
    angle_data_frames = []
    all_peaks_time = []
    all_peaks_val = []

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    print("Processing trials...")

    for trial_path in trial_folders:
        try:
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            
            if df.empty: continue

            # --- PRE-PROCESSING ---
            start_ms = df["ms"].min() 
            df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0

            # Global Filter
            df = df[df["rel_time_min"] <= HARD_CUTOFF_MIN]
            if df.empty: continue
            
            df = df.sort_values(by="ms")

            # --- COLLECTION 1: ANGLE DATA ---
            # Just grab the raw rows for the angle cloud
            df_angle = df.dropna(subset=["ellipse_angle_deg"])
            if not df_angle.empty:
                angle_data_frames.append(df_angle[["rel_time_min", "ellipse_angle_deg"]])

            # --- COLLECTION 2: VOLTAGE PEAKS (Interpolated) ---
            df_volt = df.dropna(subset=["motor_speed", VOLTAGE_COL])
            df_volt[VOLTAGE_COL] = pd.to_numeric(df_volt[VOLTAGE_COL], errors='coerce')
            df_volt = df_volt.dropna(subset=[VOLTAGE_COL])

            if not df_volt.empty:
                times = df_volt["rel_time_min"].values
                speeds = df_volt["motor_speed"].values
                volts = df_volt[VOLTAGE_COL].values

                current_idx = 0
                n_samples = len(times)

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

                    # Interpolation Logic
                    if len(chunk_volts) >= 3:
                        local_max_idx = np.argmax(chunk_volts)
                        
                        if 0 < local_max_idx < len(chunk_volts) - 1:
                            y1 = chunk_volts[local_max_idx - 1]
                            y2 = chunk_volts[local_max_idx]
                            y3 = chunk_volts[local_max_idx + 1]
                            
                            denom = (2 * (y1 - 2*y2 + y3))
                            if denom != 0:
                                delta = (y1 - y3) / denom
                                interpolated_val = y2 - (0.25 * (y1 - y3) * delta)
                            else:
                                interpolated_val = y2
                        else:
                            interpolated_val = np.max(chunk_volts)

                        all_peaks_time.append(curr_t)
                        all_peaks_val.append(interpolated_val)

                    elif len(chunk_volts) > 0:
                        all_peaks_time.append(curr_t)
                        all_peaks_val.append(np.max(chunk_volts))

                    current_idx = end_idx
                    if current_idx >= n_samples: break

        except Exception as e:
            print(f"Skipping {os.path.basename(trial_path)}: {e}")

    # ================= PLOT 1: ANGLE OF REPOSE =================
    print("Generating Angle Plot...")
    plt.figure(figsize=(12, 6))
    
    if angle_data_frames:
        master_angle_df = pd.concat(angle_data_frames)
        x_angle = master_angle_df["rel_time_min"].values
        y_angle = master_angle_df["ellipse_angle_deg"].values
        
        plt.scatter(x_angle, y_angle, s=0.5, color='black', alpha=0.02, rasterized=True)
        
        y_min, y_max = y_angle.min(), y_angle.max()
        buff = (y_max - y_min) * 0.1 if y_max != y_min else 5
        plt.ylim(y_min - buff, y_max + buff)
        plt.title("Angle Sampled At 100Hz", fontsize=20, fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No Angle Data", ha='center')
        plt.title("Angle Data Missing")

    plt.xlabel("Time (min)", fontsize=14)
    plt.ylabel("Angle (deg)", fontsize=14)
    plt.xlim(0, HARD_CUTOFF_MIN)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    angle_path = os.path.join(output_dir, "Global_Angle_Continuous.png")
    plt.savefig(angle_path, dpi=300)
    plt.close() # Close to start fresh for next plot
    print(f"Saved: {angle_path}")

    # ================= PLOT 2: PEAK VOLTAGE =================
    print("Generating Voltage Plot...")
    plt.figure(figsize=(12, 6))

    if all_peaks_time:
        x_peaks = np.array(all_peaks_time)
        y_peaks = np.array(all_peaks_val)
        
        plt.scatter(x_peaks, y_peaks, s=25, color='tab:blue', alpha=0.5, label='Interpolated Peak Voltage')
        
        y_min, y_max = y_peaks.min(), y_peaks.max()
        buff = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
        plt.ylim(y_min - buff, y_max + buff)
        plt.title("Interpolated Peak Charge Every Rotation", fontsize=20, fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No Voltage Data", ha='center')
        plt.title("Voltage Data Missing")

    plt.xlabel("Time (min)", fontsize=14)
    plt.ylabel("Peak Voltage (V)", fontsize=14)
    plt.xlim(0, HARD_CUTOFF_MIN)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    volt_path = os.path.join(output_dir, "Global_Voltage_Interpolated.png")
    plt.savefig(volt_path, dpi=300)
    plt.close()
    print(f"Saved: {volt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_separate_graphs.py <path_to_PARENT_folder>")
    else:
        process_separate_graphs(sys.argv[1])