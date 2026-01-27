import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Ensemble_Results"
VOLTAGE_COL = "CH2_volts"
HARD_CUTOFF_MIN = 60.0  # Full duration

# --- PLOT SETTINGS ---
# Fixed Y-Axis for Voltage Plot
FIXED_VOLT_YMIN = 0.0
FIXED_VOLT_YMAX = 0.225

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

def process_separate_graphs(parent_dir):
    print(f"--- Starting Dual Analysis (Colored by Trial) for: {parent_dir} ---")

    # 1. DISCOVER TRIALS
    trial_folders = []
    for root, dirs, files in os.walk(parent_dir):
        if "experiment_log.csv" in files:
            trial_folders.append(root)

    if not trial_folders:
        print("No trial folders found!")
        return
    
    # Sort folders to ensure consistent coloring order (T1, T2, T3...)
    trial_folders.sort()

    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Containers
    # We now store data as (trial_index, x_values, y_values) tuples
    angle_data_grouped = []
    voltage_data_grouped = []

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
            df_angle = df.dropna(subset=["ellipse_angle_deg"])
            if not df_angle.empty:
                # Store (trial_idx, x, y)
                angle_data_grouped.append((i, df_angle["rel_time_min"].values, df_angle["ellipse_angle_deg"].values))

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
                
                t_peaks = []
                v_peaks = []

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
                    val_to_store = None
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
                        val_to_store = interpolated_val

                    elif len(chunk_volts) > 0:
                        val_to_store = np.max(chunk_volts)
                    
                    if val_to_store is not None:
                        t_peaks.append(curr_t)
                        v_peaks.append(val_to_store)

                    current_idx = end_idx
                    if current_idx >= n_samples: break
                
                if t_peaks:
                    # Store (trial_idx, x, y)
                    voltage_data_grouped.append((i, np.array(t_peaks), np.array(v_peaks)))

        except Exception as e:
            print(f"Skipping {os.path.basename(trial_path)}: {e}")

    # ================= PLOT 1: ANGLE OF REPOSE =================
    print("Generating Angle Plot...")
    plt.figure(figsize=(12, 6))
    
    if angle_data_grouped:
        # Loop through groups to plot distinct colors
        for (trial_idx, x, y) in angle_data_grouped:
            c = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
            # Very low alpha for angle cloud because it's dense
            plt.scatter(x, y, s=0.5, color=c, alpha=0.05, rasterized=True)
            
        # Calc limits based on all data
        all_y = np.concatenate([y for _, _, y in angle_data_grouped])
        y_min, y_max = all_y.min(), all_y.max()
        buff = (y_max - y_min) * 0.1 if y_max != y_min else 5
        plt.ylim(y_min - buff, y_max + buff)
        plt.title("Angle Sampled At 100Hz", fontsize=20, fontweight='bold')
    else:
        plt.text(1, 1, "No Angle Data", ha='center')
        plt.title("Angle Data Missing")

    plt.xlabel("Time (min)", fontsize=18)
    plt.ylabel("Angle (deg)", fontsize=18)
    plt.xlim(0, HARD_CUTOFF_MIN)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14) 
    plt.tight_layout()
    
    angle_path = os.path.join(output_dir, "Global_Angle_Continuous.png")
    plt.savefig(angle_path, dpi=300)
    plt.close()
    print(f"Saved: {angle_path}")

    # ================= PLOT 2: PEAK VOLTAGE =================
    print("Generating Voltage Plot...")
    plt.figure(figsize=(12, 6))

    if voltage_data_grouped:
        for (trial_idx, x, y) in voltage_data_grouped:
            c = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
            # Higher alpha for peaks so colors are visible
            plt.scatter(x, y, s=25, color=c, alpha=0.6, label=f'Trial {trial_idx+1}')
        
        # --- FIXED LIMITS AS REQUESTED ---
        plt.ylim(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX)
        plt.title("Interpolated Peak Charge Every Rotation", fontsize=20, fontweight='bold')
    else:
        plt.text(1, 1, "No Voltage Data", ha='center')
        plt.title("Voltage Data Missing")

    plt.xlabel("Time (min)", fontsize=18)
    plt.ylabel("Peak Voltage (V)", fontsize=18)
    plt.xlim(0, HARD_CUTOFF_MIN)
    plt.grid(True, alpha=0.3)
    
    # Add Legend to identify trials
    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=14) 
    plt.tight_layout()
    
    volt_path = os.path.join(output_dir, "Global_Voltage_Interpolated.png")
    plt.savefig(volt_path, dpi=300)
    plt.close()
    print(f"Saved: {volt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_separate_graphs_colored.py <path_to_PARENT_folder>")
    else:
        process_separate_graphs(sys.argv[1])