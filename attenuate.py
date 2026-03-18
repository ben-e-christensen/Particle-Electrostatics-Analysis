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
# Fixed Y-Axis for Voltage Plots
FIXED_VOLT_YMIN = -0.275
FIXED_VOLT_YMAX = 0.275

# Color Palette for distinct trials (Only used for Voltage now)
TRIAL_COLORS = [
    'grey',           # Trial 1
    'skyblue',        # Trial 2
    'navy',           # Trial 3
    'mediumpurple',   # Trial 4 (Extra)
    'indigo',         # Trial 5 (Extra)
    'black'           # Fallback
]
# =================================================

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline, NO interpolation."""
    if len(chunk) < 1:
        return None

    # 1. Remove local DC offset (baseline wander) using the median
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    
    # 2. Find the index of the largest deviation from the LOCAL baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    
    # 3. Return the actual raw recorded value
    return chunk[local_max_idx]

def process_separate_graphs(parent_dir, attenuation_pct=0.0):
    print(f"--- Starting Analysis for: {parent_dir} ---")
    
    # Calculate the multiplier based on the passed percentage
    multiplier = 1.0 + (attenuation_pct / 100.0)
    if attenuation_pct != 0.0:
        print(f"--> Applying Attenuation: {attenuation_pct}% (Multiplier: {multiplier:.4f})")

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

            # Global Filter
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


            # --- COLLECTION 2: VOLTAGE PEAKS (CH2 Flipped & CH3) ---
            df_volt = df.dropna(subset=["motor_speed", VOLTAGE_COL_CH2, VOLTAGE_COL_CH3])
            
            # Convert to numeric, apply the attenuation multiplier, and FLIP CH2 ONLY
            df_volt.loc[:, VOLTAGE_COL_CH2] = pd.to_numeric(df_volt[VOLTAGE_COL_CH2], errors='coerce') * -1.0 * multiplier
            df_volt.loc[:, VOLTAGE_COL_CH3] = pd.to_numeric(df_volt[VOLTAGE_COL_CH3], errors='coerce') * multiplier
            
            # Drop any rows where coercion created NaNs
            df_volt = df_volt.dropna(subset=[VOLTAGE_COL_CH2, VOLTAGE_COL_CH3])

            if not df_volt.empty:
                times = df_volt["rel_time_min"].values
                speeds = df_volt["motor_speed"].values
                volts_ch2 = df_volt[VOLTAGE_COL_CH2].values
                volts_ch3 = df_volt[VOLTAGE_COL_CH3].values

                current_idx = 0
                n_samples = len(times)
                
                t_peaks_ch2, v_peaks_ch2 = [], []
                t_peaks_ch3, v_peaks_ch3 = [], []

                while current_idx < n_samples:
                    curr_t = times[current_idx]
                    curr_speed = speeds[current_idx]

                    effective_speed = 1.0 if curr_speed < 0.5 else curr_speed

                    window_min = 1.0 / effective_speed
                    window_end_t = curr_t + window_min

                    end_idx = current_idx
                    while end_idx < n_samples and times[end_idx] < window_end_t:
                        end_idx += 1

                    # Process CH2
                    chunk_ch2 = volts_ch2[current_idx:end_idx]
                    val_ch2 = get_peak(chunk_ch2)
                    if val_ch2 is not None:
                        t_peaks_ch2.append(curr_t)
                        v_peaks_ch2.append(val_ch2)

                    # Process CH3
                    chunk_ch3 = volts_ch3[current_idx:end_idx]
                    val_ch3 = get_peak(chunk_ch3)
                    if val_ch3 is not None:
                        t_peaks_ch3.append(curr_t)
                        v_peaks_ch3.append(val_ch3)

                    current_idx = end_idx
                    if current_idx >= n_samples: break
                
                if t_peaks_ch2:
                    ch2_voltage_data_grouped.append((i, np.array(t_peaks_ch2), np.array(v_peaks_ch2)))
                if t_peaks_ch3:
                    ch3_voltage_data_grouped.append((i, np.array(t_peaks_ch3), np.array(v_peaks_ch3)))

        except Exception as e:
            print(f"Skipping {os.path.basename(trial_path)}: {e}")

    # ================= PLOT 1: ANGLE OF REPOSE (ALL BLACK) =================
    print("Generating Angle Plot...")
    plt.figure(figsize=(12, 6))
    
    if angle_data_grouped:
        for (trial_idx, x, y) in angle_data_grouped:
            plt.scatter(x, y, s=0.5, color='black', alpha=0.05, rasterized=True)
            
        all_y = np.concatenate([y for _, _, y in angle_data_grouped])
        y_min, y_max = all_y.min(), all_y.max()
        buff = (y_max - y_min) * 0.1 if y_max != y_min else 5
        plt.ylim(y_min - buff, y_max + buff)
        plt.title("Angle Sampled At 100Hz", fontsize=20, fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No Angle Data", ha='center', transform=plt.gca().transAxes)
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

    # ================= PLOT 2: PEAK VOLTAGE (CH2 Flipped) =================
    print("Generating CH2 Flipped Plot...")
    plt.figure(figsize=(12, 6))

    if ch2_voltage_data_grouped:
        for (trial_idx, x, y) in ch2_voltage_data_grouped:
            c = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
            plt.scatter(x, y, s=25, color=c, alpha=0.6, label=f'Trial {trial_idx+1}')
        
        plt.ylim(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX)
        title_str = "CH2: Interpolated Peak Charge (Flipped)"
        if attenuation_pct != 0.0: title_str += f" [{attenuation_pct}% Attenuation]"
        plt.title(title_str, fontsize=20, fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No CH2 Data", ha='center', transform=plt.gca().transAxes)
        plt.title("CH2 Data Missing")

    plt.xlabel("Time (min)", fontsize=18)
    plt.ylabel("Peak Voltage (V)", fontsize=18)
    plt.xlim(0, HARD_CUTOFF_MIN)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=14) 
    plt.tight_layout()
    
    ch2_path = os.path.join(output_dir, "Global_Voltage_CH2_Flipped_Attenuated.png")
    plt.savefig(ch2_path, dpi=300)
    plt.close()
    print(f"Saved: {ch2_path}")

    # ================= PLOT 3: PEAK VOLTAGE (CH3 Standard) =================
    print("Generating CH3 Plot...")
    plt.figure(figsize=(12, 6))

    if ch3_voltage_data_grouped:
        for (trial_idx, x, y) in ch3_voltage_data_grouped:
            c = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
            plt.scatter(x, y, s=25, color=c, alpha=0.6, label=f'Trial {trial_idx+1}')
        
        plt.ylim(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX)
        title_str = "CH3: Interpolated Peak Charge"
        if attenuation_pct != 0.0: title_str += f" [{attenuation_pct}% Attenuation]"
        plt.title(title_str, fontsize=20, fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No CH3 Data", ha='center', transform=plt.gca().transAxes)
        plt.title("CH3 Data Missing")

    plt.xlabel("Time (min)", fontsize=18)
    plt.ylabel("Peak Voltage (V)", fontsize=18)
    plt.xlim(0, HARD_CUTOFF_MIN)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=14) 
    plt.tight_layout()
    
    ch3_path = os.path.join(output_dir, "Global_Voltage_CH3_Attenuated.png")
    plt.savefig(ch3_path, dpi=300)
    plt.close()
    print(f"Saved: {ch3_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_separate_graphs.py <path_to_PARENT_folder> [attenuation_percentage]")
        print("Example: python ensemble_separate_graphs.py ./data -41.35")
    else:
        parent_dir = sys.argv[1]
        
        # Check if an attenuation percentage was provided
        attenuation_arg = 0.0
        if len(sys.argv) >= 3:
            try:
                attenuation_arg = float(sys.argv[2])
            except ValueError:
                print(f"Warning: '{sys.argv[2]}' is not a valid number. Running with 0% attenuation.")
                
        process_separate_graphs(parent_dir, attenuation_pct=attenuation_arg)