import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Ensemble_Results"
VOLTAGE_COL_CH2 = "CH2_volts"
VOLTAGE_COL_CH3 = "CH3_volts"
HARD_CUTOFF_MIN = 60.0  # Full duration

# --- ACADEMIC PLOT SETTINGS ---
# Global Font Setup
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"

# Fixed Y-Axis for Voltage Plots
FIXED_VOLT_YMIN = -0.275
FIXED_VOLT_YMAX = 0.275

# Color Palette for distinct trials (Face Color, Edge Color)
TRIAL_COLORS = [
    ('#c5a3d4', '#9944ff'), # Trial 1
    ('#81aad4', '#083b70'), # Trial 2
    ('#a3d4a3', '#087008'), # Trial 3
    ('#d4a3a3', '#700808'), # Trial 4
    ('#d4c5a3', '#705c08'), # Trial 5
    ('#a3d4d4', '#087070')  # Fallback
]
# =================================================

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline, NO interpolation."""
    if len(chunk) < 1:
        return None
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    return chunk[local_max_idx]

def apply_academic_axes(ax, xlabel, ylabel):
    """Applies the specific formatting to the axes."""
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.tick_params('both', length=7, width=1, which='major')
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True, alpha=0.3)

def process_separate_graphs(parent_dir):
    print(f"--- Starting Analysis for: {parent_dir} ---")

    # 1. DISCOVER TRIALS
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

            # --- COLLECTION 2: VOLTAGE PEAKS ---
            df_volt = df.dropna(subset=["motor_speed", VOLTAGE_COL_CH2, VOLTAGE_COL_CH3])
            df_volt.loc[:, VOLTAGE_COL_CH2] = pd.to_numeric(df_volt[VOLTAGE_COL_CH2], errors='coerce') * -1.0
            df_volt.loc[:, VOLTAGE_COL_CH3] = pd.to_numeric(df_volt[VOLTAGE_COL_CH3], errors='coerce')
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

                    chunk_ch2 = volts_ch2[current_idx:end_idx]
                    val_ch2 = get_peak(chunk_ch2)
                    if val_ch2 is not None:
                        t_peaks_ch2.append(curr_t)
                        v_peaks_ch2.append(val_ch2)

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

    # ================= PLOT 1: ANGLE OF REPOSE =================
    print("Generating Angle Plot...")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    if angle_data_grouped:
        for (trial_idx, x, y) in angle_data_grouped:
            ax.scatter(x, y, s=0.5, color='black', alpha=0.05, rasterized=True)
            
        all_y = np.concatenate([y for _, _, y in angle_data_grouped])
        y_min, y_max = all_y.min(), all_y.max()
        buff = (y_max - y_min) * 0.1 if y_max != y_min else 5
        ax.set_ylim(y_min - buff, y_max + buff)
    else:
        ax.text(0.5, 0.5, "No Angle Data", ha='center', transform=ax.transAxes)

    apply_academic_axes(ax, "Time (min)", "Angle (deg)")
    ax.set_xlim(0, HARD_CUTOFF_MIN)
    plt.tight_layout()
    
    angle_path = os.path.join(output_dir, "Global_Angle_Continuous.png")
    plt.savefig(angle_path, dpi=300)
    plt.close()
    print(f"Saved: {angle_path}")

    # ================= PLOT 2: PEAK VOLTAGE (CH2 Flipped) =================
    print("Generating CH2 Flipped Plot...")
    fig, ax = plt.subplots(figsize=(7, 5))

    if ch2_voltage_data_grouped:
        for (trial_idx, x, y) in ch2_voltage_data_grouped:
            face_c, edge_c = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
            ax.scatter(x, y, s=10, facecolors=face_c, edgecolors=edge_c, 
                       linewidth=0.5, alpha=0.9, label=f'Trial {trial_idx+1}')
        
        ax.set_ylim(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX)
    else:
        ax.text(0.5, 0.5, "No CH2 Data", ha='center', transform=ax.transAxes)

    apply_academic_axes(ax, "Time (min)", "Peak Voltage (V)")
    ax.set_xlim(0, HARD_CUTOFF_MIN)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    
    ch2_path = os.path.join(output_dir, "Global_Voltage_CH2_Flipped.png")
    plt.savefig(ch2_path, dpi=300)
    plt.close()
    print(f"Saved: {ch2_path}")

    # ================= PLOT 3: PEAK VOLTAGE (CH3 Standard) =================
    print("Generating CH3 Plot...")
    fig, ax = plt.subplots(figsize=(7, 5))

    if ch3_voltage_data_grouped:
        for (trial_idx, x, y) in ch3_voltage_data_grouped:
            face_c, edge_c = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
            ax.scatter(x, y, s=10, facecolors=face_c, edgecolors=edge_c, 
                       linewidth=0.5, alpha=0.9, label=f'Trial {trial_idx+1}')
        
        ax.set_ylim(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX)
    else:
        ax.text(0.5, 0.5, "No CH3 Data", ha='center', transform=ax.transAxes)

    apply_academic_axes(ax, "Time (min)", "Peak Voltage (V)")
    ax.set_xlim(0, HARD_CUTOFF_MIN)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    
    ch3_path = os.path.join(output_dir, "Global_Voltage_CH3.png")
    plt.savefig(ch3_path, dpi=300)
    plt.close()
    print(f"Saved: {ch3_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python voltage_and_angle_dual_graph.py <path_to_PARENT_folder>")
    else:
        process_separate_graphs(sys.argv[1])