import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Comparisons_Clean_vs_Dirty"
VOLTAGE_COL = "CH2_volts"
HARD_CUTOFF_MIN = 60.0  # Full duration

# --- PLOT SETTINGS ---
FIXED_VOLT_YMIN = 0.0
FIXED_VOLT_YMAX = 0.225

# Color Palette for distinct trials (Only used for Voltage)
TRIAL_COLORS = ['grey', 'skyblue', 'navy', 'mediumpurple', 'indigo', 'black']

# Formatting
FONT_TICK = 14
FONT_LABEL = 16
FONT_TITLE = 18
# =================================================

def load_trials_from_folder(target_dir):
    """
    Scans a specific folder for trial subfolders.
    Returns parsed Angle and Voltage data.
    """
    angle_data = []
    voltage_data = []

    if not os.path.exists(target_dir):
        print(f"  [!] Folder not found: {target_dir}")
        return angle_data, voltage_data

    # Discover Trial Folders
    trial_folders = []
    for root, dirs, files in os.walk(target_dir):
        if "experiment_log.csv" in files:
            trial_folders.append(root)
    
    trial_folders.sort() 
    
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    for i, trial_path in enumerate(trial_folders):
        try:
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            if df.empty: continue

            # Pre-Processing
            start_ms = df["ms"].min() 
            df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
            df = df[df["rel_time_min"] <= HARD_CUTOFF_MIN]
            if df.empty: continue
            df = df.sort_values(by="ms")

            # 1. ANGLE DATA
            df_angle = df.dropna(subset=["ellipse_angle_deg"])
            if not df_angle.empty:
                angle_data.append((i, df_angle["rel_time_min"].values, df_angle["ellipse_angle_deg"].values))

            # 2. VOLTAGE DATA (Interpolated)
            df_volt = df.dropna(subset=["motor_speed", VOLTAGE_COL])
            df_volt[VOLTAGE_COL] = pd.to_numeric(df_volt[VOLTAGE_COL], errors='coerce')
            df_volt = df_volt.dropna(subset=[VOLTAGE_COL])

            if not df_volt.empty:
                times = df_volt["rel_time_min"].values
                speeds = df_volt["motor_speed"].values
                volts = df_volt[VOLTAGE_COL].values
                
                t_peaks, v_peaks = [], []
                current_idx, n_samples = 0, len(times)

                while current_idx < n_samples:
                    curr_t = times[current_idx]
                    curr_speed = speeds[current_idx]
                    effective_speed = 1.0 if curr_speed < 0.5 else curr_speed
                    
                    window_end_t = curr_t + (1.0 / effective_speed)
                    end_idx = current_idx
                    while end_idx < n_samples and times[end_idx] < window_end_t:
                        end_idx += 1
                    
                    chunk = volts[current_idx:end_idx]
                    
                    val = None
                    if len(chunk) >= 3:
                        idx = np.argmax(chunk)
                        if 0 < idx < len(chunk) - 1:
                            y1, y2, y3 = chunk[idx-1], chunk[idx], chunk[idx+1]
                            denom = 2 * (y1 - 2*y2 + y3)
                            if denom != 0:
                                delta = (y1 - y3) / denom
                                val = y2 - (0.25 * (y1 - y3) * delta)
                            else: val = y2
                        else: val = np.max(chunk)
                    elif len(chunk) > 0: val = np.max(chunk)
                    
                    if val:
                        t_peaks.append(curr_t)
                        v_peaks.append(val)
                    
                    current_idx = end_idx
                
                if t_peaks:
                    voltage_data.append((i, np.array(t_peaks), np.array(v_peaks)))

        except Exception as e:
            print(f"Skipping trial {os.path.basename(trial_path)}: {e}")
            
    return angle_data, voltage_data

def process_comparison(parent_dir, target_vol, target_mat):
    print(f"--- Starting Comparison: {target_mat} {target_vol} ---")
    
    # Construct Paths: Parent/Condition/Material/Volume
    clean_path = os.path.join(parent_dir, "Clean", target_mat, target_vol)
    dirty_path = os.path.join(parent_dir, "Dirty", target_mat, target_vol)
    
    print(f"Loading DIRTY data from: {dirty_path}")
    dirty_angle, dirty_volt = load_trials_from_folder(dirty_path)

    print(f"Loading CLEAN data from: {clean_path}")
    clean_angle, clean_volt = load_trials_from_folder(clean_path)
    
    out_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # === 1. ANGLE PLOT (1x2 Shared Y) ===
    print("Generating Angle Comparison...")
    # sharey=True ensures they share the axis and hides the inner labels
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, sharey=True)
    fig.suptitle(f"{target_mat} {target_vol} - Angle of Repose Comparison", fontsize=FONT_TITLE, fontweight='bold')
    
    # Global Limits
    all_y = []
    for _, _, y in clean_angle + dirty_angle: all_y.extend(y)
    
    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        buff = (y_max - y_min) * 0.1 if y_max != y_min else 5
        fixed_angle_ylim = (y_min - buff, y_max + buff)
    else:
        fixed_angle_ylim = (0, 90)

    # LEFT: Dirty
    ax_dirty = axes[0]
    ax_dirty.set_title("Dirty", fontsize=FONT_TITLE, fontweight='bold')
    if dirty_angle:
        for (_, x, y) in dirty_angle:
            ax_dirty.scatter(x, y, s=0.5, color='black', alpha=0.05, rasterized=True)
    else: ax_dirty.text(0.5, 0.5, "No Data", ha='center', fontsize=FONT_LABEL)
    
    # RIGHT: Clean
    ax_clean = axes[1]
    ax_clean.set_title("Clean", fontsize=FONT_TITLE, fontweight='bold')
    if clean_angle:
        for (_, x, y) in clean_angle:
            ax_clean.scatter(x, y, s=0.5, color='black', alpha=0.05, rasterized=True)
    else: ax_clean.text(0.5, 0.5, "No Data", ha='center', fontsize=FONT_LABEL)

    # Common Formatting
    for ax in axes:
        ax.set_ylim(fixed_angle_ylim)
        ax.set_xlim(0, HARD_CUTOFF_MIN)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    
    # Single Labels
    axes[0].set_ylabel("Angle (deg)", fontsize=FONT_LABEL)
    fig.supxlabel("Time (min)", fontsize=FONT_LABEL)
    
    save_name = f"Comparison_Angle_{target_mat}_{target_vol}.png"
    plt.savefig(os.path.join(out_dir, save_name), dpi=300)
    plt.close()

    # === 2. VOLTAGE PLOT (1x2 Shared Y) ===
    print("Generating Voltage Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, sharey=True)
    fig.suptitle(f"{target_mat} {target_vol} - Peak Voltage Comparison", fontsize=FONT_TITLE, fontweight='bold')

    # LEFT: Dirty
    ax_dirty = axes[0]
    ax_dirty.set_title("Dirty", fontsize=FONT_TITLE, fontweight='bold')
    if dirty_volt:
        for (i, x, y) in dirty_volt:
            c = TRIAL_COLORS[i % len(TRIAL_COLORS)]
            ax_dirty.scatter(x, y, s=25, color=c, alpha=0.6, label=f'Trial {i+1}')
        ax_dirty.legend(loc='upper right', fontsize=10)
    else: ax_dirty.text(0.5, 0.5, "No Data", ha='center', fontsize=FONT_LABEL)

    # RIGHT: Clean
    ax_clean = axes[1]
    ax_clean.set_title("Clean", fontsize=FONT_TITLE, fontweight='bold')
    if clean_volt:
        for (i, x, y) in clean_volt:
            c = TRIAL_COLORS[i % len(TRIAL_COLORS)]
            ax_clean.scatter(x, y, s=25, color=c, alpha=0.6, label=f'Trial {i+1}')
        ax_clean.legend(loc='upper right', fontsize=10)
    else: ax_clean.text(0.5, 0.5, "No Data", ha='center', fontsize=FONT_LABEL)

    # Common Formatting
    for ax in axes:
        ax.set_ylim(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX)
        ax.set_xlim(0, HARD_CUTOFF_MIN)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    
    axes[0].set_ylabel("Voltage (V)", fontsize=FONT_LABEL)
    fig.supxlabel("Time (min)", fontsize=FONT_LABEL)
    
    save_name = f"Comparison_Voltage_{target_mat}_{target_vol}.png"
    plt.savefig(os.path.join(out_dir, save_name), dpi=300)
    plt.close()
    
    print(f"Done! Saved to {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compare_clean_dirty_v2.py <Path_To_PARENT_Data_Folder> <Volume> <Material>")
        print("Example: python compare_clean_dirty_v2.py F:\\particle-data 500 Acrylic")
    else:
        parent_dir = sys.argv[1]
        vol_arg = sys.argv[2]
        mat_arg = sys.argv[3]
        process_comparison(parent_dir, vol_arg, mat_arg)