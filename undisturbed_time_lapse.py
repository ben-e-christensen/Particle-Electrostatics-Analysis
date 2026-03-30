import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "LongTerm_Ensemble_Results"
VOLTAGE_COL_CH2 = "CH2_volts"
VOLTAGE_COL_CH3 = "CH3_volts"

# --- TOGGLES ---
SHOW_LEGEND = False  
SHOW_TITLES = False  

# --- ACADEMIC PLOT SETTINGS ---
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
FONT_TITLE = 20

# Fixed Y-Axis for Voltage Plots
# You can tighten these limits now that the outliers are purged!
FIXED_VOLT_YMIN = -0.2
FIXED_VOLT_YMAX = 0.1

# Color Palette for distinct DAYS (Face Color, Edge Color)
DAY_COLORS = [
    ('#c5a3d4', '#9944ff'), ('#81aad4', '#083b70'), ('#a3d4a3', '#087008'),
    ('#d4a3a3', '#700808'), ('#d4c5a3', '#705c08'), ('#a3d4d4', '#087070'),
    ('#d4a3c5', '#70085c')
]
# =================================================

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline, returning RELATIVE amplitude."""
    if len(chunk) < 1:
        return None

    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    
    return centered_chunk[local_max_idx]

def apply_academic_axes(ax, xlabel, ylabel):
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.tick_params('both', length=7, width=1, which='major')
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True, alpha=0.3)

def process_continuous_graphs(parent_dir):
    print(f"--- Starting Long-Term Analysis for: {parent_dir} ---")

    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    angle_data_grouped = []
    ch2_voltage_data_grouped = []
    ch3_voltage_data_grouped = []

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    cumulative_ms = 0.0
    max_time_hours = 0.0

    print("Processing daily folders...")

    for i in range(1, 8):
        day_folder = os.path.join(parent_dir, f"day{i}")
        
        if not os.path.exists(day_folder):
            print(f"  -> Skipping Day {i} (Folder not found)")
            continue
            
        print(f"  Reading Day {i}...")
        
        csv_files = [f for f in os.listdir(day_folder) if f.endswith('.csv')]
        csv_files.sort()
        
        if not csv_files:
            continue

        for csv_file in csv_files:
            csv_path = os.path.join(day_folder, csv_file)
            try:
                df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", 
                                 dtype={"ms": "float32", "CH2_volts": "float32", "CH3_volts": "float32", "ellipse_angle_deg": "float32"})
                
                if df.empty: continue

                # --- PRE-PROCESSING (Stitching Time) ---
                start_ms = df["ms"].min() 
                df["Continuous_ms"] = (df["ms"] - start_ms) + cumulative_ms
                cumulative_ms = df["Continuous_ms"].max() + 10.0 
                df["rel_time_hours"] = df["Continuous_ms"] / 3600000.0
                max_time_hours = max(max_time_hours, df["rel_time_hours"].max())
                df.sort_values(by="Continuous_ms", inplace=True)

                # --- COLLECTION 1: ANGLE DATA ---
                df_angle = df.dropna(subset=["ellipse_angle_deg"])
                if len(df_angle) > 100000: 
                    step = len(df_angle) // 100000
                    df_angle = df_angle.iloc[::step]

                if not df_angle.empty:
                    angle_data_grouped.append((i-1, df_angle["rel_time_hours"].values, df_angle["ellipse_angle_deg"].values))

                # --- COLLECTION 2: VOLTAGE PEAKS ---
                df_volt = df.dropna(subset=["motor_speed", VOLTAGE_COL_CH2, VOLTAGE_COL_CH3])
                df_volt.loc[:, VOLTAGE_COL_CH2] = pd.to_numeric(df_volt[VOLTAGE_COL_CH2], errors='coerce') * -1.0
                df_volt.loc[:, VOLTAGE_COL_CH3] = pd.to_numeric(df_volt[VOLTAGE_COL_CH3], errors='coerce')
                df_volt = df_volt.dropna(subset=[VOLTAGE_COL_CH2, VOLTAGE_COL_CH3])

                if not df_volt.empty:
                    # =========================================================
                    # THE BLACKOUT FILTER: Surgically removing chassis arc events
                    # =========================================================
                    # 1. Fast 10-second rolling median to catch the exact moment the floor drops
                    fast_baseline = df_volt[VOLTAGE_COL_CH2].rolling(window=1000, min_periods=1, center=True).median()
                    
                    # 2. Flag moments where the baseline drops past 0.04V
                    bad_times = df_volt[fast_baseline.abs() > 0.04]["rel_time_hours"].values
                    
                    valid_mask = np.ones(len(df_volt), dtype=bool)
                    times = df_volt["rel_time_hours"].values
                    
                    # 3. Define the Blackout Window
                    blackout_before_hr = 10.0 / 3600.0  # 10 seconds before
                    blackout_after_hr = 180.0 / 3600.0  # 3 minutes after
                    
                    last_trigger_hr = -999
                    blackout_count = 0
                    
                    # 4. Apply the mask
                    for t in bad_times:
                        if (t - last_trigger_hr) > (300.0 / 3600.0): # 5 min cooldown to group one continuous event
                            bad_idx = (times >= (t - blackout_before_hr)) & (times <= (t + blackout_after_hr))
                            valid_mask[bad_idx] = False
                            last_trigger_hr = t
                            blackout_count += 1
                            
                    df_volt = df_volt[valid_mask]
                    if blackout_count > 0:
                        print(f"    -> [!] Surgically purged {blackout_count} chassis arc events (-10s to +180s buffer)")
                    # =========================================================

                    # Now proceed with normal peak extraction on the clean data
                    times = df_volt["rel_time_hours"].values
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

                        window_hours = (1.0 / effective_speed) / 60.0
                        window_end_t = curr_t + window_hours

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
                        ch2_voltage_data_grouped.append((i-1, np.array(t_peaks_ch2), np.array(v_peaks_ch2)))
                    if t_peaks_ch3:
                        ch3_voltage_data_grouped.append((i-1, np.array(t_peaks_ch3), np.array(v_peaks_ch3)))

            except Exception as e:
                print(f"Skipping {csv_file}: {e}")

    x_limit = max_time_hours if max_time_hours > 0 else 120.0

    # ================= PLOT 1: ANGLE OF REPOSE =================
    print("Generating Continuous Angle Plot...")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    if angle_data_grouped:
        for (day_idx, x, y) in angle_data_grouped:
            ax.scatter(x, y, s=0.5, color='black', alpha=0.05, rasterized=True)
            
        all_y = np.concatenate([y for _, _, y in angle_data_grouped])
        y_min, y_max = all_y.min(), all_y.max()
        buff = (y_max - y_min) * 0.1 if y_max != y_min else 5
        ax.set_ylim(y_min - buff, y_max + buff)
    else:
        ax.text(0.5, 0.5, "No Angle Data", ha='center', transform=ax.transAxes)

    if SHOW_TITLES:
        ax.set_title("Long-Term Continuous Bed Angle", fontsize=FONT_TITLE, fontweight='bold')

    apply_academic_axes(ax, "Cumulative Time (Hours)", "Angle (deg)")
    ax.set_xlim(0, x_limit)
    plt.tight_layout()
    
    angle_path = os.path.join(output_dir, "Global_Angle_Continuous.png")
    plt.savefig(angle_path, dpi=300)
    plt.close()

    # ================= PLOT 2: PEAK VOLTAGE (CH2 Flipped) =================
    print("Generating CH2 Flipped Plot...")
    fig, ax = plt.subplots(figsize=(7, 5))

    if ch2_voltage_data_grouped:
        added_to_legend = set()
        for (day_idx, x, y) in ch2_voltage_data_grouped:
            face_c, edge_c = DAY_COLORS[day_idx % len(DAY_COLORS)]
            label = f'Day {day_idx+1}' if day_idx not in added_to_legend else ""
            added_to_legend.add(day_idx)
            
            ax.scatter(x, y, s=10, facecolors=face_c, edgecolors=edge_c, 
                       linewidth=0.5, alpha=0.9, label=label)
        
        ax.set_ylim(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX)
    else:
        ax.text(0.5, 0.5, "No CH2 Data", ha='center', transform=ax.transAxes)

    if SHOW_TITLES:
        ax.set_title("CH2: Continuous Peak Charge (Flipped)", fontsize=FONT_TITLE, fontweight='bold')

    apply_academic_axes(ax, "Time (Hours)", "Voltage (V)")
    ax.set_xlim(0, x_limit)
    
    if SHOW_LEGEND:
        ax.legend(fontsize=12, loc='lower left', framealpha=0.9)
        
    plt.tight_layout()
    
    ch2_path = os.path.join(output_dir, "Global_Voltage_CH2_Flipped.png")
    plt.savefig(ch2_path, dpi=300)
    plt.close()

    # ================= PLOT 3: PEAK VOLTAGE (CH3 Standard) =================
    print("Generating CH3 Plot...")
    fig, ax = plt.subplots(figsize=(7, 5))

    if ch3_voltage_data_grouped:
        added_to_legend = set()
        for (day_idx, x, y) in ch3_voltage_data_grouped:
            face_c, edge_c = DAY_COLORS[day_idx % len(DAY_COLORS)]
            label = f'Day {day_idx+1}' if day_idx not in added_to_legend else ""
            added_to_legend.add(day_idx)
            
            ax.scatter(x, y, s=10, facecolors=face_c, edgecolors=edge_c, 
                       linewidth=0.5, alpha=0.9, label=label)
        
        ax.set_ylim(FIXED_VOLT_YMIN, FIXED_VOLT_YMAX)
    else:
        ax.text(0.5, 0.5, "No CH3 Data", ha='center', transform=ax.transAxes)

    if SHOW_TITLES:
        ax.set_title("CH3: Continuous Peak Charge", fontsize=FONT_TITLE, fontweight='bold')

    apply_academic_axes(ax, "Time (Hours)", "Voltage (V)")
    ax.set_xlim(0, x_limit)
    
    if SHOW_LEGEND:
        ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
        
    plt.tight_layout()
    
    ch3_path = os.path.join(output_dir, "Global_Voltage_CH3.png")
    plt.savefig(ch3_path, dpi=300,bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete. Check the '{OUTPUT_FOLDER_NAME}' directory.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python continuous_ensemble.py <path_to_PARENT_folder>")
    else:
        process_continuous_graphs(sys.argv[1])