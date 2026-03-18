import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ================= CONFIGURATION & PATHS =================
MAIN_PARENT_DIR = r"D:\particle-data" 
OUTPUT_FOLDER_NAME = "Grand_Voltage_Comparisons"

TIME_CUTOFF_MIN = 60.0  
RESAMPLE_INTERVAL_MIN = 0.5 

ATTENUATION_FACTORS = {
    'Acetal': -41.35, 
    'Nylon': -18.05,
    'Teflon': -25.56,
    'Acrylic': 0.0 
}

LINE_STYLES = {
    'Dirty':        {'color': '#a6611a', 'marker': 'o', 'ms': 8, 'label': 'Dirty'},
    'Clean':        {'color': '#404040', 'marker': 's', 'ms': 8, 'label': 'Clean (Standard)'},
    'Clean_Atten':  {'color': '#018571', 'marker': '^', 'ms': 8, 'ls': '--', 'label': 'Clean (Attenuated)'},
    'Faceplates':   {'color': '#d7191c', 'marker': 'D', 'ms': 8, 'label': 'Diff Faceplates'}
}
# ========================================================

def get_signed_p2p(chunk):
    """
    Calculates the Peak-to-Peak amplitude within a chunk to ignore RC drift,
    and intelligently restores the +/- polarity based on the wave's direction.
    """
    if len(chunk) < 2:
        return None
        
    # True amplitude ignoring baseline wander
    p2p_amplitude = np.max(chunk) - np.min(chunk)
    
    # Find the local baseline
    baseline = np.median(chunk)
    
    # Determine Polarity: Which side of the wave deviates further from the median?
    max_dev = np.max(chunk) - baseline
    min_dev = np.min(chunk) - baseline
    
    # If the downward spike is larger, it's a negative charging event
    if abs(min_dev) > abs(max_dev):
        return -p2p_amplitude
    else:
        return p2p_amplitude

def process_grand_voltage_comparison(base_dir, material):
    print(f"\n--- Starting Grand Voltage Analysis (Signed P2P Averaged) for {material} ---")
    
    num_samples = int(TIME_CUTOFF_MIN / RESAMPLE_INTERVAL_MIN) + 1
    common_time_grid = np.linspace(0.0, TIME_CUTOFF_MIN, num_samples)

    ensemble_averaged_data = {}

    condition_roots = {
        'Dirty': os.path.join(base_dir, 'Dirty', material),
        'Clean': os.path.join(base_dir, 'Clean', material),
        'Faceplates': os.path.join(base_dir, 'diff_face_plates', f"{material}-In-{material}")
    }

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    # --- 1. DISCOVER AND PROCESS TRIALS ---
    for condition, search_dir in condition_roots.items():
        if not os.path.exists(search_dir):
            print(f"Skipping {condition}: Directory not found ({search_dir})")
            continue

        trial_csvs = []
        for root, dirs, files in os.walk(search_dir):
            if '500' in root.split(os.sep) and 'experiment_log.csv' in files:
                trial_csvs.append(os.path.join(root, 'experiment_log.csv'))

        if not trial_csvs:
            print(f"Skipping {condition}: No trials found in 500 folders.")
            continue
            
        print(f" Processing {condition}: Found {len(trial_csvs)} trials.")
        all_resampled_peaks = []
        
        for csv_path in trial_csvs:
            try:
                df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", 
                                 dtype={"ms": "float32", "CH2_volts": "float32", "CH3_volts": "float32"})
                
                if df.empty: continue

                df = df.dropna(subset=["motor_speed", "CH2_volts", "CH3_volts"])
                if df.empty: continue

                start_ms = df["ms"].min()
                df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
                df = df[df["rel_time_min"] <= TIME_CUTOFF_MIN]
                df.sort_values(by="ms", inplace=True)

                # Flip CH2 ONLY to align with CH3
                df.loc[:, "CH2_volts"] = pd.to_numeric(df["CH2_volts"], errors='coerce') * -1.0
                df.loc[:, "CH3_volts"] = pd.to_numeric(df["CH3_volts"], errors='coerce')
                
                df = df.dropna(subset=["CH2_volts", "CH3_volts"])
                if df.empty: continue

                times = df["rel_time_min"].values
                speeds = df["motor_speed"].values
                volts_ch2 = df["CH2_volts"].values
                volts_ch3 = df["CH3_volts"].values

                t_peaks = []
                v_peaks_combined = []
                current_idx = 0
                n_samples_len = len(times)

                # Motor-Speed Windowed Extraction
                while current_idx < n_samples_len:
                    curr_t = times[current_idx]
                    curr_speed = speeds[current_idx]
                    effective_speed = 1.0 if curr_speed < 0.5 else curr_speed

                    window_min = 1.0 / effective_speed
                    window_end_t = curr_t + window_min

                    end_idx = current_idx
                    while end_idx < n_samples_len and times[end_idx] < window_end_t:
                        end_idx += 1

                    chunk_ch2 = volts_ch2[current_idx:end_idx]
                    chunk_ch3 = volts_ch3[current_idx:end_idx]
                    
                    # Extract SIGNED Peak-to-Peak
                    val_ch2 = get_signed_p2p(chunk_ch2)
                    val_ch3 = get_signed_p2p(chunk_ch3)
                    
                    if val_ch2 is not None and val_ch3 is not None:
                        avg_peak = (val_ch2 + val_ch3) / 2.0
                        t_peaks.append(curr_t)
                        v_peaks_combined.append(avg_peak)

                    current_idx = end_idx
                    if current_idx >= n_samples_len: break

                if len(t_peaks) > 1:
                    resampled_v = np.interp(common_time_grid, t_peaks, v_peaks_combined)
                    all_resampled_peaks.append(resampled_v)

            except Exception as e:
                print(f"  Error reading {os.path.basename(os.path.dirname(csv_path))}: {e}")

        if all_resampled_peaks:
            ensemble_avg = np.nanmean(np.array(all_resampled_peaks), axis=0)
            ensemble_averaged_data[condition] = ensemble_avg


    # --- 2. GENERATE ATTENUATED CLEAN LINE ---
    if 'Clean' in ensemble_averaged_data:
        atten_pct = ATTENUATION_FACTORS.get(material, 0.0)
        multiplier = 1.0 + (atten_pct / 100.0)
        print(f" Applying {atten_pct}% attenuation multiplier to Clean data: {multiplier:.4f}")
        # Because we preserved polarity, a negative multiplier works perfectly on negative voltages!
        ensemble_averaged_data['Clean_Atten'] = ensemble_averaged_data['Clean'] * multiplier


    # --- 3. GENERATE PLOT ---
    if not ensemble_averaged_data:
        print("Error: No data successfully extracted to plot.")
        return

    print(" Generating Ensemble Voltage Figure...")
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 7))

    for condition, style in LINE_STYLES.items():
        if condition in ensemble_averaged_data:
            y_data = ensemble_averaged_data[condition]
            
            ax.plot(common_time_grid, y_data, lw=2, ls=style.get('ls', '-'), 
                     color=style['color'], label=style['label'])
            
            mark_every = int(5.0 / RESAMPLE_INTERVAL_MIN)
            ax.scatter(common_time_grid[::mark_every], y_data[::mark_every], 
                        marker=style['marker'], color=style['color'], 
                        edgecolors='k', s=80, zorder=5)

    ax.set_title(f"Averaged Signed Peak-to-Peak Voltage: {material}", fontsize=20, fontweight='bold', family='sans-serif')
    ax.set_xlabel("Time (min)", fontsize=18)
    ax.set_ylabel("Peak Voltage (V)", fontsize=18)
    
    ax.set_xlim(0, TIME_CUTOFF_MIN)
    ax.grid(True, which='both', alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    
    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    out_dir = os.path.join(base_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    fig_name = f"Signed_P2P_Voltage_{material}"
    png_path = os.path.join(out_dir, f"{fig_name}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    print(f" Saved successfully to:\n {png_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grand_voltage_compare.py <MaterialFolderName>")
        print("Example: python grand_voltage_compare.py Teflon")
    else:
        material_arg = sys.argv[1]
        process_grand_voltage_comparison(MAIN_PARENT_DIR, material_arg)