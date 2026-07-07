import os
import sys
import numpy as np
import pandas as pd

# ================= CONFIGURATION & PATHS =================
MAIN_PARENT_DIR = r"E:\particle-data" 
OUTPUT_FOLDER_NAME = "Grand_Voltage_Comparisons_Data"

TIME_CUTOFF_MIN = 60.0  
RESAMPLE_INTERVAL_MIN = 0.5 

ATTENUATION_FACTORS = {
    'Acetal': -30.63, 'Nylon': -22.33 , 'Teflon': -40.98, 'Acrylic': 0.0 
}
# ========================================================

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline."""
    if len(chunk) < 1:
        return None
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    return chunk[local_max_idx]

def process_grand_voltage_comparison_data(base_dir, material):
    print(f"\n--- Starting Grand Voltage Data Extraction for {material} ---")
    
    num_samples = int(TIME_CUTOFF_MIN / RESAMPLE_INTERVAL_MIN) + 1
    common_time_grid = np.linspace(0.0, TIME_CUTOFF_MIN, num_samples)
    
    # Dictionaries to hold both the averages and the errors across trials
    ensemble_peaks = {}
    ensemble_peaks_err = {}
    
    ensemble_stds = {}
    ensemble_stds_err = {}

    condition_roots = {
        'Dirty': os.path.join(base_dir, 'Dirty', material),
        'Clean': os.path.join(base_dir, 'Clean', material),
        'Faceplates': os.path.join(base_dir, 'diff_face_plates', f"{material}-In-{material}")
    }

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    for condition, search_dir in condition_roots.items():
        if not os.path.exists(search_dir): continue

        trial_csvs = []
        for root, dirs, files in os.walk(search_dir):
            if '500' in root.split(os.sep) and 'experiment_log.csv' in files:
                trial_csvs.append(os.path.join(root, 'experiment_log.csv'))

        if not trial_csvs: continue
        
        all_resampled_peaks = []
        all_resampled_stds = []
        
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
                v_stds_combined = []
                
                current_idx = 0
                n_samples_len = len(times)

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
                    
                    val_ch2_peak = get_peak(chunk_ch2)
                    val_ch3_peak = get_peak(chunk_ch3)
                    
                    if val_ch2_peak is not None and val_ch3_peak is not None:
                        # Calculate Averages for both metrics within this chunk
                        avg_peak = (val_ch2_peak + val_ch3_peak) / 2.0
                        avg_std = (np.std(chunk_ch2) + np.std(chunk_ch3)) / 2.0
                        
                        t_peaks.append(curr_t)
                        v_peaks_combined.append(avg_peak)
                        v_stds_combined.append(avg_std)

                    current_idx = end_idx
                    if current_idx >= n_samples_len: break

                if len(t_peaks) > 1:
                    resampled_v = np.interp(common_time_grid, t_peaks, v_peaks_combined)
                    resampled_std = np.interp(common_time_grid, t_peaks, v_stds_combined)
                    
                    all_resampled_peaks.append(resampled_v)
                    all_resampled_stds.append(resampled_std)

            except Exception as e:
                print(f"  Error reading {os.path.basename(os.path.dirname(csv_path))}: {e}")

        # --- CALCULATE ERROR ACROSS TRIALS HERE ---
        if all_resampled_peaks:
            peak_arr = np.array(all_resampled_peaks)
            std_arr = np.array(all_resampled_stds)
            
            # Mean across trials
            ensemble_peaks[condition] = np.nanmean(peak_arr, axis=0)
            ensemble_stds[condition] = np.nanmean(std_arr, axis=0)
            
            # Standard Deviation across trials (used for error bars)
            # Note: Change np.nanstd to (np.nanstd / np.sqrt(peak_arr.shape[0])) if you want SEM instead
            ensemble_peaks_err[condition] = np.nanstd(peak_arr, axis=0)
            ensemble_stds_err[condition] = np.nanstd(std_arr, axis=0)

    # Apply Attenuation Multipliers to Clean Data (And the errors!)
    if 'Clean' in ensemble_peaks:
        atten_pct = ATTENUATION_FACTORS.get(material, 0.0)
        multiplier = 1.0 + (atten_pct / 100.0)
        
        ensemble_peaks['Clean_Atten'] = ensemble_peaks['Clean'] * multiplier
        ensemble_peaks_err['Clean_Atten'] = ensemble_peaks_err['Clean'] * abs(multiplier)
        
        ensemble_stds['Clean_Atten'] = ensemble_stds['Clean'] * multiplier
        ensemble_stds_err['Clean_Atten'] = ensemble_stds_err['Clean'] * abs(multiplier)

    if not ensemble_peaks: 
        print(f"No valid data processed for {material}.")
        return

    out_dir = os.path.join(base_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # --- CSV EXPORT LOGIC FOR BOTH METRICS AND ERRORS ---
    def save_csv(data_dict, err_dict, file_prefix):
        export_data = {'Time (min)': common_time_grid}
        for condition, y_data in data_dict.items():
            export_data[condition] = y_data
            
            # If we calculated error for this condition, save it as an adjacent column
            if condition in err_dict:
                export_data[f"{condition}_Std"] = err_dict[condition]
                
        export_df = pd.DataFrame(export_data)
        csv_filename = f"{file_prefix}_{material}.csv"
        output_path = os.path.join(out_dir, csv_filename)
        export_df.to_csv(output_path, index=False)
        print(f"Exported: {output_path}")

    # Save both data streams passing in their respective error dictionaries
    save_csv(ensemble_peaks, ensemble_peaks_err, "MaxPeak_Voltage_Data")
    save_csv(ensemble_stds, ensemble_stds_err, "StdDev_Voltage_Data")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python grand_voltage_extractor.py <MaterialFolderName>")
    process_grand_voltage_comparison_data(MAIN_PARENT_DIR, sys.argv[1])