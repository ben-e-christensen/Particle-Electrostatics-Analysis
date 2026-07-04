import os
import pandas as pd
import numpy as np

# ================= CONFIGURATION =================
BASE_DIR = r"E:\particle-data\Dirty"
OUTPUT_FILENAME = "Volume_Angle_and_Time_Split_Voltage_Summary.csv"

# Time windows (in minutes) to define "beginning" and "end"
TIME_WINDOW_MINS = 5.0  

COLS = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
        "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
        "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
        "ch2_flag", "ch3_flag"]
# =================================================

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline."""
    if len(chunk) < 1:
        return None
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    return chunk[local_max_idx]

def extract_top_10_percent_avg(times, speeds, volts):
    """Chunks the data by rotation, finds peaks, and averages the top 10%."""
    if len(times) == 0:
        return np.nan
        
    current_idx = 0
    n_samples = len(times)
    peaks = []

    while current_idx < n_samples:
        curr_t = times[current_idx]
        curr_speed = speeds[current_idx]
        effective_speed = 1.0 if curr_speed < 0.5 else curr_speed
        window_min = 1.0 / effective_speed
        window_end_t = curr_t + window_min

        end_idx = current_idx
        while end_idx < n_samples and times[end_idx] < window_end_t:
            end_idx += 1

        chunk = volts[current_idx:end_idx]
        val = get_peak(chunk)
        if val is not None:
            peaks.append(abs(val)) 

        current_idx = end_idx
        
    if not peaks: return np.nan
    
    peaks.sort(reverse=True)
    top_10_count = max(1, int(len(peaks) * 0.10))
    top_10_peaks = peaks[:top_10_count]
    
    return np.mean(top_10_peaks)

def process_data():
    print(f"--- Starting Time-Split Batch Analysis in: {BASE_DIR} ---")
    all_trial_data = []

    for material in os.listdir(BASE_DIR):
        material_path = os.path.join(BASE_DIR, material)
        if not os.path.isdir(material_path) or material.startswith('.') or material.endswith('.csv'):
            continue
            
        for volume in os.listdir(material_path):
            volume_path = os.path.join(material_path, volume)
            if not os.path.isdir(volume_path):
                continue
                
            for trial in os.listdir(volume_path):
                trial_path = os.path.join(volume_path, trial)
                if not os.path.isdir(trial_path):
                    continue
                    
                csv_path = os.path.join(trial_path, "experiment_log.csv")
                if not os.path.exists(csv_path):
                    continue
                    
                print(f"Processing: {material} | {volume} | {trial}")
                
                try:
                    df = pd.read_csv(csv_path, names=COLS, header=0, on_bad_lines="skip", 
                                     dtype={"ms": "float32", "CH2_volts": "float32", "CH3_volts": "float32", "ellipse_angle_deg": "float32", "motor_speed": "float32"})
                    
                    if df.empty: continue

                    start_ms = df["ms"].min() 
                    df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
                    max_time = df["rel_time_min"].max()
                    
                    # --- METRIC 1: ANGLE OF REPOSE (1 RPM) ---
                    df_1rpm = df[(df["motor_speed"] >= 0.95) & (df["motor_speed"] <= 1.05)].dropna(subset=["ellipse_angle_deg"])
                    
                    start_angle = df_1rpm.loc[df_1rpm["rel_time_min"] <= TIME_WINDOW_MINS, "ellipse_angle_deg"].mean()
                    end_angle = df_1rpm.loc[df_1rpm["rel_time_min"] >= (max_time - TIME_WINDOW_MINS), "ellipse_angle_deg"].mean()

                    # --- METRIC 2: TIME-SPLIT PEAK VOLTAGES (Top 10%) ---
                    df_volt = df.dropna(subset=["motor_speed", "CH2_volts"])
                    
                    # Slice the voltage dataframe into start and end blocks
                    df_volt_start = df_volt[df_volt["rel_time_min"] <= TIME_WINDOW_MINS]
                    df_volt_end = df_volt[df_volt["rel_time_min"] >= (max_time - TIME_WINDOW_MINS)]
                    
                    # Calculate voltages independently
                    start_volt = extract_top_10_percent_avg(df_volt_start["rel_time_min"].values, df_volt_start["motor_speed"].values, df_volt_start["CH2_volts"].values)
                    end_volt = extract_top_10_percent_avg(df_volt_end["rel_time_min"].values, df_volt_end["motor_speed"].values, df_volt_end["CH2_volts"].values)

                    all_trial_data.append({
                        "Material": material,
                        "Volume": int(volume) if volume.isdigit() else volume,
                        "Trial": trial,
                        "Start_Angle_1RPM": start_angle,
                        "End_Angle_1RPM": end_angle,
                        "Start_CH2_Volt": start_volt,
                        "End_CH2_Volt": end_volt
                    })

                except Exception as e:
                    print(f"  Error processing {csv_path}: {e}")

    # ================= AGGREGATION & EXPORT =================
    if all_trial_data:
        results_df = pd.DataFrame(all_trial_data)
        
        ensemble_summary = results_df.groupby(["Material", "Volume"]).agg({
            "Start_Angle_1RPM": "mean",
            "End_Angle_1RPM": "mean",
            "Start_CH2_Volt": "mean",
            "End_CH2_Volt": "mean"
        }).reset_index()
        
        raw_out = os.path.join(BASE_DIR, "Trial_Raw_TimeSplit_Metrics.csv")
        ens_out = os.path.join(BASE_DIR, OUTPUT_FILENAME)
        
        results_df.to_csv(raw_out, index=False)
        ensemble_summary.to_csv(ens_out, index=False)
        
        print(f"\n--- DONE ---")
        print(f"Ensemble summary saved to: {ens_out}")

if __name__ == "__main__":
    process_data()