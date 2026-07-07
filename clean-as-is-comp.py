import numpy as np
import os
import sys
import pandas as pd
import collections
from scipy.signal import find_peaks

# ================= CONFIGURATION & PATHS =================
# --- SIGNAL PROCESSING ---
VOLTAGE_COL = "CH2_volts"
FLIP_CH2 = True  

# Expected target speeds to snap the raw motor data to
SPEEDS = [1, 6, 11, 16, 21, 26]

# --- AVALANCHE PROCESSING ---
FIXED_PROMINENCE = 1.5
# ========================================================

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline, NO interpolation."""
    if len(chunk) < 1:
        return None
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    return chunk[local_max_idx]

def main(parent_dir):
    # Normalize path to handle both Windows and Unix slashes
    normalized_path = parent_dir.replace('\\', '/').rstrip('/')
    path_parts = normalized_path.split('/')
    
    # --- SMART PATH PARSING ---
    # Convert to lowercase to search for our keywords reliably
    path_parts_lower = [p.lower() for p in path_parts]
    
    state = "Unknown"
    material = "Unknown"
    
    if "dirty" in path_parts_lower:
        idx = path_parts_lower.index("dirty")
        state = path_parts[idx] # Keeps original capitalization
        if idx + 1 < len(path_parts):
            material = path_parts[idx + 1]
    elif "clean" in path_parts_lower:
        idx = path_parts_lower.index("clean")
        state = path_parts[idx] # Keeps original capitalization
        if idx + 1 < len(path_parts):
            material = path_parts[idx + 1]
    else:
        # Fallback just in case
        material = path_parts[-1]
        state = path_parts[-2] if len(path_parts) >= 2 else "Unknown"
    
    dataset_name = f"{state}-{material}"  # e.g., "Dirty-Acetal"
    file_prefix = dataset_name.lower().replace('-', '_') # e.g., "dirty_acetal"
    
    output_dir = os.path.join(normalized_path, f"{dataset_name}_Volume_500_Data")
    
    print(f"=== Starting Volume 500 Data Extraction for: {dataset_name} ===")
    
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]

    all_raw_data = []
    all_peak_data = []

    for root, dirs, files in os.walk(normalized_path):
        if "experiment_log.csv" in files:
            # ONLY target the "500" volume directory
            current_path_parts = root.split(os.sep)
            if "500" not in current_path_parts: 
                continue
                
            csv_path = os.path.join(root, "experiment_log.csv")
            try:
                df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
                df = df.dropna(subset=["motor_speed", "ellipse_angle_deg", "CH2_volts"])
                if df.empty: continue
                
                start_ms = df["ms"].min()
                df["Time (min)"] = (df["ms"] - start_ms) / 60000.0
                df["Motor Speed (RPM)"] = df["motor_speed"].apply(lambda x: min(SPEEDS, key=lambda s: abs(s - x)) if x > 0.5 else 0)
                
                max_val = df["motor_speed"].max()
                max_indices = df.index[df["motor_speed"] == max_val].tolist()
                if not max_indices: continue
                
                mid_idx = max_indices[len(max_indices) // 2]
                df["Direction"] = "Increasing"
                df.loc[mid_idx + 1:, "Direction"] = "Decreasing"
                
                df["Volume"] = "500"
                df["Trial_ID"] = os.path.basename(root) # Saves just the specific trial folder name
                df["Angle (deg)"] = df["ellipse_angle_deg"]
                
                # Flip the voltage!
                multiplier = -1.0 if FLIP_CH2 else 1.0
                df["Voltage (V)"] = pd.to_numeric(df[VOLTAGE_COL], errors='coerce') * multiplier
                df = df.dropna(subset=["Voltage (V)"])

                # --- 1. AVALANCHE PEAK DETECTION ---
                df["Is_Avalanche_Peak"] = False
                peak_ilocs, _ = find_peaks(df["Angle (deg)"].values, height=20, prominence=FIXED_PROMINENCE)
                if len(peak_ilocs) > 0:
                    peak_indices = df.iloc[peak_ilocs].index
                    df.loc[peak_indices, "Is_Avalanche_Peak"] = True
                
                # Save the raw data for Angle and Hysteresis calculations
                all_raw_data.append(df[["Trial_ID", "Time (min)", "Motor Speed (RPM)", "Direction", "Angle (deg)", "Voltage (V)", "Is_Avalanche_Peak"]])

                # --- 2. EXTRACT VOLTAGE PEAKS ---
                times = df["Time (min)"].values
                speeds = df["motor_speed"].values 
                volts = df["Voltage (V)"].values
                
                t_peaks, v_peaks = [], []
                current_idx = 0
                n_samples = len(times)
                
                # Create a rolling memory of the last 5 valid peaks to track the dominant trend
                recent_peaks = collections.deque(maxlen=5)

                while current_idx < n_samples:
                    curr_t = times[current_idx]
                    curr_speed = speeds[current_idx]
                    effective_speed = 1.0 if curr_speed < 0.5 else curr_speed
                    window_min = 1.0 / effective_speed
                    window_end_t = curr_t + window_min

                    end_idx = current_idx
                    while end_idx < n_samples and times[end_idx] < window_end_t:
                        end_idx += 1

                    chunk_v = volts[current_idx:end_idx]
                    
                    if len(chunk_v) > 0:
                        baseline = np.median(chunk_v)
                        c_chunk = chunk_v - baseline
                        
                        # Find both the local max and local min deviation
                        local_max = np.max(c_chunk)
                        local_min = np.min(c_chunk)
                        
                        # The absolute strongest signal in this window is our candidate
                        candidate = local_max if abs(local_max) > abs(local_min) else local_min
                        
                        # --- SMART RC BOUNCE FILTER ---
                        is_valid = True
                        if len(recent_peaks) == 5:
                            recent_median = np.median(recent_peaks)
                            recent_mag = abs(recent_median)
                            
                            if (candidate * recent_median) < 0:
                                if abs(candidate) < (recent_mag * 0.75):
                                    is_valid = False 
                                    
                        if is_valid:
                            local_idx = np.where(c_chunk == candidate)[0][0]
                            t_peaks.append(times[current_idx + local_idx])
                            v_peaks.append(candidate)
                            recent_peaks.append(candidate) 

                    current_idx = end_idx
                    if current_idx >= n_samples: break
                
                peaks_df = pd.DataFrame({
                    "Trial_ID": os.path.basename(root),
                    "Time (min)": t_peaks,
                    "Voltage Peak (V)": v_peaks
                })
                all_peak_data.append(peaks_df)

            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")

    if not all_raw_data:
        print(f"No valid Volume 500 data found for {dataset_name}!")
        return

    # Combine Data
    master_raw_df = pd.concat(all_raw_data, ignore_index=True)
    master_peak_df = pd.concat(all_peak_data, ignore_index=True)

    ts_raw_df = master_raw_df[master_raw_df["Motor Speed (RPM)"] > 0]

    # --- 3. AGGREGATION & HYSTERESIS CREATION ---
    
    # 3a. Standard Trial Aggregation
    trial_df = ts_raw_df.groupby(["Trial_ID", "Motor Speed (RPM)", "Direction"]).agg(
        Trial_Angle_Mean=("Angle (deg)", "mean"),
        Trial_Voltage_Std=("Voltage (V)", "std") 
    ).reset_index()

    # 3b. Avalanche Trial Aggregation (Filter for peaks only)
    avalanche_raw = ts_raw_df[ts_raw_df["Is_Avalanche_Peak"]]
    avalanche_trial = avalanche_raw.groupby(["Trial_ID", "Motor Speed (RPM)", "Direction"]).agg(
        Trial_Avalanche_Mean=("Angle (deg)", "mean")
    ).reset_index()

    # Merge Avalanche data into Trial data
    trial_df = pd.merge(trial_df, avalanche_trial, on=["Trial_ID", "Motor Speed (RPM)", "Direction"], how="left")

    # 3c. Final Summary Aggregation
    hyst_df = trial_df.groupby(["Motor Speed (RPM)", "Direction"]).agg(
        Angle_Mean=("Trial_Angle_Mean", "mean"),
        Angle_Std=("Trial_Angle_Mean", "std"),
        Avalanche_Mean=("Trial_Avalanche_Mean", "mean"),
        Avalanche_Std=("Trial_Avalanche_Mean", "std"),
        Voltage_Std_Mean=("Trial_Voltage_Std", "mean"), 
        Voltage_Std_Err=("Trial_Voltage_Std", "std")    
    ).reset_index()

    os.makedirs(output_dir, exist_ok=True)
    
    # File 1: Angle Time Series (Raw data during active speeds)
    angle_ts_path = os.path.join(output_dir, f"{file_prefix}_angle_timeseries_500.csv")
    ts_raw_df[["Trial_ID", "Time (min)", "Angle (deg)"]].to_csv(angle_ts_path, index=False)
    
    # File 2: Filtered Voltage Peaks Time Series
    voltage_peaks_path = os.path.join(output_dir, f"{file_prefix}_voltage_peaks_timeseries_500.csv")
    master_peak_df.to_csv(voltage_peaks_path, index=False)
    
    # File 3: Unified Hysteresis Profile
    hyst_summary_path = os.path.join(output_dir, f"{file_prefix}_hysteresis_summary_500.csv")
    hyst_df.to_csv(hyst_summary_path, index=False)

    print(f"  -> Data saved successfully to: {output_dir}")
    print(f"=== Process Complete for {dataset_name} ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_extractor.py <path_to_data_folder>")
    else:
        main(sys.argv[1])