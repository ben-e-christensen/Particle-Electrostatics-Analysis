import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import sys
import os

# ================= CONFIGURATION =================
TIME_POINTS = ["t0", "day-1", "day-7"]

# Analysis Settings
SAMPLE_RATE = 100 
SPEED_ROUNDING_PRECISION = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5
HARD_CUTOFF_MIN = 60.0  

# --- SIGNAL PROCESSING ---
VOLTAGE_COL = "CH2_volts"
SMOOTHING_WINDOW = 7 
FLIP_CH2 = True  
# =================================================

def get_material_name(csv_path, time_point_root):
    trial_dir = os.path.dirname(csv_path)
    parent_dir = os.path.dirname(trial_dir)
    parent_name = os.path.basename(parent_dir)
    
    if parent_name == os.path.basename(time_point_root) or parent_name in TIME_POINTS:
        trial_name = os.path.basename(trial_dir)
        candidate = trial_name.split("-")[0].split("_")[0]
        return candidate.lower()
    return parent_name.lower()

def get_peak(chunk):
    """Finds the absolute max peak relative to the local baseline, NO interpolation."""
    if len(chunk) < 1:
        return None
    baseline = np.median(chunk)
    centered_chunk = chunk - baseline
    local_max_idx = np.argmax(np.abs(centered_chunk))
    return chunk[local_max_idx]

def load_data(root_path):
    print(f"--- SCANNING ROOT: {root_path} ---")
    minute_data_list = []
    raw_trials_list = []

    for tp in TIME_POINTS:
        tp_path = os.path.join(root_path, tp)
        if not os.path.exists(tp_path):
            found = False
            for actual in os.listdir(root_path):
                if actual.lower() == tp.lower():
                    tp_path = os.path.join(root_path, actual)
                    found = True; break
            if not found:
                continue

        csv_files = []
        for root, dirs, files in os.walk(tp_path):
            if "experiment_log.csv" in files:
                csv_files.append(os.path.join(root, "experiment_log.csv"))
        csv_files.sort()

        for trial_idx, input_csv in enumerate(csv_files):
            try:
                if "images" in input_csv.split(os.sep): continue
                material_name = get_material_name(input_csv, tp_path)
                trial_name_folder = os.path.basename(os.path.dirname(input_csv))

                cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
                        "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
                        "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
                        "ch2_flag", "ch3_flag"]
                df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
                if df.empty: continue

                start_ms = df["ms"].min()
                df["rel_time_min"] = (df["ms"] - start_ms) / 60000.0
                df = df[df["rel_time_min"] <= HARD_CUTOFF_MIN]
                if df.empty: continue

                max_val = df["motor_speed"].max()
                max_indices = df.index[df["motor_speed"] == max_val].tolist()
                
                if max_indices:
                    mid_idx = max_indices[len(max_indices)//2]
                    df["direction"] = "Increasing"
                    df.loc[mid_idx+1:, "direction"] = "Decreasing"
                    
                    t0 = df["timestamp"].iloc[0]
                    df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int) + 1
                    
                    p_idxs, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
                    angle_df = df.iloc[p_idxs].copy()
                    
                    charge_agg = df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                        voltage_std=("CH2_volts", "std"), count=("CH2_volts", "count")).reset_index()
                    angle_agg = angle_df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                        angle_mean=("ellipse_angle_deg", "mean")).reset_index()
                    
                    merged = pd.merge(charge_agg, angle_agg, on=["minute_bin", "direction", "motor_speed"])
                    merged = merged[merged["count"] > (SAMPLE_RATE * MIN_SECONDS_PER_BIN)]
                    
                    if not merged.empty:
                        merged["material"] = material_name
                        merged["time_point"] = tp
                        merged["trial_id"] = trial_name_folder 
                        merged["grouped_speed"] = merged["motor_speed"].round(SPEED_ROUNDING_PRECISION).astype(int)
                        minute_data_list.append(merged)

                raw_angle = df.dropna(subset=["ellipse_angle_deg"])
                angle_pts = None
                if not raw_angle.empty:
                    angle_pts = (raw_angle["rel_time_min"].values, raw_angle["ellipse_angle_deg"].values)

                volt_pts = None
                df_v = df.dropna(subset=["motor_speed", VOLTAGE_COL]).copy()
                
                multiplier = -1.0 if FLIP_CH2 else 1.0
                df_v[VOLTAGE_COL] = pd.to_numeric(df_v[VOLTAGE_COL], errors='coerce') * multiplier
                df_v = df_v.dropna(subset=[VOLTAGE_COL])
                
                if not df_v.empty:
                    if SMOOTHING_WINDOW > 1:
                        df_v[VOLTAGE_COL] = df_v[VOLTAGE_COL].rolling(window=SMOOTHING_WINDOW, center=True).mean()
                        df_v = df_v.dropna(subset=[VOLTAGE_COL])
                    
                    times_full = df_v["rel_time_min"].values
                    speeds_full = df_v["motor_speed"].values
                    volts_full = df_v[VOLTAGE_COL].values
                    
                    t_peaks, v_peaks = [], []
                    current_idx = 0
                    n_samples = len(times_full)

                    while current_idx < n_samples:
                        curr_t = times_full[current_idx]
                        curr_speed = speeds_full[current_idx]
                        effective_speed = 1.0 if curr_speed < 0.5 else curr_speed
                        window_min = 1.0 / effective_speed
                        window_end_t = curr_t + window_min

                        end_idx = current_idx
                        while end_idx < n_samples and times_full[end_idx] < window_end_t:
                            end_idx += 1

                        chunk_v = volts_full[current_idx:end_idx]
                        val = get_peak(chunk_v)
                        if val is not None:
                            t_peaks.append(curr_t)
                            v_peaks.append(val)

                        current_idx = end_idx
                        if current_idx >= n_samples: break
                    
                    if len(t_peaks) > 0:
                        volt_pts = (np.array(t_peaks), np.array(v_peaks))

                raw_trials_list.append({
                    "material": material_name,
                    "time_point": tp,
                    "trial_idx": trial_idx, 
                    "trial_id": trial_name_folder,
                    "angle_data": angle_pts,
                    "voltage_data": volt_pts
                })
                print(f"    Processed: {material_name} ({tp}) - Trial {trial_idx}")

            except Exception as e:
                print(f"    Error processing {input_csv}: {e}")

    return pd.concat(minute_data_list, ignore_index=True) if minute_data_list else pd.DataFrame(), raw_trials_list

def export_processed_data(agg_df, raw_list, output_dir):
    """Flattens the processed lists and DataFrames and exports them cleanly to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export Hysteresis Data
    if not agg_df.empty:
        agg_path = os.path.join(output_dir, "hysteresis_data.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path}")

    # 2. Export Time-Series Data (Flattening the nested arrays)
    voltage_rows = []
    angle_rows = []

    for trial in raw_list:
        mat = trial["material"]
        tp = trial["time_point"]
        t_idx = trial["trial_idx"]
        t_id = trial["trial_id"]

        # Flatten Voltage Data
        if trial["voltage_data"] is not None:
            t_vals, v_vals = trial["voltage_data"]
            for t, v in zip(t_vals, v_vals):
                voltage_rows.append([mat, tp, t_idx, t_id, t, v])

        # Flatten Angle Data
        if trial["angle_data"] is not None:
            t_vals, a_vals = trial["angle_data"]
            for t, a in zip(t_vals, a_vals):
                angle_rows.append([mat, tp, t_idx, t_id, t, a])

    # Convert to DataFrames and save
    if voltage_rows:
        df_volts = pd.DataFrame(voltage_rows, columns=["material", "time_point", "trial_idx", "trial_id", "time_min", "voltage_v"])
        volts_path = os.path.join(output_dir, "timeseries_voltage.csv")
        df_volts.to_csv(volts_path, index=False)
        print(f"Saved: {volts_path}")

    if angle_rows:
        df_angles = pd.DataFrame(angle_rows, columns=["material", "time_point", "trial_idx", "trial_id", "time_min", "angle_deg"])
        angles_path = os.path.join(output_dir, "timeseries_angle.csv")
        df_angles.to_csv(angles_path, index=False)
        print(f"Saved: {angles_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_time_lapse_data.py <PATH_TO_ROOT_FOLDER>")
        sys.exit(1)
        
    root_path = sys.argv[1]
    output_dir = os.path.join(root_path, "Processed_Data_Exports")
    
    agg_df, raw_list = load_data(root_path)
    
    if agg_df.empty and not raw_list: 
        sys.exit("No valid data found to export.")
    
    print("\nExporting processed data to CSV...")
    export_processed_data(agg_df, raw_list, output_dir)
    print("Done! Data is ready for graphing.")