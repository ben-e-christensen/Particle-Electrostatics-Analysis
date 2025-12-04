#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import glob
from scipy.signal import medfilt, find_peaks

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data/"
SAMPLE_RATE = 100  # Hz
BASELINE_WINDOW_SEC = 4  # smoothing window duration

# !!! CHANGE THIS VALUE TO ADJUST THE GRAPH !!!
TOP_PERCENT_CHARGE = 10  # e.g., 10 for Top 10%

# --- SPEED CATEGORY CUTOFFS ---
RPM_CUTOFF_LOW = 10      # Below this is "Slow"
RPM_CUTOFF_HIGH = 20     # Above this is "Fast" (Between is "Medium")

# --- Parse argument ---
if len(sys.argv) < 2:
    print("Usage: python3 main.py <Material/RunFolder>")
    print("Example: python3 main.py Acetal-Dirty/400/12mins")
    sys.exit(1)

rel_path = sys.argv[1]
parent_dir = os.path.join(BASE_DIR, rel_path)

if not os.path.isdir(parent_dir):
    print(f"❌ Error: {parent_dir} is not a directory")
    sys.exit(1)

# === OUTPUT PATHS ===
output_dir = os.path.join(parent_dir, "aggregated_results")
os.makedirs(output_dir, exist_ok=True)
plot_dir = output_dir 

run_name = os.path.basename(os.path.normpath(parent_dir))
material_name = run_name.replace("-", " ")

print(f"📂 Analyzing Group: {run_name}")
print(f"⚙️  Filtering for Top {TOP_PERCENT_CHARGE}% Charge events")
print(f"📉 Splitting trends: Slow (<{RPM_CUTOFF_LOW}), Medium ({RPM_CUTOFF_LOW}-{RPM_CUTOFF_HIGH}), Fast (>{RPM_CUTOFF_HIGH})")

# Find all subfolders containing experiment_log.csv
subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]

# Storage
all_summaries = []     # For Angle/Speed analysis
all_charge_stats = []  # For Charge analysis
all_minute_data = []   # For Correlation analysis

# === HELPER FUNCTIONS FOR CORRELATION ===
def avg_top_percent(series):
    """Calculates the mean of the top X percent of values (absolute)."""
    abs_series = series.abs()
    if len(abs_series) == 0: return 0
    q = 1.0 - (TOP_PERCENT_CHARGE / 100.0)
    cutoff = abs_series.quantile(q)
    top_values = abs_series[abs_series >= cutoff]
    return top_values.mean() if len(top_values) > 0 else 0

def avg_voltage_peaks(series):
    if len(series) == 0: return 0
    peaks, _ = find_peaks(series.abs(), height=0.01, distance=10)
    if len(peaks) == 0: return 0
    return series.abs().iloc[peaks].mean()

# =========================================================
# 🔄 MASTER LOOP: PROCESS EACH TRIAL
# =========================================================

for trial_folder in subfolders:
    input_csv = os.path.join(trial_folder, "experiment_log.csv")
    
    if not os.path.isfile(input_csv):
        continue

    trial_name = os.path.basename(trial_folder)
    print(f"   running trial: {trial_name}...")

    # --- Load CSV ---
    cols = [
        "index", "timestamp", "seq", "ms",
        "motor_angle_deg", "motor_speed", "CH0_volts", "CH2_volts", "CH3_volts",
        "ellipse_angle_deg", "ellipse_area_px2", "frame_name",
        "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"
    ]
    df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
    
    # === STEP 1: Peak Detection & Speed Splitting ===
    peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=3.5)
    result_df = df.iloc[peak_indices].copy()
    result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

    if len(result_df) == 0: continue

    # Speed Splitting
    result_df["speed_change"] = result_df["motor_speed"].ne(result_df["motor_speed"].shift())
    result_df["run_id"] = result_df["speed_change"].cumsum()

    # Fix specific 26 RPM split logic if needed (Legacy support)
    mask26 = (result_df["motor_speed"] == 26)
    count26 = mask26.sum()
    if count26 > 0:
        idxs_26 = result_df.index[mask26].to_list()
        half = count26 // 2
        run_forward = idxs_26[:half]
        run_backward = idxs_26[half:]
        
        if len(run_forward) > 0 and len(run_backward) > 0:
            fwd_id = result_df.loc[run_forward[0], "run_id"]
            result_df.loc[run_forward, "run_id"] = fwd_id
            result_df.loc[run_backward, "run_id"] = fwd_id + 1
            result_df.loc[result_df.index > run_backward[-1], "run_id"] += 1

    # Summarize Trial
    trial_summary = (
        result_df.groupby("run_id")
        .agg(
            motor_speed=("motor_speed", "first"),
            angle_mean=("ellipse_angle_deg", "mean"),
            angle_std=("ellipse_angle_deg", "std"),
            start_index=("index", "min"),
            end_index=("index", "max")
        )
        .reset_index(drop=True)
    )
    
    trial_summary["trial"] = trial_name
    half_pt = len(trial_summary) // 2
    trial_summary["direction"] = "Backward"
    trial_summary.iloc[:half_pt, trial_summary.columns.get_loc("direction")] = "Forward"
    all_summaries.append(trial_summary)

    # === STEP 2: Charge Calculation ===
    kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE) | 1
    df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
    df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
    df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
    df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

    for _, row in trial_summary.iterrows():
        mask = (df["index"] >= row["start_index"]) & (df["index"] <= row["end_index"])
        segment = df.loc[mask]
        if len(segment) < 10: continue

        ch2_std = segment["CH2_clean"].std()
        ch3_std = segment["CH3_clean"].std()
        
        def get_top_pct_stat(s):
            q_val = 1.0 - (TOP_PERCENT_CHARGE / 100.0)
            return s.abs()[s.abs() >= s.abs().quantile(q_val)].mean()
        
        all_charge_stats.append({
            "trial": trial_name,
            "motor_speed": row["motor_speed"],
            "direction": row["direction"],
            "ch2_std": ch2_std,
            "ch3_std": ch3_std,
            "ch2_top_pct": get_top_pct_stat(segment["CH2_clean"])
        })

    # === STEP 3: MINUTE-BY-MINUTE AGGREGATION ===
    t0 = df["timestamp"].iloc[0]
    df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)
    result_df["minute_bin"] = ((result_df["timestamp"] - t0) / 60).astype(int)

    # Aggregate Charge per minute
    charge_per_minute = (
        df.groupby("minute_bin")
        .agg(
            ch2_std=("CH2_clean", "std"),
            ch2_top_pct=("CH2_volts", avg_top_percent), 
            ch2_peak_avg=("CH2_volts", avg_voltage_peaks),
            motor_speed=("motor_speed", "mean")
        )
    )

    # Aggregate Angle per minute
    angle_per_minute = (
        result_df.groupby("minute_bin")
        .agg(
            angle_mean=("ellipse_angle_deg", "mean"),
            num_peaks=("ellipse_angle_deg", "size")
        )
    )

    # Merge
    minute_data = pd.merge(charge_per_minute, angle_per_minute, on="minute_bin", how="inner")
    minute_data["trial"] = trial_name
    all_minute_data.append(minute_data)


# =========================================================
# 📊 PLOTTING
# =========================================================

if not all_summaries:
    print("❌ No valid data found.")
    sys.exit(0)

# Combine Data
master_angle_df = pd.concat(all_summaries, ignore_index=True)
master_charge_df = pd.DataFrame(all_charge_stats)
master_minute_df = pd.concat(all_minute_data, ignore_index=True) if all_minute_data else pd.DataFrame()

# --- 1. Angle vs Speed ---
plt.figure(figsize=(10, 6))
global_stats = master_angle_df.groupby(["motor_speed", "direction"])["angle_mean"].agg(["mean", "std"]).reset_index()
fwd = global_stats[global_stats["direction"] == "Forward"].sort_values("motor_speed")
bwd = global_stats[global_stats["direction"] == "Backward"].sort_values("motor_speed", ascending=False)

plt.errorbar(fwd["motor_speed"], fwd["mean"], yerr=fwd["std"], fmt="-o", label="AVG Forward")
plt.errorbar(bwd["motor_speed"], bwd["mean"], yerr=bwd["std"], fmt="--s", label="AVG Backward")

plt.xlabel("Motor Speed (RPM)")
plt.ylabel("Angle of Repose (°)")
plt.title(f"{material_name} — Aggregated Angle Sweep")
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig(os.path.join(plot_dir, "Combined_Angle_Sweep.png"))
plt.close()

# --- 2. Charge vs Speed ---
if not master_charge_df.empty:
    plt.figure(figsize=(10, 6))
    c_stats = master_charge_df.groupby(["motor_speed", "direction"])["ch2_std"].agg(["mean", "std"]).reset_index()
    c_fwd = c_stats[c_stats["direction"] == "Forward"].sort_values("motor_speed")
    
    plt.plot(c_fwd["motor_speed"], c_fwd["mean"], "-o", color="green", label="CH2 Noise (Fwd)")
    plt.fill_between(c_fwd["motor_speed"], c_fwd["mean"] - c_fwd["std"], c_fwd["mean"] + c_fwd["std"], color="green", alpha=0.1)
    
    plt.xlabel("Motor Speed (RPM)")
    plt.ylabel("Electrostatic Activity (Std Dev)")
    plt.title(f"{material_name} — Aggregated Charge Activity")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(plot_dir, "Combined_Charge_Sweep.png"))
    plt.close()

# --- 3. UPDATED SEGMENTED CORRELATION ANALYSIS (3-WAY SPLIT) ---
if not master_minute_df.empty:
    print(f"⏱️ plotting minute-by-minute correlations across {len(master_minute_df)} total minutes...")
    
    master_minute_df.to_csv(os.path.join(output_dir, "ALL_TRIALS_minute_correlation.csv"))

    def plot_correlation(x_col, x_label, filename):
        plt.figure(figsize=(10, 7))
        valid = master_minute_df[master_minute_df[x_col] > 0].copy()
        
        if len(valid) == 0:
            plt.close()
            return

        # Scatter Plot (Colored by RPM)
        sc = plt.scatter(
            valid[x_col], 
            valid["angle_mean"], 
            c=valid["motor_speed"], 
            cmap="viridis", 
            s=50, 
            edgecolors="black", 
            alpha=0.7
        )
        cbar = plt.colorbar(sc)
        cbar.set_label("Motor Speed (RPM)")
        
        # --- 3-WAY SEGMENTED TREND LINES ---
        
        # 1. SLOW (< 10)
        slow_data = valid[valid["motor_speed"] < RPM_CUTOFF_LOW]
        if len(slow_data) > 1:
            m, b = np.polyfit(slow_data[x_col], slow_data["angle_mean"], 1)
            x_fit = np.linspace(slow_data[x_col].min(), slow_data[x_col].max(), 100)
            y_fit = m * x_fit + b
            plt.plot(x_fit, y_fit, "--", color="blue", lw=2.5, label=f"Slow (<{RPM_CUTOFF_LOW}): y={m:.2f}x + {b:.2f}")

        # 2. MEDIUM (10 <= x < 20)
        mask_med = (valid["motor_speed"] >= RPM_CUTOFF_LOW) & (valid["motor_speed"] < RPM_CUTOFF_HIGH)
        med_data = valid[mask_med]
        if len(med_data) > 1:
            m, b = np.polyfit(med_data[x_col], med_data["angle_mean"], 1)
            x_fit = np.linspace(med_data[x_col].min(), med_data[x_col].max(), 100)
            y_fit = m * x_fit + b
            plt.plot(x_fit, y_fit, "--", color="orange", lw=2.5, label=f"Med ({RPM_CUTOFF_LOW}-{RPM_CUTOFF_HIGH}): y={m:.2f}x + {b:.2f}")

        # 3. FAST (>= 20)
        fast_data = valid[valid["motor_speed"] >= RPM_CUTOFF_HIGH]
        if len(fast_data) > 1:
            m, b = np.polyfit(fast_data[x_col], fast_data["angle_mean"], 1)
            x_fit = np.linspace(fast_data[x_col].min(), fast_data[x_col].max(), 100)
            y_fit = m * x_fit + b
            plt.plot(x_fit, y_fit, "--", color="red", lw=2.5, label=f"Fast (≥{RPM_CUTOFF_HIGH}): y={m:.2f}x + {b:.2f}")

        plt.xlabel(x_label)
        plt.ylabel("Mean Angle of Repose (°)")
        plt.title(f"{material_name}\nAngle vs {x_label} (Split: Slow/Med/Fast)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()

    # Generate the 3 correlation plots
    plot_correlation("ch2_std", "Charge Noise (Std Dev)", "Combined_Corr_Std.png")
    
    pct_label = f"Avg Voltage of Top {TOP_PERCENT_CHARGE}%"
    plot_correlation("ch2_top_pct", pct_label, f"Combined_Corr_Top{TOP_PERCENT_CHARGE}Pct.png")
    
    plot_correlation("ch2_peak_avg", "Avg Voltage Peak Height", "Combined_Corr_Peaks.png")

print(f"🚀 Done! Aggregated results + 3-Way Split Correlation plots saved to: {output_dir}")