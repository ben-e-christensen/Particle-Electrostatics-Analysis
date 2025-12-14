#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import glob
from scipy.signal import medfilt, find_peaks
from scipy.stats import gaussian_kde  # <--- NEW IMPORT NEEDED HERE

# === CONFIG ===
BASE_DIR = "D:/particle-data"
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

# Find all subfolders containing experiment_log.csv
subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]

# Storage
all_summaries = []     # For Angle/Speed analysis
all_charge_stats = []  # For Charge analysis
all_minute_data = []   # For Correlation analysis

# === HELPER FUNCTIONS ===
def avg_top_percent(series):
    abs_series = series.abs()
    if len(abs_series) == 0: return 0
    q = 1.0 - (TOP_PERCENT_CHARGE / 100.0)
    cutoff = abs_series.quantile(q)
    return abs_series[abs_series >= cutoff].mean() if len(abs_series[abs_series >= cutoff]) > 0 else 0

def avg_voltage_peaks(series):
    if len(series) == 0: return 0
    peaks, _ = find_peaks(series.abs(), height=0.01, distance=10)
    if len(peaks) == 0: return 0
    return series.abs().iloc[peaks].mean()

# =========================================================
# 🔄 MASTER LOOP (Data Processing - Unchanged)
# =========================================================
# ... (This section of your code remains exactly the same) ...
# ... (Copy the loop from your original script up to the PLOTTING section) ...

for trial_folder in subfolders:
    input_csv = os.path.join(trial_folder, "experiment_log.csv")
    if not os.path.isfile(input_csv): continue

    trial_name = os.path.basename(trial_folder)
    print(f"   running trial: {trial_name}...")

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"]
    
    df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
    
    # Peak Detection & Speed Splitting
    peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=3.5)
    result_df = df.iloc[peak_indices].copy()
    result_df = result_df[result_df["ellipse_angle_deg"] <= 70]
    if len(result_df) == 0: continue

    result_df["speed_change"] = result_df["motor_speed"].ne(result_df["motor_speed"].shift())
    result_df["run_id"] = result_df["speed_change"].cumsum()

    # Summarize Trial
    trial_summary = (result_df.groupby("run_id")
        .agg(motor_speed=("motor_speed", "first"), angle_mean=("ellipse_angle_deg", "mean"),
             start_index=("index", "min"), end_index=("index", "max"))
        .reset_index(drop=True))
    
    trial_summary["trial"] = trial_name
    all_summaries.append(trial_summary)

    # Charge Calculation
    kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE) | 1
    df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
    df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]

    for _, row in trial_summary.iterrows():
        mask = (df["index"] >= row["start_index"]) & (df["index"] <= row["end_index"])
        segment = df.loc[mask]
        if len(segment) < 10: continue

        all_charge_stats.append({
            "trial": trial_name, "motor_speed": row["motor_speed"],
            "ch2_std": segment["CH2_clean"].std(),
            "ch2_top_pct": avg_top_percent(segment["CH2_clean"])
        })

    # Minute Aggregation
    t0 = df["timestamp"].iloc[0]
    df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int)
    result_df["minute_bin"] = ((result_df["timestamp"] - t0) / 60).astype(int)

    charge_per_minute = df.groupby("minute_bin").agg(
        ch2_std=("CH2_clean", "std"),
        ch2_top_pct=("CH2_volts", avg_top_percent),
        ch2_peak_avg=("CH2_volts", avg_voltage_peaks),
        motor_speed=("motor_speed", "mean")
    )
    angle_per_minute = result_df.groupby("minute_bin").agg(angle_mean=("ellipse_angle_deg", "mean"))
    
    minute_data = pd.merge(charge_per_minute, angle_per_minute, on="minute_bin", how="inner")
    minute_data["trial"] = trial_name
    all_minute_data.append(minute_data)

# =========================================================
# 📊 NEW CONTOUR PLOTTING SECTION
# =========================================================

if not all_minute_data:
    print("❌ No valid data found.")
    sys.exit(0)

master_minute_df = pd.concat(all_minute_data, ignore_index=True)
master_minute_df.to_csv(os.path.join(output_dir, "ALL_TRIALS_minute_correlation.csv"))

print("🎨 Generating Contour Maps...")

# --- PLOT 1: THE PHASE DIAGRAM (Speed vs Angle vs Charge) ---
# This creates a topological map where X=Speed, Y=Angle, and Color=Charge
if len(master_minute_df) > 5:
    plt.figure(figsize=(10, 7))
    
    x = master_minute_df["motor_speed"]
    y = master_minute_df["angle_mean"]
    z = master_minute_df["ch2_std"]
    
    # Use tricontourf for non-gridded data
    # 'levels' determines how smooth the color gradients are
    cntr = plt.tricontourf(x, y, z, levels=20, cmap="inferno")
    
    plt.colorbar(cntr, label="Electrostatic Charge (Volts Std Dev)")
    plt.plot(x, y, 'ko', ms=3, alpha=0.3, label="Data Points") # Show original points faintly
    
    plt.xlabel("Motor Speed (RPM)")
    plt.ylabel("Angle of Repose (°)")
    plt.title(f"{material_name}\nElectrostatic Phase Diagram")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "Contour_Phase_Diagram.png"))
    plt.close()

# --- PLOT 2: DENSITY CONTOURS FOR CORRELATION ---
# Instead of scatter plots, this shows density (Where do points cluster?)

def plot_density_contour(x_col, x_label, filename):
    plt.figure(figsize=(10, 7))
    # Filter for valid data > 0 to avoid log issues or empty plots
    valid = master_minute_df[master_minute_df[x_col] > 0].copy()
    
    # Check if we have enough data points to calculate density
    if len(valid) < 5: 
        plt.close()
        return

    x = valid[x_col]
    y = valid["angle_mean"]
    
    try:
        # Calculate Point Density (Gaussian KDE)
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # Sort so highest density is plotted on top
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = x.iloc[idx], y.iloc[idx], z[idx]
        
        # Scatter plot colored by density
        sc = plt.scatter(x_sorted, y_sorted, c=z_sorted, s=50, cmap="viridis", edgecolor='none', label="Density Data")
        plt.colorbar(sc, label="Data Point Density")
        
        # --- Add Trendlines (Linear Regressions) ---
        
        # Slow Trend
        slow = valid[valid["motor_speed"] < RPM_CUTOFF_LOW]
        if len(slow) > 2:
            m, b = np.polyfit(slow[x_col], slow["angle_mean"], 1)
            # Create a sorted x-range for a clean line
            x_line = np.linspace(slow[x_col].min(), slow[x_col].max(), 100)
            plt.plot(x_line, m*x_line + b, "b--", lw=2, label="Slow Trend")
            
        # Fast Trend
        fast = valid[valid["motor_speed"] >= RPM_CUTOFF_HIGH]
        if len(fast) > 2:
            m, b = np.polyfit(fast[x_col], fast["angle_mean"], 1)
            # Create a sorted x-range for a clean line
            x_line = np.linspace(fast[x_col].min(), fast[x_col].max(), 100)
            plt.plot(x_line, m*x_line + b, "r--", lw=2, label="Fast Trend")

    except Exception as e:
        print(f"⚠️ Could not generate density for {x_label}: {e}")
        # Fallback: Normal scatter if density fails
        plt.scatter(x, y, alpha=0.5, label="Data Points") 

    plt.xlabel(x_label)
    plt.ylabel("Angle of Repose (°)")
    plt.title(f"{material_name}\nAngle vs {x_label} (Density Contour)")
    
    # Only show legend if labels were actually created
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
        
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

# Generate the 3 Density plots
plot_density_contour("ch2_std", "Charge Noise (Std Dev)", "Contour_Corr_Std.png")
plot_density_contour("ch2_top_pct", f"Top {TOP_PERCENT_CHARGE}% Voltage", "Contour_Corr_TopPct.png")
plot_density_contour("ch2_peak_avg", "Avg Voltage Peak Height", "Contour_Corr_Peaks.png")

print(f"🚀 Done! Contour plots saved to: {output_dir}")