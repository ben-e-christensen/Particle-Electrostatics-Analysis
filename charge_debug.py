import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ================= CONFIGURATION =================
CSV_FILE_PATH = "D:/particle-data/time_lapse/Clean-Undisturbed/day7/experiment_log.csv"
OUTPUT_GRAPH_PATH = "day7_verified_peak_snapshot.png"
# =================================================

# --- TOGGLES ---
SHOW_LEGEND = False  
SHOW_TITLES = False  

# Analysis Settings
SAMPLE_RATE_HZ = 100
WINDOW_SECONDS = 500

# --- SIGNAL PROCESSING ---
SMOOTHING_WINDOW = 1
FLIP_CH2 = True  
BASELINE_WINDOW = 6000 # 1-minute rolling median to isolate the relative spikes

# --- ACADEMIC PLOT SETTINGS ---
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
FONT_TICK = 18
FONT_LABEL = 18
FONT_TITLE = 20
# =================================================

def apply_academic_axes(ax, xlabel="", ylabel=""):
    ax.xaxis.set_tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_tick_params(labelsize=FONT_TICK)
    ax.tick_params('both', length=7, width=1, which='major')
    if xlabel: ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel: ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.grid(True, alpha=0.3)

def load_relative_extreme_window(csv_path, window_seconds):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None

    print("Reading CSV data...")
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]
            
    try:
        df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", 
                         dtype={"CH2_volts": "float32"})
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    df = df.dropna(subset=["CH2_volts"]).copy()
    multiplier = -1.0 if FLIP_CH2 else 1.0
    df["CH2_volts"] = pd.to_numeric(df["CH2_volts"], errors='coerce') * multiplier
    df = df.dropna(subset=["CH2_volts"])

    if SMOOTHING_WINDOW > 1:
        df["CH2_volts"] = df["CH2_volts"].rolling(window=SMOOTHING_WINDOW, center=True).mean()
        df = df.dropna(subset=["CH2_volts"])

    print("Calculating baseline to find true relative peaks...")
    df["Baseline_V"] = df["CH2_volts"].rolling(window=BASELINE_WINDOW, min_periods=1, center=True).median()
    df["Relative_V"] = df["CH2_volts"] - df["Baseline_V"]

    # --- THE FIX: THE HEALTHY MASK ---
    # We only look for peaks when the baseline is sitting comfortably near zero.
    # This completely blinds the script to the massive hardware crash later in the day.
    healthy_tolerance = 0.05
    healthy_df = df[df["Baseline_V"].abs() < healthy_tolerance]
    
    if healthy_df.empty:
        print("Error: No healthy baseline data found in this CSV!")
        return None

    # Find the absolute largest RELATIVE spike, strictly within the healthy data
    extreme_idx = healthy_df["Relative_V"].abs().idxmax()
    extreme_relative_val = healthy_df.loc[extreme_idx, "Relative_V"]
    extreme_raw_val = df.loc[extreme_idx, "CH2_volts"]
    
    extreme_iloc = df.index.get_loc(extreme_idx)

    half_window_samples = int((window_seconds * SAMPLE_RATE_HZ) / 2)
    start_idx = max(0, extreme_iloc - half_window_samples)
    end_idx = min(len(df), extreme_iloc + half_window_samples)

    print(f"-> TARGET LOCKED (Healthy Zone): Found true relative peak of {extreme_relative_val:.4f} V (Raw: {extreme_raw_val:.4f} V) at row {extreme_iloc}")
    print(f"-> Selected {window_seconds}s Window: Data Index {start_idx} to {end_idx}")

    return df["CH2_volts"].iloc[start_idx:end_idx].values

def plot_charge_graph(ch2_data):
    t_vals = [i / SAMPLE_RATE_HZ for i in range(len(ch2_data))]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    
    ax.plot(t_vals, ch2_data, color='black', linewidth=1.5, label="Charge")

    if SHOW_TITLES:
        ax.set_title(f"{WINDOW_SECONDS}s True Peak Snapshot", fontsize=FONT_TITLE, fontweight='bold')

    ax.set_xlim(0, max(t_vals)) 
    apply_academic_axes(ax, "Time (s)", "Voltage (V)")
    
    if SHOW_LEGEND:
        ax.legend(fontsize=12, loc='best')

    print(f"Saving graph to {OUTPUT_GRAPH_PATH}...")
    plt.savefig(OUTPUT_GRAPH_PATH, dpi=300)
    plt.close()

if __name__ == "__main__":
    ch2_slice = load_relative_extreme_window(CSV_FILE_PATH, WINDOW_SECONDS)

    if ch2_slice is not None:
        plot_charge_graph(ch2_slice)
        print("\nDone. Check the output directory for the snapshot.")