import pandas as pd
import matplotlib.pyplot as plt
import random
import os

# ================= CONFIGURATION =================
# File Paths
CSV_FILE_PATH = "D:/particle-data/Clean/Teflon/500/Teflon-Clean-500-60mins-T3/experiment_log.csv"

# Output Filenames
OUTPUT_GRAPH_PATH = "charge_graph_snapshot.png"

# Analysis Settings
SAMPLE_RATE_HZ = 100
WINDOW_SECONDS = 15

# --- SIGNAL PROCESSING ---
# Number of data points to average together. 1 = No smoothing.
SMOOTHING_WINDOW = 1
FLIP_CH2 = True  # Set to True to flip downward spikes up
# =================================================

# ---------------------------------------------------------
# 1. Load and Smooth Data
# ---------------------------------------------------------
def load_random_window(csv_path, window_seconds):
    """
    Reads the CSV, applies a smoothing filter, and returns 
    a random continuous slice of CH2 data.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None

    print("Reading CSV data...")
    
    # Standard columns matching your main script
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

    # Clean and optionally flip the data
    df = df.dropna(subset=["CH2_volts"]).copy()
    multiplier = -1.0 if FLIP_CH2 else 1.0
    df["CH2_volts"] = pd.to_numeric(df["CH2_volts"], errors='coerce') * multiplier
    df = df.dropna(subset=["CH2_volts"])

    # Apply Smoothing Filter
    if SMOOTHING_WINDOW > 1:
        print(f"Applying moving average filter (Window: {SMOOTHING_WINDOW} points)...")
        df["CH2_volts"] = df["CH2_volts"].rolling(window=SMOOTHING_WINDOW, center=True).mean()
        df = df.dropna(subset=["CH2_volts"])

    samples_needed = window_seconds * SAMPLE_RATE_HZ
    total_points = len(df)
    
    if total_points < samples_needed:
        print(f"Not enough data found ({total_points} points). Need {samples_needed}.")
        return None

    # Pick a random start index
    max_start = total_points - samples_needed
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + samples_needed

    print(f"Selected {window_seconds}s Window: Data Index {start_idx} to {end_idx}")

    # Return just the CH2 slice as a numpy array
    return df["CH2_volts"].iloc[start_idx:end_idx].values


def plot_charge_graph(ch2_data):
    # Create Time-axis starting at 0
    t_vals = [i / SAMPLE_RATE_HZ for i in range(len(ch2_data))]

    plt.figure(figsize=(12, 6))
    plt.plot(t_vals, ch2_data, color='black', linewidth=1.5, label="Smoothed Charge")

    # Force the X-axis to start exactly at 0 and end at the max time
    plt.xlim(0, max(t_vals)) 

    # Styling
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Voltage (V)", fontsize=18)
    
    title_text = f"{WINDOW_SECONDS}-Second Charge Snapshot"
    if SMOOTHING_WINDOW > 1:
        title_text += f" (Smoothed: {SMOOTHING_WINDOW} pts)"
    
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14) 
    plt.tight_layout()

    print(f"Saving graph to {OUTPUT_GRAPH_PATH}...")
    plt.savefig(OUTPUT_GRAPH_PATH, dpi=300, bbox_inches='tight')

# ================= MAIN =================
if __name__ == "__main__":
    ch2_slice = load_random_window(CSV_FILE_PATH, WINDOW_SECONDS)

    if ch2_slice is not None:
        plot_charge_graph(ch2_slice)
        print("\nDone. Showing generated plot...")
        plt.show()