import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ================= CONFIGURATION =================
CSV_FILE_PATH = "D:/particle-data/time_lapse/Clean-Undisturbed/day7/experiment_log.csv"
OUTPUT_GRAPH_PATH = "Day7_Custom_Window.png"

# --- THE TUNING VARIABLES ---
# Set the time window in HOURS relative to the start of Day 7
START_TIME_HOURS = 0.0
END_TIME_HOURS = 1.0

# Setup
FLIP_CH2 = True
SAMPLE_RATE_HZ = 100

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
# =================================================

def plot_custom_window():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: {CSV_FILE_PATH} not found.")
        return

    print(f"Reading Day 7 data from Hour {START_TIME_HOURS} to Hour {END_TIME_HOURS}...")
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]
            
    try:
        # We read the whole file here so we don't have to guess row counts for later hours
        df = pd.read_csv(CSV_FILE_PATH, names=cols, header=0, on_bad_lines="skip", 
                         dtype={"ms": "float32", "CH2_volts": "float32"})
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df = df.dropna(subset=["CH2_volts", "ms"]).copy()
    
    # Apply CH2 flip
    multiplier = -1.0 if FLIP_CH2 else 1.0
    df["CH2_volts"] = df["CH2_volts"] * multiplier

    # Convert ms to relative hours
    start_ms = df["ms"].min()
    df["Time_Hours"] = (df["ms"] - start_ms) / 3600000.0

    # Apply the tuning variables
    mask = (df["Time_Hours"] >= START_TIME_HOURS) & (df["Time_Hours"] <= END_TIME_HOURS)
    window_df = df[mask]
    
    if window_df.empty:
        print("Error: No data found in that specific time window!")
        return

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    # Plotted with a slightly thinner line and alpha so the dense spikes don't become a solid black block
    ax.plot(window_df["Time_Hours"], window_df["CH2_volts"], color='black', linewidth=0.5, alpha=0.8)
    
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlabel("Time (Hours)", fontsize=16)
    ax.set_ylabel("Voltage (V)", fontsize=16)
    ax.set_title(f"Day 7: Hour {START_TIME_HOURS} to {END_TIME_HOURS}", fontsize=18, fontweight='bold')
    
    ax.set_xlim(START_TIME_HOURS, END_TIME_HOURS)

    print(f"Saving windowed graph to {OUTPUT_GRAPH_PATH}...")
    plt.savefig(OUTPUT_GRAPH_PATH, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_custom_window()
    print("\nDone.")