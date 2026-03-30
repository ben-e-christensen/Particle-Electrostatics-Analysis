import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ================= CONFIGURATION =================
PARENT_DIR = "D:/particle-data/time_lapse/Clean-Undisturbed/"
OUTPUT_DIR = os.path.join(PARENT_DIR, "Macro_Event_Gallery")

# Trigger Settings
# If the script finds too many/too few events, tweak the STDEV_THRESHOLD.
STDEV_WINDOW = 500      # 5 seconds at 100Hz
STDEV_THRESHOLD = 0.03  # Voltage variance that defines a "Macro-Event"
COOLDOWN_S = 300        # Don't trigger again for 5 mins (prevents multi-plotting one event)

# Plotting Settings
BUFFER_S = 90           # 90s before and 90s after
SAMPLE_RATE_HZ = 100
FLIP_CH2 = True

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
# =================================================

def apply_academic_axes(ax, xlabel="", ylabel=""):
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    if xlabel: ax.set_xlabel(xlabel, fontsize=16)
    if ylabel: ax.set_ylabel(ylabel, fontsize=16)

def hunt_events():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    for day_idx in range(1, 8):
        day_key = f"day{day_idx}"
        day_path = os.path.join(PARENT_DIR, day_key, "experiment_log.csv")
        
        if not os.path.exists(day_path):
            print(f"Skipping {day_key}: File not found.")
            continue

        print(f"--- Scanning {day_key} for Macro-Events... ---")
        
        try:
            # Use chunking if files are massive, but for 24h a standard read is usually okay on your tower
            df = pd.read_csv(day_path, names=cols, header=0, on_bad_lines="skip", 
                             dtype={"ms": "float32", "CH2_volts": "float32"})
            
            df = df.dropna(subset=["CH2_volts", "ms"]).copy()
            multiplier = -1.0 if FLIP_CH2 else 1.0
            df["CH2_volts"] = df["CH2_volts"] * multiplier
            
            # 1. Calculate Activity Trigger (Rolling StDev)
            df["Activity"] = df["CH2_volts"].rolling(window=STDEV_WINDOW, center=True).std()
            
            # 2. Find peaks in activity that cross the threshold
            potential_triggers = df[df["Activity"] > STDEV_THRESHOLD]
            
            if potential_triggers.empty:
                print(f"  -> No Macro-Events found in {day_key}.")
                continue

            last_trigger_time = -99999
            event_count = 0

            for idx in potential_triggers.index:
                curr_ms = df.loc[idx, "ms"]
                
                # Check cooldown to ensure we don't plot the same event 10 times
                if (curr_ms - last_trigger_time) < (COOLDOWN_S * 1000):
                    continue
                
                # 3. Define Window (90s before/after)
                half_window = BUFFER_S * 1000
                start_ms = curr_ms - half_window
                end_ms = curr_ms + half_window
                
                slice_df = df[(df["ms"] >= start_ms) & (df["ms"] <= end_ms)].copy()
                
                if slice_df.empty: continue
                
                # Plotting
                event_count += 1
                last_trigger_time = curr_ms
                t_rel = (slice_df["ms"] - slice_df["ms"].min()) / 1000.0
                
                fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
                ax.plot(t_rel, slice_df["CH2_volts"], color='black', linewidth=0.8)
                
                # Format
                apply_academic_axes(ax, "Time (s)", "Voltage (V)")
                ax.set_title(f"Macro-Event: {day_key} (Event #{event_count})", fontsize=18, fontweight='bold')
                ax.set_xlim(0, BUFFER_S * 2)
                
                # Save
                fname = f"{day_key}_Event_{event_count}.png"
                plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
                plt.close()
                
                print(f"  [!] Event {event_count} captured and saved.")

        except Exception as e:
            print(f"Error processing {day_key}: {e}")

if __name__ == "__main__":
    hunt_events()
    print(f"\nDone. All captured events are in: {OUTPUT_DIR}")