import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ================= CONFIGURATION =================
OUTPUT_FOLDER_NAME = "Ensemble_Macro_Results"
SEGMENT_SIZE_MIN = 5  
HARD_CUTOFF_MIN = 30.0 # <--- NEW: Strict limit
# =================================================

def process_macro_segmented(parent_dir):
    print(f"--- Starting Segmented Analysis ({SEGMENT_SIZE_MIN} min segments, {HARD_CUTOFF_MIN} min cutoff) ---")
    
    # 1. DISCOVER TRIALS
    trial_folders = []
    for root, dirs, files in os.walk(parent_dir):
        if "experiment_log.csv" in files:
            trial_folders.append(root)
    
    if not trial_folders:
        print("No trial folders found!")
        return

    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 2. MERGE DATA
    print("Merging Data...")
    increasing_data = []
    decreasing_data = []

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]

    for trial_path in trial_folders:
        try:
            csv_path = os.path.join(trial_path, "experiment_log.csv")
            df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
            df = df.dropna(subset=["ellipse_angle_deg", "motor_speed", "ms"])
            
            # SPLIT INCREASING / DECREASING
            max_val = df["motor_speed"].max()
            max_indices = df.index[df["motor_speed"] == max_val].tolist()
            if not max_indices: continue
            mid_idx = max_indices[len(max_indices) // 2]
            
            # INCREASING
            df_inc = df.loc[:mid_idx].copy()
            if not df_inc.empty:
                start_ms = df_inc["ms"].iloc[0]
                df_inc["rel_time_min"] = (df_inc["ms"] - start_ms) / 60000.0
                
                # --- HARD CUTOFF ---
                df_inc = df_inc[df_inc["rel_time_min"] <= HARD_CUTOFF_MIN]
                
                if not df_inc.empty:
                    increasing_data.append(df_inc)

            # DECREASING
            df_dec = df.loc[mid_idx+1:].copy()
            if not df_dec.empty:
                start_dec_ms = df_dec["ms"].iloc[0]
                df_dec["rel_time_min"] = (df_dec["ms"] - start_dec_ms) / 60000.0
                
                # --- HARD CUTOFF ---
                df_dec = df_dec[df_dec["rel_time_min"] <= HARD_CUTOFF_MIN]
                
                if not df_dec.empty:
                    decreasing_data.append(df_dec)

        except Exception as e:
            print(f"Skipping {os.path.basename(trial_path)}: {e}")

    # 3. COMBINE & FIT SEGMENTS
    phases = [
        ("INCREASING", pd.concat(increasing_data) if increasing_data else pd.DataFrame()),
        ("DECREASING", pd.concat(decreasing_data) if decreasing_data else pd.DataFrame())
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Global Y Limits
    all_angles = pd.concat([p[1]["ellipse_angle_deg"] for p in phases if not p[1].empty])
    if not all_angles.empty:
        y_min, y_max = all_angles.min(), all_angles.max()
        buff = (y_max - y_min) * 0.1
        ylim = (y_min - buff, y_max + buff)
    else:
        ylim = (0, 90)

    for i, (name, data) in enumerate(phases):
        ax = axes[i]
        
        if data.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        x_all = data["rel_time_min"].values
        y_all = data["ellipse_angle_deg"].values

        # Scatter
        ax.scatter(x_all, y_all, s=0.5, color='black', alpha=0.02, label='Raw Data Points', rasterized=True)

        # --- PIECEWISE LINEAR FIT ---
        # Generate bins strictly up to the Hard Cutoff
        bins = np.arange(0, HARD_CUTOFF_MIN + SEGMENT_SIZE_MIN, SEGMENT_SIZE_MIN)
        
        color = 'red' if name == "INCREASING" else 'purple'
        
        print(f"\n--- {name} Slopes ---")
        
        for j in range(len(bins) - 1):
            t_start = bins[j]
            t_end = bins[j+1]
            
            # Stop loop if we exceed the hard cutoff
            if t_start >= HARD_CUTOFF_MIN: break
            
            # Mask data for this segment
            mask = (x_all >= t_start) & (x_all < t_end)
            x_seg = x_all[mask]
            y_seg = y_all[mask]
            
            if len(x_seg) > 1:
                z = np.polyfit(x_seg, y_seg, 1)
                p = np.poly1d(z)
                
                fit_x = np.linspace(t_start, t_end, 10)
                fit_y = p(fit_x)
                
                lbl = f'Segment Fits ({SEGMENT_SIZE_MIN} min)' if j == 0 else "_nolegend_"
                ax.plot(fit_x, fit_y, color=color, linewidth=3, label=lbl)
                ax.axvline(x=t_end, color='gray', linestyle=':', alpha=0.5)

                print(f"  Time {int(t_start):02d}-{int(t_end):02d} min | Slope: {z[0]:.2f}")

        ax.set_title(f"{name} Phase (Max {int(HARD_CUTOFF_MIN)} min)", fontsize=18, fontweight='bold')
        ax.set_xlabel("Time (min)", fontsize=14)
        ax.set_ylabel("Angle (deg)", fontsize=14)
        ax.set_ylim(ylim)
        ax.set_xlim(0, HARD_CUTOFF_MIN) # Force X-Axis to stop at 30
        ax.grid(True, alpha=0.3)
        
        leg = ax.legend(fontsize=12, loc='upper right')
        for lh in leg.legend_handles: lh.set_alpha(1)

    plt.suptitle(f"Global Hysteresis: Piecewise Linear Fit ({SEGMENT_SIZE_MIN} min segments)", fontsize=22, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "Global_Macro_Segmented_Cutoff.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_macro_segmented_cutoff.py <path_to_PARENT_folder>")
    else:
        process_macro_segmented(sys.argv[1])