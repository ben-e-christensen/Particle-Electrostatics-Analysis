#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIG ===
DATA_PATH = "F:/particle-data/750/Dirty/60mins/analysis_comparative_metrics/master_comparison_data.csv"
BASE_OUTPUT_DIR = "F:/particle-data/750/Dirty/60mins/physics_angle_contours/"

# PROPERTY DATABASE
PHYSICS_DB = {
    "acetal":  {"density": 1.42, "tribo_rank": -1, "resistivity": 15},
    "acrylic": {"density": 1.18, "tribo_rank": 0,  "resistivity": 14},
    "nylon":   {"density": 1.14, "tribo_rank": 10, "resistivity": 12},
    "teflon":  {"density": 2.20, "tribo_rank": -20,"resistivity": 18}
}

def run_physics_contours():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df['material'] = df['material'].str.lower()

    if 'grouped_speed' not in df.columns:
        df['grouped_speed'] = df['motor_speed'].round(0).astype(int) if 'motor_speed' in df.columns else 0

    # Map physics properties
    for prop in ["density", "tribo_rank", "resistivity"]:
        df[prop] = df['material'].map(lambda x: PHYSICS_DB.get(x, {}).get(prop, np.nan))

    unique_speeds = sorted(df['grouped_speed'].unique())
    
    properties = [
        ("density", "Density (g per cm3)"),
        ("tribo_rank", "Triboelectric Rank"),
        ("resistivity", "Log Resistivity (Ohm per sq)")
    ]

    for prop_col, prop_label in properties:
        # Folder per property
        prop_dir = os.path.join(BASE_OUTPUT_DIR, prop_col.capitalize())
        os.makedirs(prop_dir, exist_ok=True)
        
        print(f"Generating Angle Heatmaps for: {prop_label}")

        for speed in unique_speeds:
            speed_df = df[df['grouped_speed'] == speed].dropna()
            
            if speed_df['material'].nunique() < 3:
                continue

            plt.figure(figsize=(10, 7))
            
            # X = Physical Property
            # Y = Voltage Std Dev (Charge)
            # Z = Angle of Repose (The Heat)
            x = speed_df[prop_col]
            y = speed_df['voltage_std']
            z = speed_df['angle_mean']

            try:
                # Using 'viridis' or 'plasma' to represent the steepness of the angle
                cntr = plt.tricontourf(x, y, z, levels=20, cmap="viridis", alpha=0.9)
                cbar = plt.colorbar(cntr)
                cbar.set_label("Angle of Repose (Degrees)", fontsize=12)
                
                # Topographical lines
                plt.tricontour(x, y, z, levels=20, colors='black', linewidths=0.3, alpha=0.2)
                
                # Data points (Clean, no labels)
                plt.scatter(x, y, c=z, cmap="viridis", edgecolors='white', s=110, linewidths=1.2, zorder=5)

            except Exception as e:
                print(f"  Failed RPM {speed} for {prop_col}: {e}")
                plt.close()
                continue

            plt.title(f"{prop_label} vs Charge | Color: Angle | {speed} RPM", fontsize=14, pad=15)
            plt.xlabel(prop_label, fontsize=12)
            plt.ylabel("Voltage Std Dev (Charge Intensity)", fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(prop_dir, f"RPM_{speed}.png"))
            plt.close()

    print(f"\nAll angle-heat contours generated in: {BASE_OUTPUT_DIR}")

if __name__ == '__main__':
    run_physics_contours()