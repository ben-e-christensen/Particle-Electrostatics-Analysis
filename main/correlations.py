#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data/"
# ======================================================

# PROPERTY DATABASE
PHYSICS_DB = {
    "acetal":  {"density": 1.42, "tribo_rank": -1, "resistivity": 15},
    "acrylic": {"density": 1.18, "tribo_rank": 0,  "resistivity": 14},
    "nylon":   {"density": 1.14, "tribo_rank": 10, "resistivity": 12},
    "teflon":  {"density": 2.20, "tribo_rank": -20,"resistivity": 18}
}

def run_physics_contours(run_rel_path):
    full_run_dir = os.path.join(BASE_DIR, run_rel_path)
    data_path = os.path.join(full_run_dir, "analysis_comparative_metrics/master_comparison_data.csv")
    base_output_dir = os.path.join(full_run_dir, "physics_angle_contours/")

    if not os.path.exists(data_path):
        print(f"Error: Master CSV not found at {data_path}")
        return

    os.makedirs(base_output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df['material'] = df['material'].str.lower()

    # Re-calculate speed grouping if missing
    if 'grouped_speed' not in df.columns:
        df['grouped_speed'] = df['motor_speed'].round(0).astype(int) if 'motor_speed' in df.columns else 0

    # Map physics properties
    for prop in ["density", "tribo_rank", "resistivity"]:
        df[prop] = df['material'].map(lambda x: PHYSICS_DB.get(x, {}).get(prop, np.nan))

    unique_speeds = sorted(df['grouped_speed'].unique())
    
    # Define the X-Axis properties (Adding Collapses here)
    properties = [
        ("density", "Density (g per cm3)"),
        ("tribo_rank", "Triboelectric Rank"),
        ("resistivity", "Log Resistivity"),
        ("collapse_count", "Collapses per Minute") # <--- Now an X-axis correlation
    ]

    for prop_col, prop_label in properties:
        prop_dir = os.path.join(base_output_dir, prop_col.capitalize())
        os.makedirs(prop_dir, exist_ok=True)
        
        print(f"Generating Angle Heatmaps for Correlation: {prop_label}")

        for speed in unique_speeds:
            # Filter and drop rows missing our target X, Y, or Z
            speed_df = df[df['grouped_speed'] == speed].dropna(subset=[prop_col, 'voltage_std', 'angle_mean'])
            
            # Need at least 3 unique points to interpolate a surface
            if speed_df['material'].nunique() < 3:
                continue

            plt.figure(figsize=(10, 7))
            
            # X = The Correlation Variable (Density or Collapse Count etc)
            # Y = Voltage Std Dev (Charge)
            # Z = Angle of Repose (The HEAT)
            x = speed_df[prop_col]
            y = speed_df['voltage_std']
            z = speed_df['angle_mean']

            try:
                # Generate triangulated contour
                cntr = plt.tricontourf(x, y, z, levels=20, cmap="viridis", alpha=0.9)
                cbar = plt.colorbar(cntr)
                cbar.set_label("Angle of Repose (Degrees)", fontsize=12)
                
                # Subtle topography lines
                plt.tricontour(x, y, z, levels=20, colors='black', linewidths=0.3, alpha=0.2)
                
                # Scatter points (Clean circles)
                plt.scatter(x, y, c=z, cmap="viridis", edgecolors='white', s=110, linewidths=1.2, zorder=5)

            except Exception as e:
                print(f"  Failed RPM {speed} for {prop_col}: {e}")
                plt.close()
                continue

            plt.title(f"{prop_label} vs Charge | Heat: Angle | {speed} RPM", fontsize=14, pad=15)
            plt.xlabel(prop_label, fontsize=12)
            plt.ylabel("Voltage Std Dev (Charge Intensity)", fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(prop_dir, f"RPM_{speed}.png"))
            plt.close()

    print(f"\nAnalysis complete. Folders created in: {base_output_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <Vol>/<Cond>/<Dur>")
        sys.exit(1)

    run_rel_path = sys.argv[1].replace('\\', '/').strip('/')
    run_physics_contours(run_rel_path)