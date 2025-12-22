#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys, os

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data/"

PHYSICS_DB = {
    "acetal":  {"density": 1.42, "tribo_rank": -1, "resistivity": 15},
    "acrylic": {"density": 1.18, "tribo_rank": 0,  "resistivity": 14},
    "nylon":   {"density": 1.14, "tribo_rank": 10, "resistivity": 12},
    "teflon":  {"density": 2.20, "tribo_rank": -20,"resistivity": 18}
}

def run_4d_ridge_analysis(run_rel_path):
    full_run_dir = os.path.join(BASE_DIR, run_rel_path)
    data_path = os.path.join(full_run_dir, "analysis_comparative_metrics/master_comparison_data.csv")
    output_dir = os.path.join(full_run_dir, "physics_4D_analysis/")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    df['material'] = df['material'].str.lower()
    
    for prop in ["density", "tribo_rank", "resistivity"]:
        df[prop] = df['material'].map(lambda x: PHYSICS_DB.get(x, {}).get(prop, np.nan))

    properties = [
        ("density", "Density"), 
        ("tribo_rank", "Tribo Rank"), 
        ("resistivity", "Resistivity"), 
        ("collapse_count", "Collapses")
    ]

    for prop_col, prop_label in properties:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Standardize axis data
        x_data = df[prop_col]
        y_data = df['grouped_speed'] # Speed
        z_data = df['voltage_std']   # Charge
        angles = df['angle_mean']

        # --- RIDGE DETECTION ---
        # Get the highest angle for every speed bin
        ridge_df = df.sort_values('angle_mean', ascending=False).drop_duplicates('grouped_speed')
        ridge_df = ridge_df.sort_values('grouped_speed')
        
        rx = ridge_df[prop_col].values
        ry = ridge_df['grouped_speed'].values
        rz = ridge_df['voltage_std'].values

        # --- 3D BEST FIT CALCULATION ---
        try:
            # We treat Speed (Y) as the independent variable
            # Fit Z (Charge) vs Y (Speed)
            z_coef = np.polyfit(ry, rz, 1)
            # Fit X (Property) vs Y (Speed)
            x_coef = np.polyfit(ry, rx, 1)
            
            # Create the line coordinates
            y_line = np.linspace(ry.min(), ry.max(), 100)
            z_line = np.polyval(z_coef, y_line)
            x_line = np.polyval(x_coef, y_line)
            
            # Plot the thick Red Ridge Line
            ax.plot(x_line, y_line, z_line, color='red', linewidth=2.5, label='Friction Ridge Line', zorder=100)
            
            # Create Equation Strings
            eq_text = (f"Ridge Equations:\n"
                       f"Charge = {z_coef[0]:.4f}*RPM + {z_coef[1]:.4f}\n"
                       f"{prop_label} = {x_coef[0]:.4f}*RPM + {x_coef[1]:.4f}")
            
            # Place equation on the graph
            ax.text2D(0.05, 0.95, eq_text, transform=ax.transAxes, 
                      fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.7))

        except Exception as e:
            print(f"Fit failed for {prop_label}: {e}")

        # --- SCATTER PLOT ---
        img = ax.scatter(x_data, y_data, z_data, c=angles, cmap='viridis', s=80, alpha=0.5, edgecolors='w', linewidths=0.5)
        
        # Observed Peak Points (larger markers for clarity)
        ax.scatter(rx, ry, rz, color='red', s=150, edgecolors='black', label='Observed Peaks', zorder=101)

        ax.set_xlabel(prop_label, fontweight='bold')
        ax.set_ylabel('Motor Speed (RPM)', fontweight='bold')
        ax.set_zlabel('Charge (Voltage Std Dev)', fontweight='bold')
        
        cbar = fig.colorbar(img, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label('Angle of Repose (Degrees)')
        ax.legend(loc='upper right')

        plt.title(f"4D Ridge Analysis: {prop_label}", fontsize=15)
        
        save_path = os.path.join(output_dir, f"4D_Ridge_Eq_{prop_col}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Generated Ridge Line with Equation: {save_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <Vol>/<Cond>/<Dur>")
        sys.exit(1)
    run_4d_ridge_analysis(sys.argv[1])