#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys, os

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data/"
PARTICLE_VOL_CM3 = 0.01675 

# Physical Properties Database
PHYSICS_DB = {
    "acetal":  {"density": 1.42, "resistivity": 15},
    "acrylic": {"density": 1.18, "resistivity": 14},
    "nylon":   {"density": 1.14, "resistivity": 12},
    "teflon":  {"density": 2.20, "resistivity": 18}
}

def generate_4d_plot(df, x_col, x_label, condition, filename):
    """Core plotting engine for 4D Ridge Analysis"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x_data = df[x_col]
    y_data = df['grouped_speed']
    z_data = df['voltage_std']
    angles = df['angle_mean']

    # --- RIDGE DETECTION ---
    # Find peak friction for every speed bin across the entire dataset
    ridge_df = df.sort_values('angle_mean', ascending=False).drop_duplicates('grouped_speed')
    ridge_df = ridge_df.sort_values('grouped_speed')
    rx, ry, rz = ridge_df[x_col].values, ridge_df['grouped_speed'].values, ridge_df['voltage_std'].values

    # --- 3D BEST FIT RIDGE ---
    try:
        z_coef = np.polyfit(ry, rz, 1) # Charge Trend
        x_coef = np.polyfit(ry, rx, 1) # X-Variable Trend
        
        y_line = np.linspace(ry.min(), ry.max(), 100)
        z_line = np.polyval(z_coef, y_line)
        x_line = np.polyval(x_coef, y_line)
        
        # Plot the Ridge Line
        ax.plot(x_line, y_line, z_line, color='cyan', linewidth=5, label='Friction Ridge', zorder=100)
        
        # Overlay Equations
        eq_text = (f"GLOBAL RIDGE ({condition}):\n"
                   f"Charge = {z_coef[0]:.4f}*RPM + {z_coef[1]:.4f}\n"
                   f"{x_label} = {x_coef[0]:.4f}*RPM + {x_coef[1]:.4f}")
        ax.text2D(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=11, color='cyan', 
                  bbox=dict(facecolor='black', alpha=0.8, edgecolor='cyan'))
    except Exception as e:
        print(f"Fit failed for {x_label}: {e}")

    # --- SCATTER ---
    img = ax.scatter(x_data, y_data, z_data, c=angles, cmap='plasma', s=60, alpha=0.4, edgecolors='none')
    ax.scatter(rx, ry, rz, color='cyan', s=120, edgecolors='white', label='Observed Peaks', zorder=101)

    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel('Motor Speed (RPM)', fontweight='bold')
    ax.set_zlabel('Charge (Voltage Std Dev)', fontweight='bold')
    
    cbar = fig.colorbar(img, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Angle of Repose (Degrees)')
    ax.legend(loc='upper right')

    plt.title(f"GLOBAL 4D {x_label.upper()} ANALYSIS: {condition.upper()}", fontsize=16)
    
    output_path = os.path.join(BASE_DIR, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Generated Plot: {output_path}")

def run_multi_global_analysis(condition):
    global_data = []
    condition = condition.capitalize()
    
    volumes = [v for v in os.listdir(BASE_DIR) if v.isdigit()]
    print(f"Crawling global data for {condition}...")

    for vol_str in volumes:
        num_particles = int(vol_str)
        cond_path = os.path.join(BASE_DIR, vol_str, condition)
        if not os.path.exists(cond_path): continue
            
        durations = [d for d in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, d))]
        for dur in durations:
            csv_path = os.path.join(cond_path, dur, "analysis_comparative_metrics", "master_comparison_data.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['num_particles'] = num_particles
                
                # Physics Mapping
                mat_lower = df['material'].str.lower()
                df['density'] = mat_lower.map(lambda x: PHYSICS_DB.get(x, {}).get('density'))
                df['resistivity'] = mat_lower.map(lambda x: PHYSICS_DB.get(x, {}).get('resistivity'))
                
                # Calculations
                df['total_mass_g'] = df['density'] * PARTICLE_VOL_CM3 * df['num_particles']
                df['charge_density'] = df['voltage_std'] / df['total_mass_g']
                global_data.append(df)

    if not global_data:
        print(f"No data found for {condition}")
        return

    master_df = pd.concat(global_data, ignore_index=True).dropna(subset=['total_mass_g', 'resistivity', 'voltage_std'])
    master_df['grouped_speed'] = master_df['motor_speed'].round(0).astype(int)

    # Output 1: The Mass Model
    generate_4d_plot(master_df, 'total_mass_g', 'Total Mass (g)', condition, f"Global_4D_MASS_{condition}.png")

    # Output 2: The Resistivity Model
    generate_4d_plot(master_df, 'resistivity', 'Log Surface Resistivity', condition, f"Global_4D_RESISTIVITY_{condition}.png")

    # Console Summary: Charge per Gram
    print(f"\n--- {condition.upper()} CHARGE DENSITY SUMMARY ---")
    summary = master_df.groupby('material')['charge_density'].mean().sort_values(ascending=False)
    for mat, val in summary.items():
        print(f"{mat.capitalize():<10}: {val:.6f} V/g")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <Dirty/Clean>")
        sys.exit(1)
    run_multi_global_analysis(sys.argv[1])