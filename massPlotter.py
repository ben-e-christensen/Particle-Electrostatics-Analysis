#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data/"
# Sphere volume for 1/8" particle in cm^3
PARTICLE_VOL_CM3 = 0.01675

# Physical Properties
DENSITIES = {
    "acetal": 1.42,
    "acrylic": 1.18,
    "nylon": 1.14,
    "teflon": 2.20
}

def run_global_analysis(condition):
    """
    Crawls BASE_DIR/[Volume]/[Condition]/[Duration]/...
    to build a global mass-charge-angle map.
    """
    global_data = []
    
    # 1. Walk through the Volume folders (500, 750, 1000)
    volumes = [v for v in os.listdir(BASE_DIR) if v.isdigit()]
    print(f"Searching for {condition} data across volumes: {volumes}")

    for vol_str in volumes:
        num_particles = int(vol_str)
        cond_path = os.path.join(BASE_DIR, vol_str, condition)
        
        if not os.path.exists(cond_path):
            continue
            
        # 2. Walk through Duration folders (12mins, 60mins, etc.)
        durations = [d for d in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, d))]
        
        for dur in durations:
            csv_path = os.path.join(cond_path, dur, "analysis_comparative_metrics", "master_comparison_data.csv")
            
            if os.path.exists(csv_path):
                print(f"  Loading: {vol_str}/{condition}/{dur}")
                df = pd.read_csv(csv_path)
                
                # Add metadata
                df['num_particles'] = num_particles
                df['condition'] = condition
                df['duration'] = dur
                
                # 3. Mass Calculation
                # Map density then multiply: Mass = Density * Vol_p * N
                df['particle_density'] = df['material'].str.lower().map(DENSITIES)
                df['total_mass_g'] = df['particle_density'] * PARTICLE_VOL_CM3 * df['num_particles']
                
                global_data.append(df)

    if not global_data:
        print(f"No master_comparison_data.csv files found for condition: {condition}")
        return

    # Combine everything
    master_df = pd.concat(global_data, ignore_index=True).dropna(subset=['total_mass_g', 'voltage_std', 'angle_mean'])
    
    # 4. Plotting the Global Contour
    # X = Total Mass, Y = Charge (Std Dev), Z = Angle (Heat)
    plt.figure(figsize=(12, 8))
    
    x = master_df['total_mass_g']
    y = master_df['voltage_std']
    z = master_df['angle_mean']

    try:
        # Triangulated Contour
        cntr = plt.tricontourf(x, y, z, levels=25, cmap="viridis", alpha=0.9)
        cbar = plt.colorbar(cntr)
        cbar.set_label("Angle of Repose (Degrees)", fontsize=12)
        
        # Overlay the data points
        plt.scatter(x, y, c=z, cmap="viridis", edgecolors='black', s=40, linewidths=0.5, alpha=0.7)
        
        plt.title(f"Global Mechanics: {condition} Particles\nMass vs. Charge | Heat: Angle of Repose", fontsize=15)
        plt.xlabel("Total Conglomerate Mass (grams)", fontsize=12)
        plt.ylabel("Voltage Std Dev (Electrostatic Charge)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.4)
        
        output_name = f"Global_Mass_Analysis_{condition}.png"
        plt.savefig(os.path.join(BASE_DIR, output_name))
        print(f"\nSUCCESS: Global plot saved to {os.path.join(BASE_DIR, output_name)}")
        plt.show()

    except Exception as e:
        print(f"Plotting failed: {e}")
        # If contour fails (e.g. not enough variation in mass), fallback to scatter
        plt.scatter(x, y, c=z, cmap="viridis")
        plt.colorbar(label="Angle")
        plt.savefig(os.path.join(BASE_DIR, f"Global_Scatter_{condition}.png"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 global_plotter.py <Dirty/Clean>")
        sys.exit(1)
    
    target_condition = sys.argv[1].capitalize()
    run_global_analysis(target_condition)