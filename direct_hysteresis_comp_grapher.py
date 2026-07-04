import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib import rcParams
import os

# ================= CONFIGURATION =================
BASE_DIR = r"E:\particle-data\Dirty"
INPUT_FILE = "Volume_Angle_and_Time_Split_Voltage_Summary.csv"
OUTPUT_FILENAME = "Ensemble_Dual_Colored_Bars_Nested_Legend.png"

# Global Font Setup
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
rcParams['font.size'] = 12
# =================================================

def generate_multi_volume_plot():
    csv_path = os.path.join(BASE_DIR, INPUT_FILE)
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {INPUT_FILE} in {BASE_DIR}")
        return

    df = pd.read_csv(csv_path)
    
    volumes = sorted(df['Volume'].unique())
    materials = sorted(df['Material'].unique())
    
    fig, axes = plt.subplots(1, len(volumes), figsize=(16, 6), sharey=True)
    if len(volumes) == 1: axes = [axes]
    
    # --- UNIFIED MIN/MAX FOR COLOR SCALING ---
    global_min_volt = min(df['Start_CH2_Volt'].min(), df['End_CH2_Volt'].min())
    global_max_volt = max(df['Start_CH2_Volt'].max(), df['End_CH2_Volt'].max())
    global_max_angle = max(df['Start_Angle_1RPM'].max(), df['End_Angle_1RPM'].max())
    
    cmap = plt.get_cmap('viridis') 
    norm = mcolors.Normalize(vmin=global_min_volt, vmax=global_max_volt)

    x = np.arange(len(materials))
    bar_width = 0.35

    # --- PLOTTING LOOP ---
    for ax, vol in zip(axes, volumes):
        vol_df = df[df['Volume'] == vol].set_index('Material')
        
        start_angles, end_angles = [], []
        start_colors, end_colors = [], []
        
        for mat in materials:
            if mat in vol_df.index:
                start_angles.append(vol_df.loc[mat, 'Start_Angle_1RPM'])
                end_angles.append(vol_df.loc[mat, 'End_Angle_1RPM'])
                
                # Fetch specific colors for start and end blocks
                start_colors.append(cmap(norm(vol_df.loc[mat, 'Start_CH2_Volt'])))
                end_colors.append(cmap(norm(vol_df.loc[mat, 'End_CH2_Volt'])))
            else:
                start_angles.append(0)
                end_angles.append(0)
                start_colors.append('white')
                end_colors.append('white')
                
        # Bar 1: Start Angle (Colored by Start Voltage)
        ax.bar(x - bar_width/2, start_angles, bar_width, 
               color=start_colors, edgecolor='black', linewidth=1.5, zorder=3)
               
        # Bar 2: End Angle (Colored by End Voltage)
        ax.bar(x + bar_width/2, end_angles, bar_width, 
               color=end_colors, edgecolor='black', linewidth=1.5, zorder=3)

        # Subplot Formatting (Titles are strictly the numbers now)
        ax.set_title(f'{vol}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(materials, rotation=45, ha='right', fontsize=12)
        ax.grid(True, axis='y', color='gray', alpha=0.2, linestyle='-', zorder=2)

    # --- GLOBAL FIGURE FORMATTING ---
    axes[0].set_ylabel("Angle of Repose (deg)", fontsize=14)
    axes[0].set_ylim(0, global_max_angle * 1.1) 
    
    # 1. The Colorbar 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, aspect=30)
    cbar.set_label('Top 10% Peak Voltage', fontsize=14, rotation=270, labelpad=20)
    
    # 2. Nested Legend (Placed directly into the last subplot)
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='black', linewidth=1.5, label='Left Bar: First 5 mins'),
        Patch(facecolor='gray', edgecolor='black', linewidth=1.5, label='Right Bar: Last 5 mins')
    ]

    # axes[-1] targets the last graph in the loop (the 1000 volume graph)
    axes[-1].legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)

    output_path = os.path.join(BASE_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"--- SUCCESS ---")
    print(f"Saved dual-colored bar chart with nested legend to: {output_path}")

if __name__ == "__main__":
    generate_multi_volume_plot()