import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# ================= CONFIGURATION & STYLING =================
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
FONT_TICK = 14
FONT_LABEL = 16
FONT_TITLE = 18

# Viridis-inspired categorical palette
LINE_STYLES = {
    #'Dirty':        {'color': '#440154', 'marker': 'o', 'ms': 8, 'ls': '-',  'label': 'As-Is'},               
    'Faceplates':   {'color': '#463480', 'marker': 'D', 'ms': 8, 'ls': '-',  'label': 'Same Material Faceplates'},     
    #'Clean':        {'color': '#35b779', 'marker': 's', 'ms': 8, 'ls': '-',  'label': 'Clean (Standard)'},    
    'Clean_Atten':  {'color': '#1e9c89', 'marker': '^', 'ms': 9, 'ls': '--', 'label': 'Clean (Attenuated)'}   
}

MATERIAL_ORDER = ['Acetal', 'Acrylic', 'Nylon', 'Teflon']
# ===========================================================

def apply_clean_axes(ax):
    ax.xaxis.set_tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_tick_params(labelsize=FONT_TICK)
    ax.tick_params('both', length=6, width=1, which='major')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))

def generate_combined_grid(data_dir, output_filename):
    """Generates a 2-row grid for Peak and Std Dev metrics sharing the X axis."""
    
    # Check what materials exist across BOTH metrics
    peak_files = glob.glob(os.path.join(data_dir, "MaxPeak_Voltage_Data_*.csv"))
    std_files = glob.glob(os.path.join(data_dir, "StdDev_Voltage_Data_*.csv"))
    
    if not peak_files or not std_files:
        print("Missing CSV files for one or both metrics. Ensure data is extracted.")
        return

    peak_mats = [os.path.basename(f).replace("MaxPeak_Voltage_Data_", "").replace(".csv", "") for f in peak_files]
    std_mats = [os.path.basename(f).replace("StdDev_Voltage_Data_", "").replace(".csv", "") for f in std_files]
    
    # Only plot materials that we have data for in both sets
    found_materials = set(peak_mats).intersection(std_mats)
    plot_materials = [m for m in MATERIAL_ORDER if m in found_materials]
    
    if not plot_materials:
        print("None of the standard materials were found in both datasets.")
        return

    print(f"Plotting Combined Master Grid for: {', '.join(plot_materials)}...")

    # Create the 2-row figure
    # sharex=True pushes X ticks to the bottom. sharey='row' gives each metric its own scale
    # squeeze=False ensures axes is always a 2D array [row, col] even if there's only 1 material
    fig, axes = plt.subplots(nrows=2, ncols=len(plot_materials), 
                             figsize=(4.5 * len(plot_materials), 8), 
                             sharex=True, sharey='row', squeeze=False)

    metrics = [
        ("MaxPeak_Voltage_Data", "Avg Peak Voltage (V)"),
        ("StdDev_Voltage_Data", "Voltage Std Dev (V)")
    ]

    for row_idx, (prefix, ylabel) in enumerate(metrics):
        
        # Set the Y-label only on the leftmost column of the current row
        axes[row_idx, 0].set_ylabel(ylabel, fontsize=FONT_LABEL)

        for col_idx, material in enumerate(plot_materials):
            ax = axes[row_idx, col_idx]
            csv_path = os.path.join(data_dir, f"{prefix}_{material}.csv")
            df = pd.read_csv(csv_path)
            time_data = df['Time (min)']

            for condition, style in LINE_STYLES.items():
                if condition in df.columns:
                    
                    # Dynamically check for error columns
                    err_col = f"{condition}_Std"
                    if err_col not in df.columns:
                        err_col = f"{condition}_Err"
                    
                    # Plot with error bars if the column exists
                    if err_col in df.columns:
                        ax.errorbar(time_data, df[condition], yerr=df[err_col], 
                                    lw=2, ls=style['ls'], color=style['color'], 
                                    alpha=0.85, label=style['label'],
                                    marker=style['marker'], markerfacecolor=style['color'],
                                    markeredgecolor='black', markeredgewidth=0.5,
                                    markersize=style['ms'], markevery=10, 
                                    errorevery=10, capsize=4, zorder=5)
                    else:
                        ax.plot(time_data, df[condition], 
                                lw=2, ls=style['ls'], color=style['color'], 
                                alpha=0.85, label=style['label'],
                                marker=style['marker'], markerfacecolor=style['color'],
                                markeredgecolor='black', markeredgewidth=0.5,
                                markersize=style['ms'], markevery=10, zorder=5)

            apply_clean_axes(ax)
            ax.set_xlim(0, time_data.max())
            
            # Apply Material Titles to the top row only
            if row_idx == 0:
                ax.set_title(material, fontsize=FONT_TITLE, fontweight='bold', pad=12)

    # Master X-Axis Label at the bottom
    fig.supxlabel("Time (min)", fontsize=FONT_LABEL, y=0.03)

    # Place Legend in the top right of the top-right subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, -1].legend(handles, labels, loc='upper right', fontsize=10, framealpha=0.9)

    # hspace controls the vertical gap between the top and bottom rows
    plt.subplots_adjust(wspace=0.08, hspace=0.1) 
    
    output_path = os.path.join(data_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Exported: {output_path}")

def main(data_dir):
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return

    generate_combined_grid(
        data_dir=data_dir, 
        output_filename="Master_Combined_Voltage_Grid.png"
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graph_grand_voltage.py <path_to_Grand_Voltage_Comparisons_Data_folder>")
    else:
        main(sys.argv[1])