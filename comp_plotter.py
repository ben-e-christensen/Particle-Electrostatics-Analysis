import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ================= CONFIGURATION & STYLING =================
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
FONT_TICK = 12
FONT_LABEL = 14
FONT_TITLE = 16

COLOR_PALETTE = {
    'Clean': {
        'Increasing': '#73c2b6',  # lighter tint of Decreasing (viridis t=0.55)
        'Decreasing': '#1e9c89',  # viridis t=0.55
    },
    'As-Is': {
        'Increasing': '#8c81b0',  # lighter tint of Decreasing (viridis t=0.15)
        'Decreasing': '#463480',  # viridis t=0.15
    }
}

LINE_STYLES = {
    'Clean': '--',   # Dashed for Clean
    'As-Is': '-'     # Solid for As-Is
}

# Define the exact order of materials for the columns
MATERIAL_ORDER = ['Acetal', 'Acrylic', 'Nylon', 'Teflon']
# ===========================================================

def apply_clean_axes(ax):
    ax.xaxis.set_tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)

def load_data(folder_path, prefix, file_suffix):
    if not os.path.exists(folder_path):
        return pd.DataFrame()
    file_path = os.path.join(folder_path, f"{prefix}_{file_suffix}")
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return pd.DataFrame()

def draw_hyst_lines(ax_obj, df, y_mean, y_err, state_lbl):
    """Draws hysteresis lines on a specific subplot axis."""
    if df.empty: return
    line_style = LINE_STYLES[state_lbl]
    for direct in ['Increasing', 'Decreasing']:
        d_data = df[df['Direction'] == direct].sort_values('Motor Speed (RPM)')
        if not d_data.empty:
            color = COLOR_PALETTE[state_lbl][direct]
            dir_lbl = "Accel" if direct == 'Increasing' else "Decel"
            label_string = f"{state_lbl} ({dir_lbl})"
            
            # Triangles (^) for Accel, Squares (s) for Decel
            marker_shape = '^' if direct == 'Increasing' else 's'
            
            # Black edges to keep the lighter Accel colors crisp against the background
            ax_obj.errorbar(d_data['Motor Speed (RPM)'], d_data[y_mean], yerr=d_data[y_err],
                            fmt=marker_shape, linestyle=line_style, color=color,
                            capsize=4, lw=2, markersize=7, label=label_string,
                            markeredgecolor='black', markeredgewidth=0.5)

def main(base_dir):
    normalized_path = base_dir.replace('\\', '/').rstrip('/')
    if not os.path.exists(normalized_path):
        print(f"Error: The directory {normalized_path} does not exist.")
        return

    graphs_dir = os.path.join(normalized_path, "Graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Validate materials exist in the folder structure
    available_materials = set()
    for item in os.listdir(normalized_path):
        if os.path.isdir(os.path.join(normalized_path, item)) and "_Volume_500_Data" in item:
            if "-" in item:
                mat = item.split("_Volume_500_Data")[0].split("-")[1]
                available_materials.add(mat)

    # Filter our desired order to only include materials we actually have data for
    plot_materials = [m for m in MATERIAL_ORDER if m in available_materials]

    if not plot_materials:
        print("No valid material folders found in the target directory.")
        return

    print(f"Processing Master Grid for: {', '.join(plot_materials)}...")

    fig, axes = plt.subplots(nrows=2, ncols=len(plot_materials), 
                             figsize=(16, 8), 
                             sharex=True, sharey='row')
    
    # Ensure axes is always a 2D array
    if len(plot_materials) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, material in enumerate(plot_materials):
        clean_folder = os.path.join(normalized_path, f"Clean-{material}_Volume_500_Data")
        dirty_folder = os.path.join(normalized_path, f"Dirty-{material}_Volume_500_Data")
        clean_prefix = f"clean_{material.lower()}"
        dirty_prefix = f"dirty_{material.lower()}"

        df_clean = load_data(clean_folder, clean_prefix, "hysteresis_summary_500.csv")
        df_dirty = load_data(dirty_folder, dirty_prefix, "hysteresis_summary_500.csv")

        # Top Row (Row 0): Voltage Std Dev
        ax_top = axes[0, col_idx]
        draw_hyst_lines(ax_top, df_clean, 'Voltage_Std_Mean', 'Voltage_Std_Err', 'Clean')
        draw_hyst_lines(ax_top, df_dirty, 'Voltage_Std_Mean', 'Voltage_Std_Err', 'As-Is')
        apply_clean_axes(ax_top)
        
        ax_top.set_title(material, fontsize=FONT_TITLE, fontweight='bold', pad=15)
        
        # Bottom Row (Row 1): Mean Angle
        ax_bot = axes[1, col_idx]
        draw_hyst_lines(ax_bot, df_clean, 'Angle_Mean', 'Angle_Std', 'Clean')
        draw_hyst_lines(ax_bot, df_dirty, 'Angle_Mean', 'Angle_Std', 'As-Is')
        apply_clean_axes(ax_bot)
        
        # Explicit X-axis ticks
        ax_bot.set_xticks([1, 5, 10, 15, 20, 25])

    # Add Y-axis labels only to the first column
    axes[0, 0].set_ylabel("Voltage Std Dev (V)", fontsize=FONT_LABEL)
    axes[1, 0].set_ylabel("Angle of Repose (deg)", fontsize=FONT_LABEL)

    # Single Master X-Axis Label
    fig.supxlabel("Motor Speed (RPM)", fontsize=FONT_LABEL, y=0.04)

    # Legend targets the top right graph
    axes[0, -1].legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Adjust layout
    plt.subplots_adjust(wspace=0.15, hspace=0.1)
    
    # Save the master grid
    output_path = os.path.join(graphs_dir, "Master_Hysteresis_Grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Master grid successfully exported to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python comp_plotter.py <path_to_comparison_data_folder>")
    else:
        main(sys.argv[1])