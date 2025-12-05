import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os

DOT_METRIC = "charge_std" 
DOT_LABEL = "Charge Signal (Std Dev) ●"

def run_comparison(target_dir):
    """
    Looks inside an RPM folder (e.g. '1000'), finds '12mins', '60mins', etc., 
    and generates the stacked comparison plot.
    """
    if not os.path.isdir(target_dir): return

    material_name = os.path.basename(os.path.normpath(target_dir))
    print(f"   📊 Comparing Runs in: {material_name}")

    subdirs = [f.path for f in os.scandir(target_dir) if f.is_dir()]
    all_data = []

    for folder in subdirs:
        summary_file = os.path.join(folder, "aggregated_results", "summary_stats.csv")
        if os.path.exists(summary_file):
            folder_name = os.path.basename(folder)
            try:
                # Extract number for sorting (12 from "12mins")
                duration = int(''.join(filter(str.isdigit, folder_name)))
            except ValueError:
                duration = 999 
            
            df = pd.read_csv(summary_file)
            df["Duration"] = folder_name
            df["Duration_Num"] = duration
            all_data.append(df)

    if not all_data:
        print("      ⚠️ No valid summary files found.")
        return

    master_df = pd.concat(all_data, ignore_index=True)
    master_df = master_df.sort_values(["Duration_Num", "motor_speed"])

    # Plot Setup
    all_speeds = np.sort(master_df["motor_speed"].unique())
    durations = master_df["Duration"].unique()
    n_groups = len(durations)
    bar_width = 0.8 / n_groups 
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_groups))

    fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharey='row')

    def plot_subplot(ax, direction, title):
        subset = master_df[master_df["direction"] == direction].copy()
        if subset.empty: return
        ax2 = ax.twinx() 

        for i, (duration, color) in enumerate(zip(durations, colors)):
            group_data = subset[subset["Duration"] == duration]
            x_positions = []
            for s in group_data["motor_speed"]:
                idx = np.where(all_speeds == s)[0][0]
                x_positions.append(idx + (i - (n_groups - 1) / 2) * bar_width)
            
            ax.bar(x_positions, group_data["mean"], width=bar_width, color=color, edgecolor="black", alpha=0.85)
            ax2.plot(x_positions, group_data[DOT_METRIC], 'o', color='white', markeredgecolor=color, markeredgewidth=2, markersize=8, zorder=10)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Mean Angle (°)", fontsize=12)
        ax.set_ylim(0, 50) 
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax2.set_ylabel(DOT_LABEL, fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        ax.set_xticks(np.arange(len(all_speeds)))
        ax.set_xticklabels(all_speeds)

    plot_subplot(axes[0], "Forward", "⬆ FORWARD SWEEP")
    plot_subplot(axes[1], "Backward", "⬇ BACKWARD SWEEP")

    axes[1].set_xlabel("Motor Speed (RPM)", fontsize=12, fontweight='bold')
    plt.suptitle(f"Hysteresis & Charge: {material_name}", fontsize=16)

    # Legend
    handles = [mpatches.Patch(color='gray', label='Bar=Angle'), mlines.Line2D([],[], marker='o', color='w', markeredgecolor='k', label='Dot=Charge')]
    for dur, col in zip(durations, colors): handles.append(mpatches.Patch(color=col, label=dur))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    out_file = os.path.join(target_dir, "Combined_Comparison.png")
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.savefig(out_file)
    plt.close()
    print(f"      ✅ Saved Stacked Plot: {out_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: run_comparison(sys.argv[1])