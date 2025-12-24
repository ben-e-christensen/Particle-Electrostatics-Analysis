#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import sys, os

# === CONFIGURATION ===
# Which variable to plot as the DOTS?
# Options: "charge_std" (Noise) or "charge_mag" (Top %)
DOT_METRIC = "charge_std" 
DOT_LABEL = "Charge Signal (Std Dev) ●"

# --- Parse argument ---
if len(sys.argv) < 2:
    print("Usage: python3 compare_runs.py <Path/To/Material/RPM_Group>")
    sys.exit(1)

target_dir = sys.argv[1]

if not os.path.isdir(target_dir):
    print(f"❌ Error: {target_dir} is not a directory")
    sys.exit(1)

material_name = os.path.basename(os.path.normpath(target_dir))
print(f"📊 Comparing Runs in: {material_name}")

# --- FIND DATA ---
subdirs = [f.path for f in os.scandir(target_dir) if f.is_dir()]
all_data = []

for folder in subdirs:
    summary_file = os.path.join(folder, "aggregated_results", "summary_stats.csv")
    
    if os.path.exists(summary_file):
        folder_name = os.path.basename(folder)
        try:
            duration = int(''.join(filter(str.isdigit, folder_name)))
        except ValueError:
            duration = 999 
            
        df = pd.read_csv(summary_file)
        df["Duration"] = folder_name
        df["Duration_Num"] = duration
        all_data.append(df)

if not all_data:
    print("❌ No data found. (Did you run the updated main.py first?)")
    sys.exit(1)

master_df = pd.concat(all_data, ignore_index=True)
master_df = master_df.sort_values(["Duration_Num", "motor_speed"])

# =========================================================
# 📊 PLOTTING LOGIC
# =========================================================

# Get unique speeds across ALL runs to ensure alignment
all_speeds = np.sort(master_df["motor_speed"].unique())
durations = master_df["Duration"].unique()
n_groups = len(durations)
bar_width = 0.8 / n_groups 
colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_groups))

# Create the Stacked Figure
fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharey='row')

def plot_subplot(ax, direction, title):
    subset = master_df[master_df["direction"] == direction].copy()
    if subset.empty: return

    ax2 = ax.twinx() # Secondary axis for Charge

    for i, (duration, color) in enumerate(zip(durations, colors)):
        group_data = subset[subset["Duration"] == duration]
        
        # Calculate offset positions
        x_positions = []
        for s in group_data["motor_speed"]:
            # Find index of this speed in the master list
            idx = np.where(all_speeds == s)[0][0]
            x_positions.append(idx + (i - (n_groups - 1) / 2) * bar_width)
        
        # 1. BARS (Angle)
        ax.bar(x_positions, group_data["mean"], 
               width=bar_width, color=color, edgecolor="black", alpha=0.85, label=duration)
        
        # 2. DOTS (Charge)
        ax2.plot(x_positions, group_data[DOT_METRIC], 
                 'o', color='white', markeredgecolor=color, markeredgewidth=2, markersize=8, zorder=10)

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Angle (°)", fontsize=12)
    ax.set_ylim(0, 50) # Angle Limit
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Right Axis Formatting
    ax2.set_ylabel(DOT_LABEL, fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    # Optional: Fix charge scale so top/bottom match
    # ax2.set_ylim(0, 0.05) 

    # Set X Ticks (Use master list indices)
    ax.set_xticks(np.arange(len(all_speeds)))
    ax.set_xticklabels(all_speeds)

    return ax, ax2

# --- Plot Top (Forward) ---
ax_top, ax2_top = plot_subplot(axes[0], "Forward", "⬆ FORWARD SWEEP (Spin Up)")

# --- Plot Bottom (Backward) ---
ax_bot, ax2_bot = plot_subplot(axes[1], "Backward", "⬇ BACKWARD SWEEP (Spin Down)")

# --- Global Layout ---
axes[1].set_xlabel("Motor Speed (RPM)", fontsize=12, fontweight='bold')
plt.suptitle(f"Hysteresis & Charge Levels: {material_name}", fontsize=16, y=0.98)

# --- Unified Legend (Placed on Top Right) ---
# Create custom handles
legend_handles = [mpatches.Patch(facecolor='gray', edgecolor='black', label='Bar Height = Angle')]
legend_handles.append(mlines.Line2D([], [], color='white', marker='o', markeredgecolor='black', markersize=8, label='Dot Height = Charge'))
legend_handles.append(mpatches.Patch(color='none', label='Duration (mins):')) # Spacer

for dur, col in zip(durations, colors):
    legend_handles.append(mpatches.Patch(facecolor=col, edgecolor='black', label=dur))

fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.95, 0.95), 
           ncol=1, frameon=True, fontsize=10)

# Save
out_file = os.path.join(target_dir, "Combined_Comparison.png")
plt.tight_layout(rect=[0, 0, 0.88, 0.96]) # Make room for legend
plt.savefig(out_file)
print(f"🚀 Saved Stacked Plot: {out_file}")