#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import re

# === CONFIGURATION ===
DEFAULT_PATH = "." 

def parse_duration(folder_name):
    """Extracts duration number from folder name (e.g., '...-12mins')."""
    match = re.search(r"-(\d+)mins$", folder_name)z
    if match:
        return int(match.group(1))
    return None

# === 1. SETUP PATHS ===
if len(sys.argv) > 1:
    target_dir = sys.argv[1]
else:
    target_dir = DEFAULT_PATH

if not os.path.isdir(target_dir):
    print(f"❌ Error: Directory not found: {target_dir}")
    sys.exit(1)

print(f"📂 Scanning directory: {target_dir}")

# === 2. COLLECT DATA ===
all_fwd = []
all_bwd = []
found_durations = []

for item in os.listdir(target_dir):
    full_path = os.path.join(target_dir, item)
    
    if os.path.isdir(full_path):
        duration = parse_duration(item)
        
        if duration is not None:
            angle_csv_path = os.path.join(full_path, "csv_files", "speed_run_summary.csv")
            charge_csv_path = os.path.join(full_path, "csv_files", "charge_analysis.csv")
            
            if os.path.isfile(angle_csv_path):
                print(f"   Found {duration} mins run: {item}")
                
                # Load Angle Data & Fix missing run_id
                df_angle = pd.read_csv(angle_csv_path)
                if 'run_id' not in df_angle.columns:
                    df_angle['run_id'] = range(1, len(df_angle) + 1)

                # Load Charge Data
                if os.path.isfile(charge_csv_path):
                    df_charge = pd.read_csv(charge_csv_path)
                    if 'run_id' not in df_charge.columns:
                        df_charge['run_id'] = range(1, len(df_charge) + 1)
                    
                    # Merge
                    df = pd.merge(df_angle, df_charge[['run_id', 'ch2_std']], on='run_id', how='left')
                else:
                    print(f"      ⚠️ Warning: No charge analysis found for {item}")
                    df = df_angle
                    df['ch2_std'] = np.nan 

                # --- DETECT FORWARD VS BACKWARD ---
                if "start_index" in df.columns:
                    df = df.sort_values("start_index")
                    df['sweep_rank'] = df.groupby("motor_speed")["start_index"].rank(method="first")
                else:
                    df = df.sort_values("run_id")
                    df['sweep_rank'] = df.groupby("motor_speed")["run_id"].rank(method="first")

                # Split
                df_fwd = df[df['sweep_rank'] == 1].copy()
                df_bwd = df[df['sweep_rank'] == 2].copy()
                
                # Add duration tag
                df_fwd["duration"] = duration
                df_bwd["duration"] = duration
                
                cols = ["motor_speed", "angle_mean", "ch2_std", "duration"]
                all_fwd.append(df_fwd[cols])
                all_bwd.append(df_bwd[cols])
                
                found_durations.append(duration)

if not all_fwd:
    print("❌ No valid data found.")
    sys.exit(1)

# === 3. AGGREGATE & PIVOT ===
def create_pivots(data_list):
    if not data_list: return pd.DataFrame(), pd.DataFrame()
    combined = pd.concat(data_list)
    
    p_angle = combined.groupby(["motor_speed", "duration"])["angle_mean"].mean().reset_index()
    p_angle = p_angle.pivot(index="motor_speed", columns="duration", values="angle_mean").sort_index(axis=1)
    
    p_charge = combined.groupby(["motor_speed", "duration"])["ch2_std"].mean().reset_index()
    p_charge = p_charge.pivot(index="motor_speed", columns="duration", values="ch2_std").sort_index(axis=1)
    
    return p_angle, p_charge

pivot_angle_fwd, pivot_charge_fwd = create_pivots(all_fwd)
pivot_angle_bwd, pivot_charge_bwd = create_pivots(all_bwd)

print(f"✅ Data Aggregated. Durations found: {found_durations}")

# === 4. PLOT ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

sorted_durations = sorted(list(set(found_durations)))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_durations)))
color_map = {dur: col for dur, col in zip(sorted_durations, colors)}

# --- HELPER: Overlay Dots on Bars ---
def plot_dots_on_bars(ax_primary, pivot_charge_data, duration_columns):
    # Create Twin Axis
    ax_secondary = ax_primary.twinx()
    
    # ax_primary.containers contains the groups of bars.
    # container[0] = all bars for 12mins, container[1] = all bars for 60mins, etc.
    
    for i, duration in enumerate(duration_columns):
        # 1. Get the exact X-centers of the bars for this duration
        bars = ax_primary.containers[i]
        x_centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        
        # 2. Get the corresponding Charge Values
        y_charges = pivot_charge_data[duration].values
        
        # 3. Plot Dots
        # We use a white facecolor with colored edge to make it stand out against the colored bar
        col = color_map.get(duration, 'black')
        ax_secondary.scatter(
            x_centers, 
            y_charges, 
            s=60,            # Size of dot
            facecolors='white', 
            edgecolors=col, 
            linewidth=2,
            zorder=10,       # Ensure it sits on top
            label=f"{duration}min Charge" if i==0 else "" # Simple label logic
        )

    # Styling the Secondary Axis
    ax_secondary.set_ylabel("Charge Signal (Std Dev) ●", fontsize=11, fontweight='bold', rotation=270, labelpad=20)
    return ax_secondary

# --- Plot 1: Forward ---
if not pivot_angle_fwd.empty:
    pivot_angle_fwd.plot(kind='bar', ax=ax1, color=colors, width=0.8, edgecolor='black', alpha=0.85)
    
    if not pivot_charge_fwd.empty:
        plot_dots_on_bars(ax1, pivot_charge_fwd, pivot_angle_fwd.columns)

    ax1.set_title("⬆️ FORWARD SWEEP (Spin Up)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Mean Angle (°)", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(title="Duration (mins)", bbox_to_anchor=(1.08, 1), loc='upper left')

# --- Plot 2: Backward ---
if not pivot_angle_bwd.empty:
    pivot_angle_bwd.plot(kind='bar', ax=ax2, color=colors, width=0.8, edgecolor='black', alpha=0.85)
    
    if not pivot_charge_bwd.empty:
        plot_dots_on_bars(ax2, pivot_charge_bwd, pivot_angle_bwd.columns)

    ax2.set_title("⬇️ BACKWARD SWEEP (Spin Down)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Mean Angle (°)", fontsize=10)
    ax2.set_xlabel("Motor Speed (RPM)", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.get_legend().remove() 

# --- Formatting ---
parent_name = os.path.basename(os.path.normpath(target_dir))
plt.suptitle(f"Hysteresis & Charge Levels: {parent_name}", fontsize=14, y=0.98)
plt.xticks(rotation=0)
plt.tight_layout()

# === 5. SAVE ===
output_filename = f"comparison_dots_{parent_name}.png"
save_path = os.path.join(target_dir, output_filename)
plt.savefig(save_path)
print(f"📊 Graph saved to: {save_path}")