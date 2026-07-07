import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import sys
import warnings

# --- ACADEMIC PLOT SETTINGS ---
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
FONT_TICK = 16
FONT_LABEL = 18
FONT_TITLE = 20

# Define the folders and their bolded subplot titles
CONDITION_MAP = {
    "Processed_Data_Exports_Dirty": "As-Is",
    "Processed_Data_Exports_Cleaned_once": "Cleaned Once",
    "Processed_Data_Exports_Cleaned_Thrice": "Cleaned Thrice"
}

TIME_POINTS = ["t0", "day-1", "day-7"]
TITLE_MAP = {"t0": r"$t_0$", "day-1": "Day 1", "day-7": "Day 7"}

HARD_CUTOFF_MIN = 60.0

def apply_academic_axes(ax, xlabel="", ylabel=""):
    ax.xaxis.set_tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_tick_params(labelsize=FONT_TICK)
    ax.tick_params('both', length=7, width=1, which='major')
    if xlabel: ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel: ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.grid(True, alpha=0.3)

def load_combined_data(base_path, file_name):
    """Loads specific data (voltage or angle) from the three condition folders."""
    all_data = []
    
    for folder_name, condition_label in CONDITION_MAP.items():
        csv_path = os.path.join(base_path, folder_name, file_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["condition"] = condition_label
            all_data.append(df)
        else:
            print(f"Warning: Could not find {csv_path}")
            
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def get_binned_line(df, y_col, time_bins):
    """Helper function to bin data by minute and return X and Y arrays."""
    means, valid_x = [], []
    for j in range(len(time_bins) - 1):
        mask = (df['time_min'] >= time_bins[j]) & (df['time_min'] < time_bins[j+1])
        v_data = df.loc[mask, y_col]
        if not v_data.empty:
            means.append(v_data.mean())
            valid_x.append(time_bins[j] + 0.5)
    return np.array(valid_x), np.array(means)

def plot_1x3_shaded_conditions(df, output_dir, y_col, y_label, metric_name):
    """Plots a 1x3 grid per material: Subplots = Conditions, Lines = Time Points."""
    if df.empty:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    df['material'] = df['material'].str.title()
    materials = sorted(df['material'].unique())
    
    # Setup Viridis colors for t0, day-1, day-7
    cmap = plt.cm.viridis
    colors = dict(zip(TIME_POINTS, cmap(np.linspace(0.15, 0.85, len(TIME_POINTS)))))
    time_bins = np.arange(0, HARD_CUTOFF_MIN + 1, 1)
    
    for mat in materials:
        # 1x3 grid for the current material
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
        
        mat_df = df[df['material'] == mat]
        
        for i, cond_title in enumerate(CONDITION_MAP.values()):
            ax = axs[i]
            # Bolded subplot title, NO overall figure title
            ax.set_title(cond_title, fontsize=FONT_TITLE, pad=10, fontweight='bold')
            
            cond_df = mat_df[mat_df['condition'] == cond_title]
            
            for tp in TIME_POINTS:
                tp_df = cond_df[cond_df['time_point'] == tp]
                if tp_df.empty:
                    continue
                
                # --- 1. PLOT INDIVIDUAL TRIAL LINES ---
                # Loop through each unique trial and plot it semi-transparently
                for trial_id in tp_df['trial_id'].unique():
                    trial_df = tp_df[tp_df['trial_id'] == trial_id]
                    x_trial, y_trial = get_binned_line(trial_df, y_col, time_bins)
                    
                    if len(x_trial) > 0:
                        ax.plot(x_trial, y_trial, lw=1.5, color=colors[tp], alpha=0.4)

                # --- 2. CALCULATE OVERALL MEAN AND STD ---
                # We calculate standard deviation across the individual raw data points per bin
                binned_means, binned_stds, valid_x = [], [], []
                for j in range(len(time_bins) - 1):
                    mask = (tp_df['time_min'] >= time_bins[j]) & (tp_df['time_min'] < time_bins[j+1])
                    v_data = tp_df.loc[mask, y_col]
                    
                    if not v_data.empty:
                        binned_means.append(v_data.mean())
                        binned_stds.append(v_data.std() if pd.notna(v_data.std()) else 0) 
                        valid_x.append(time_bins[j] + 0.5)
                
                if not valid_x:
                    continue

                x = np.array(valid_x)
                mean = np.array(binned_means)
                std = np.array(binned_stds)
                
                label_name = TITLE_MAP.get(tp, tp)
                
                # --- 3. PLOT OVERALL MEAN AND SHADING ---
                # Thick opaque main line
                ax.plot(x, mean, lw=3.5, color=colors[tp], label=label_name)
                # Faint background shading
                ax.fill_between(x, mean - std, mean + std, color=colors[tp], alpha=0.15, edgecolor='none')

            # Formatting
            apply_academic_axes(ax)
            ax.set_xlim(0, HARD_CUTOFF_MIN)
            
            # Ensure the angle graph's Y-axis doesn't float in negative space
            if metric_name == "Angle":
                current_ymin, current_ymax = ax.get_ylim()
                ax.set_ylim(min(0, current_ymin), current_ymax)
            
            # Y-axis label only on the left
            if i == 0:
                ax.set_ylabel(y_label, fontsize=FONT_LABEL)
                
            # Legend only on the right
            if i == 2:
                ax.legend(frameon=False, fontsize=14, loc='best')

        # Single X-axis label centered at the bottom
        fig.supxlabel("Time (min)", fontsize=FONT_LABEL)
        
        save_path = os.path.join(output_dir, f"{mat}_1x3_Conditions_{metric_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plotter.py <PATH_TO_PARENT_DIR_OF_EXPORTS>")
        sys.exit(1)
        
    base_dir = sys.argv[1]
    output_dir = os.path.join(base_dir, "Combined_1x3_Graphs")
    
    print("Loading exported data...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # --- PLOT VOLTAGE ---
        vol_df = load_combined_data(base_dir, "timeseries_voltage.csv")
        if not vol_df.empty:
            print("\nGenerating Voltage Graphs...")
            plot_1x3_shaded_conditions(vol_df, output_dir, "voltage_v", "Voltage (V)", "Voltage")
            
        # --- PLOT ANGLE ---
        ang_df = load_combined_data(base_dir, "timeseries_angle.csv")
        if not ang_df.empty:
            print("\nGenerating Angle Graphs...")
            plot_1x3_shaded_conditions(ang_df, output_dir, "angle_deg", "Angle (deg)", "Angle")
            
    print("\nDone!")