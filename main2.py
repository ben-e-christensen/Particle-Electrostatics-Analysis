import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
import sys
import os
import math

# ================= CONFIGURATION =================
BASE_DIR = "F:/particle-data/Dirty"
SAMPLE_RATE = 100 
SPEED_ROUNDING = 0 
MIN_SECONDS_PER_BIN = 30 
FIXED_PROMINENCE = 1.5

# Markers for "Scatter by Speed" (Trial distinction)
TRIAL_MARKERS = ['o', 's', '^', 'D', 'X', 'P', '*', 'v']

# Markers for "Global Summary" (Material distinction)
MATERIAL_MARKERS = {
    'acrylic': 'o',  # Circle
    'acetal': 's',   # Square
    'nylon': '^',    # Triangle
    'teflon': 'D',   # Diamond
    'unknown': 'X'
}

# Plotting Constants
FONT_TICK = 18
FONT_LABEL = 20
FONT_TITLE = 22
# =================================================

def parse_metadata_from_path(rel_path):
    parts = rel_path.replace('\\', '/').strip('/').split('/')
    vol_keys = ["500", "750", "1000"]
    
    for i, part in enumerate(parts):
        if part in vol_keys:
            volume = part
            material = parts[i-1] if i > 0 else "Unknown"
            trial = parts[i+1] if i < len(parts) - 1 else "Unknown"
            return material.lower(), volume, trial
            
    return "Unknown", "Unknown", "Unknown"

def load_all_data(base_dir):
    print(f"Scanning {base_dir}...")
    all_data = []
    
    for root, dirs, files in os.walk(base_dir):
        if "experiment_log.csv" in files:
            csv_path = os.path.join(root, "experiment_log.csv")
            rel_path = os.path.relpath(root, base_dir)
            material, volume, trial = parse_metadata_from_path(rel_path)
            
            if volume == "Unknown": continue

            try:
                cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
                        "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
                        "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
                        "ch2_flag", "ch3_flag"]
                df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
                
                if df.empty or "motor_speed" not in df.columns: continue
                
                max_val = df["motor_speed"].max()
                max_indices = df.index[df["motor_speed"] == max_val].tolist()
                if not max_indices: continue
                mid_idx = max_indices[len(max_indices) // 2]
                df["direction"] = "Increasing"
                df.loc[mid_idx + 1:, "direction"] = "Decreasing"
                
                t0 = df["timestamp"].iloc[0]
                df["minute_bin"] = ((df["timestamp"] - t0) / 60).astype(int) + 1
                
                peak_indices, _ = find_peaks(df["ellipse_angle_deg"], height=20, prominence=FIXED_PROMINENCE)
                angle_df = df.iloc[peak_indices].copy()
                
                charge_agg = df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                    voltage_std=("CH2_volts", "std"), 
                    count=("CH2_volts", "count")
                ).reset_index()
                
                angle_agg = angle_df.groupby(["minute_bin", "direction", "motor_speed"]).agg(
                    angle_mean=("ellipse_angle_deg", "mean")
                ).reset_index()
                
                merged = pd.merge(charge_agg, angle_agg, on=["minute_bin", "direction", "motor_speed"])
                merged["material"] = material
                merged["volume"] = volume
                merged["trial"] = trial
                merged["grouped_speed"] = merged["motor_speed"].round(SPEED_ROUNDING).astype(int)
                
                merged = merged[merged["count"] > (SAMPLE_RATE * MIN_SECONDS_PER_BIN)]
                
                all_data.append(merged)
                print(f"  Loaded: {material} {volume} - {trial}")
                
            except Exception as e:
                print(f"  Error loading {rel_path}: {e}")

    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

# ======================================================
# HELPER: FIXED AXES
# ======================================================
def get_global_limits(df, x_col, y_col):
    if df.empty: return (0,1), (0,1)
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    
    x_buff = (x_max - x_min) * 0.1 if x_max != x_min else 1
    y_buff = (y_max - y_min) * 0.1 if y_max != y_min else 1
    
    return (x_min - x_buff, x_max + x_buff), (y_min - y_buff, y_max + y_buff)

# ======================================================
# PLOTTING FUNCTIONS
# ======================================================

def plot_intra_material_hysteresis(df, output_dir):
    sub_dir = os.path.join(output_dir, "Intra_Material_Comparison")
    os.makedirs(sub_dir, exist_ok=True)
    
    materials = sorted(df["material"].unique())
    volumes = ["500", "750", "1000"]
    metrics = [("angle_mean", "Angle of Repose (deg)"), ("voltage_std", "Voltage Std Dev (V)")]

    for metric_col, metric_label in metrics:
        for mat in materials:
            mat_df = df[df["material"] == mat]
            if mat_df.empty: continue
            
            # Local limits for consistent 1x3
            y_lims = (mat_df[metric_col].min()*0.9, mat_df[metric_col].max()*1.1) 
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
            fig.suptitle(f"{mat.title()} - {metric_label} Hysteresis", fontsize=FONT_TITLE, fontweight='bold')
            
            for i, vol in enumerate(volumes):
                ax = axes[i]
                vol_df = mat_df[mat_df["volume"] == vol]
                
                if vol_df.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    ax.axis('off'); continue

                stats = vol_df.groupby(['grouped_speed', 'direction'])[metric_col].agg(['mean', 'std']).reset_index()
                inc = stats[stats['direction'] == 'Increasing']
                dec = stats[stats['direction'] == 'Decreasing']
                
                ax.errorbar(inc['grouped_speed'], inc['mean'], yerr=inc['std'], fmt='s-', color='grey', label='Increasing', capsize=4, lw=2)
                ax.errorbar(dec['grouped_speed'], dec['mean'], yerr=dec['std'], fmt='^-', color='black', label='Decreasing', capsize=4, lw=2)
                
                ax.set_title(f"Volume: {vol}", fontsize=FONT_LABEL, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
                ax.set_ylim(y_lims)
                
                if i == 0: ax.set_ylabel(metric_label, fontsize=FONT_LABEL)
                else: ax.tick_params(labelleft=False)

            fig.supxlabel("Motor Speed (RPM)", fontsize=FONT_LABEL, fontweight='bold')
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=14)
            plt.savefig(os.path.join(sub_dir, f"{mat}_{metric_col}_Intra.png"), bbox_inches='tight')
            plt.close()

def plot_inter_material_hysteresis(df, output_dir):
    sub_dir = os.path.join(output_dir, "Inter_Material_Comparison")
    os.makedirs(sub_dir, exist_ok=True)
    
    volumes = sorted(df["volume"].unique())
    materials = sorted(df["material"].unique())
    metrics = [("angle_mean", "Angle of Repose (deg)"), ("voltage_std", "Voltage Std Dev (V)")]

    for metric_col, metric_label in metrics:
        for vol in volumes:
            vol_df = df[df["volume"] == vol]
            if vol_df.empty: continue

            # Volume-wide limits
            y_min, y_max = vol_df[metric_col].min(), vol_df[metric_col].max()
            buff = (y_max - y_min)*0.1 if y_max!=y_min else 1
            FIXED_YLIM = (y_min - buff, y_max + buff)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
            fig.suptitle(f"{vol} Volume - {metric_label} Comparison", fontsize=FONT_TITLE, fontweight='bold')
            axes_flat = axes.flatten()
            
            for i, mat in enumerate(materials):
                if i >= 4: break
                ax = axes_flat[i]
                m_df = vol_df[vol_df["material"] == mat]
                
                if m_df.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    ax.set_title(mat.title(), fontsize=FONT_LABEL); continue
                
                stats = m_df.groupby(['grouped_speed', 'direction'])[metric_col].agg(['mean', 'std']).reset_index()
                inc = stats[stats['direction'] == 'Increasing']
                dec = stats[stats['direction'] == 'Decreasing']
                
                ax.errorbar(inc['grouped_speed'], inc['mean'], yerr=inc['std'], fmt='s-', color='grey', label='Increasing', capsize=4, lw=2)
                ax.errorbar(dec['grouped_speed'], dec['mean'], yerr=dec['std'], fmt='^-', color='black', label='Decreasing', capsize=4, lw=2)
                
                ax.set_title(mat.title(), fontsize=FONT_LABEL, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
                ax.set_ylim(FIXED_YLIM)

                if i >= 2: ax.set_xlabel("RPM", fontsize=FONT_LABEL)
                if i % 2 == 0: ax.set_ylabel(metric_label, fontsize=FONT_LABEL)
                else: ax.tick_params(labelleft=False)

            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=14)
            plt.savefig(os.path.join(sub_dir, f"{vol}_{metric_col}_Inter_2x2.png"), bbox_inches='tight')
            plt.close()

def plot_temporal_scatter_with_shapes(df, output_dir):
    sub_dir = os.path.join(output_dir, "Scatter_Plots_By_Speed")
    os.makedirs(sub_dir, exist_ok=True)
    volumes = df["volume"].unique()
    
    for vol in volumes:
        vol_df = df[df["volume"] == vol]
        speeds = sorted(vol_df["grouped_speed"].unique())
        materials = sorted(vol_df["material"].unique())
        
        for speed in speeds:
            s_df = vol_df[vol_df["grouped_speed"] == speed]
            if s_df.empty: continue
            
            xlim, ylim = get_global_limits(s_df, "voltage_std", "angle_mean")

            fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
            fig.suptitle(f"{speed} RPM | Volume {vol} - Temporal Scatter", fontsize=FONT_TITLE, fontweight='bold')
            axes_flat = axes.flatten()
            global_sc = None
            
            for i, mat in enumerate(materials):
                if i >= 4: break
                ax = axes_flat[i]
                m_df = s_df[s_df["material"] == mat]
                
                if m_df.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center'); continue
                
                trials = sorted(m_df["trial"].unique())
                for t_idx, trial_id in enumerate(trials):
                    t_df = m_df[m_df["trial"] == trial_id]
                    if t_df.empty: continue
                    marker = TRIAL_MARKERS[t_idx % len(TRIAL_MARKERS)]
                    
                    sc = ax.scatter(t_df["voltage_std"], t_df["angle_mean"], c=t_df["minute_bin"], cmap="coolwarm",
                                    marker=marker, s=80, edgecolors='black', alpha=0.8, vmin=0, vmax=60)
                    global_sc = sc
                    
                    if len(t_df) > 2:
                        z = np.polyfit(t_df["voltage_std"], t_df["angle_mean"], 1); p = np.poly1d(z)
                        xr = np.linspace(t_df["voltage_std"].min(), t_df["voltage_std"].max(), 10)
                        ax.plot(xr, p(xr), color='black', linestyle='--', alpha=0.5)

                ax.set_title(mat.title(), fontsize=FONT_LABEL, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
                ax.set_xlim(xlim); ax.set_ylim(ylim)

                if i >= 2: ax.set_xlabel("Voltage Std (V)", fontsize=FONT_LABEL)
                if i % 2 == 0: ax.set_ylabel("Angle (deg)", fontsize=FONT_LABEL)
                else: ax.tick_params(labelleft=False)
            
            if global_sc:
                cbar = fig.colorbar(global_sc, ax=axes, shrink=0.7, location='right', pad=0.05)
                cbar.set_label("Elapsed Time (min)", fontsize=14)
                
                shape_handles = [Line2D([0],[0], marker=TRIAL_MARKERS[t], color='w', label=f"Trial {t+1}", 
                                        markerfacecolor='grey', markersize=10, markeredgecolor='k') for t in range(3)]
                fig.legend(handles=shape_handles, loc='upper right', bbox_to_anchor=(1.08, 0.95), title="Trials", fontsize=12)

            plt.savefig(os.path.join(sub_dir, f"Scatter_{vol}_{speed}RPM.png"), bbox_inches='tight')
            plt.close()

# ======================================================
# NEW FUNCTION: GLOBAL SUMMARY (ALL MATERIALS)
# ======================================================
def plot_global_summary_all_materials(df, output_dir):
    """
    Plots ALL materials on a SINGLE graph per Volume.
    Color = Speed.
    Shape = Material.
    """
    sub_dir = os.path.join(output_dir, "Global_Summary_All_Materials")
    os.makedirs(sub_dir, exist_ok=True)
    
    volumes = sorted(df["volume"].unique())
    
    # Define Speed Colors (Consistent across all plots)
    all_speeds = sorted(df["grouped_speed"].unique())
    cmap = plt.get_cmap('tab10')
    if len(all_speeds) > 10: cmap = plt.get_cmap('jet')
    speed_color_map = {s: cmap(i/len(all_speeds)) for i, s in enumerate(all_speeds)}

    for vol in volumes:
        vol_df = df[df["volume"] == vol]
        if vol_df.empty: continue
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        materials = sorted(vol_df["material"].unique())
        
        for mat in materials:
            m_df = vol_df[vol_df["material"] == mat]
            marker = MATERIAL_MARKERS.get(mat.lower(), 'X') # Default to X if unknown
            
            # Group by Speed to reduce scatter noise? Or plot all points?
            # User asked for "Whole Run Summary", which usually implies raw points or minute-averages.
            # Let's plot the minute-bin averages we already have in the DF.
            
            for speed in all_speeds:
                s_df = m_df[m_df["grouped_speed"] == speed]
                if s_df.empty: continue
                
                c = speed_color_map[speed]
                
                # Plot Points
                ax.scatter(s_df["voltage_std"], s_df["angle_mean"], 
                           color=c, marker=marker, s=60, alpha=0.6, edgecolors='k', linewidth=0.5)
                
                # Optional: Trendlines per material/speed? Too messy.
                # Let's just do points for now to see the clusters.

        ax.set_title(f"Global Summary: All Materials @ {vol} Volume", fontsize=FONT_TITLE, fontweight='bold')
        ax.set_xlabel("Voltage Std (V)", fontsize=FONT_LABEL)
        ax.set_ylabel("Angle of Repose (deg)", fontsize=FONT_LABEL)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
        
        # --- DUAL LEGEND (Right Side) ---
        
        # 1. Shape Legend (Material)
        shape_handles = []
        for mat in materials:
            marker = MATERIAL_MARKERS.get(mat.lower(), 'X')
            shape_handles.append(Line2D([0], [0], marker=marker, color='w', label=mat.title(), 
                                        markerfacecolor='grey', markersize=10, markeredgecolor='k'))
            
        leg1 = ax.legend(handles=shape_handles, title="Material", loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12)
        ax.add_artist(leg1)
        
        # 2. Color Legend (Speed)
        color_handles = []
        for speed in all_speeds:
            c = speed_color_map[speed]
            color_handles.append(Line2D([0], [0], marker='o', color='w', label=f"{speed} RPM", 
                                        markerfacecolor=c, markersize=10))
            
        ax.legend(handles=color_handles, title="Speed", loc='upper left', bbox_to_anchor=(1.01, 0.6), fontsize=12)

        plt.tight_layout() # Adjusts for the external legends usually, but bbox might need extra margins
        plt.subplots_adjust(right=0.8) # Manual margin for double legend
        
        plt.savefig(os.path.join(sub_dir, f"Global_Summary_{vol}.png"), bbox_inches='tight')
        plt.close()

# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    master_df = load_all_data(BASE_DIR)
    
    if not master_df.empty:
        out_path = os.path.join(BASE_DIR, "Master_Analysis_Results_V4")
        os.makedirs(out_path, exist_ok=True)
        
        master_df.to_csv(os.path.join(out_path, "all_materials_aggregated.csv"), index=False)
        print(f"\nData loaded. {len(master_df)} rows.")
        
        print("1/4: Intra-Material Grids...")
        plot_intra_material_hysteresis(master_df, out_path)
        
        print("2/4: Inter-Material Grids...")
        plot_inter_material_hysteresis(master_df, out_path)
        
        print("3/4: Temporal Scatter Plots...")
        plot_temporal_scatter_with_shapes(master_df, out_path)
        
        print("4/4: Global Summary (All Materials)...")
        plot_global_summary_all_materials(master_df, out_path)
        
        print(f"\nDone! Results saved to: {out_path}")
    else:
        print("No valid data found.")