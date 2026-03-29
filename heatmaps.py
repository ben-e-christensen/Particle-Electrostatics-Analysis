import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from matplotlib.lines import Line2D
from matplotlib import rcParams
import sys
import os
import math

# ================= CONFIGURATION =================
CALIBRATE_MODE = False  
ROI_CENTER = (753, 485)  # (X, Y)
ROI_RADIUS = 505

# --- TOGGLES ---
SHOW_LEGEND = True  # Set to True to show legends AND the Heatmap Colorbar
SHOW_TITLES = True  # Set to True to show plot titles

IMAGE_SUBFOLDER = "images" 
OUTPUT_FOLDER_NAME = "Ensemble_Multi_Volume_Results"
SPEED_ROUNDING = 0 
BASELINE_SPEED = 1
BASELINE_DIR = "Increasing" # Internal data logic, visuals use Accel/Decel
ORDER_KEYS = ["500", "750", "1000"]

# --- ACADEMIC PLOT SETTINGS ---
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
FONT_TICK = 18
FONT_LABEL = 18
FONT_TITLE = 20

# Two-tone Grey/Black for Lines
DIR_COLORS = {
    'Accelerating': ('#d3d3d3', '#555555'), 
    'Decelerating': ('#888888', '#000000')
}

# Color mapping for different speeds
SPEED_COLORS = [
    ('#c5a3d4', '#9944ff'), ('#81aad4', '#083b70'), ('#a3d4a3', '#087008'),
    ('#d4a3a3', '#700808'), ('#d4c5a3', '#705c08'), ('#a3d4d4', '#087070'),
    ('#d4a3c5', '#70085c'), ('#b3b3b3', '#4d4d4d'), ('#ffb366', '#cc6600'),
    ('#99ccff', '#0044cc'), ('#ff99cc', '#cc0066'), ('#c2f0c2', '#339933')
]
# =================================================

def apply_academic_axes(ax, xlabel="", ylabel=""):
    """Applies the specific formatting to the axes."""
    ax.xaxis.set_tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_tick_params(labelsize=FONT_TICK)
    ax.tick_params('both', length=7, width=1, which='major')
    if xlabel: ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel: ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.grid(True, alpha=0.3)

# --- ROI UTILS ---
def run_calibration(parent_dir):
    print("--- ROI CALIBRATION MODE ---")
    sample_img_path = None
    for root, dirs, files in os.walk(parent_dir):
        if IMAGE_SUBFOLDER in root or "frames" in root:
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    sample_img_path = os.path.join(root, f)
                    break
        if sample_img_path: break
    
    if not sample_img_path:
        print("Error: No images found to calibrate with!")
        sys.exit()

    img = cv2.imread(sample_img_path)
    print("INSTRUCTIONS: Click Center, Drag to Edge, Enter to Finish.")
    sys.exit()

def apply_circular_mask(img):
    """Zeros out everything outside the hardcoded circle."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, ROI_CENTER, ROI_RADIUS, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def save_heatmap_grids(heatmap_store, unique_speeds, material_name, subfolder_name, output_dir):
    ncols = 3 
    total_speed_groups = math.ceil(len(unique_speeds) / ncols)
    
    # 1. DYNAMIC GRID MATH (Match Dummy Tester)
    # 2 rows for data, 1 row for spacer per group
    total_gs_rows = total_speed_groups * 3 - 1
    row_heights = []
    for i in range(total_speed_groups):
        row_heights.extend([1, 1]) 
        if i < total_speed_groups - 1:
            row_heights.append(0.3) # Spacer height

    # 2. ASPECT RATIO LOCK (12x16.8 logic)
    total_height_units = sum(row_heights)
    fig_width = 12.0
    fig_height = fig_width * (total_height_units / ncols)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    if SHOW_TITLES:
        title_text = f"{material_name} {subfolder_name} Consolidated Heatmap"
        fig.suptitle(title_text, fontsize=FONT_TITLE, fontweight='bold', y=0.96)

    # 3. BOUNDARY LOCK (Reserves the 18% right-side gap for colorbar)
    fig.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.05)

    gs = fig.add_gridspec(nrows=total_gs_rows, ncols=ncols, 
                          wspace=0.05, hspace=0.05, 
                          height_ratios=row_heights)

    shared_im = None
    data_dirs = ["Increasing", "Decreasing"] # Internal keys

    for i in range(total_speed_groups):
        start_idx = i * ncols
        current_speeds = unique_speeds[start_idx : start_idx + ncols]
        
        accel_gs_row = i * 3
        decel_gs_row = i * 3 + 1

        for col_i in range(ncols):
            if col_i < len(current_speeds):
                speed = current_speeds[col_i]
                
                # --- ACCELERATING ---
                ax_accel = fig.add_subplot(gs[accel_gs_row, col_i])
                if data_dirs[0] in heatmap_store.get(speed, {}):
                    im = ax_accel.imshow(heatmap_store[speed][data_dirs[0]], cmap='inferno', vmin=0, vmax=100)
                    shared_im = im
                    if SHOW_TITLES:
                        ax_accel.set_title(f"{speed} RPM", fontsize=FONT_LABEL, fontweight='bold', pad=10)
                
                ax_accel.set_xticks([]); ax_accel.set_yticks([])
                for spine in ax_accel.spines.values(): spine.set_visible(False)
                if col_i == 0:
                    ax_accel.set_ylabel("ACCELERATING \u2191", fontsize=FONT_LABEL, fontweight='bold', labelpad=10)

                # --- DECELERATING ---
                ax_decel = fig.add_subplot(gs[decel_gs_row, col_i])
                if data_dirs[1] in heatmap_store.get(speed, {}):
                    ax_decel.imshow(heatmap_store[speed][data_dirs[1]], cmap='inferno', vmin=0, vmax=100)
                
                ax_decel.set_xticks([]); ax_decel.set_yticks([])
                for spine in ax_decel.spines.values(): spine.set_visible(False)
                if col_i == 0:
                    ax_decel.set_ylabel("DECELERATING \u2193", fontsize=FONT_LABEL, fontweight='bold', labelpad=10)

    # 4. COLORBAR SAFE ZONE
    if SHOW_LEGEND and shared_im:
        # Match the dummy tester placement precisely
        cbar_ax = fig.add_axes([0.86, 0.25, 0.03, 0.5]) 
        cbar = fig.colorbar(shared_im, cax=cbar_ax)
        cbar.set_label("Frequency (%)", fontsize=FONT_LABEL)
        cbar.ax.tick_params(labelsize=FONT_TICK)

    safe_name = f"{material_name}_{subfolder_name}_Consolidated_Report.png".replace(" ", "_")
    plt.savefig(os.path.join(output_dir, safe_name), dpi=300)
    plt.close()
    
def get_experiment_data(folder_path, output_dir, material_name, subfolder_key):
    csv_path = None
    target_root = None
    for root, dirs, files in os.walk(folder_path):
        if "experiment_log.csv" in files:
            csv_path = os.path.join(root, "experiment_log.csv")
            target_root = root
            break
            
    if not csv_path: 
        print(f"  -> Could not find 'experiment_log.csv' inside {folder_path}")
        return None, None

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]
    try:
        df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
        
        if df["frame_name"].isnull().all():
            image_dir = os.path.join(target_root, "images")
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
                print(f"  -> CSV missing frame names. Matching {len(image_files)} images by timestamp...")
                
                img_data = []
                for img in image_files:
                    try:
                        ts_ms = int(img.split('_')[0])
                        img_data.append({"actual_frame": img, "timestamp": ts_ms / 1000.0})
                    except ValueError:
                        continue 
                        
                if img_data:
                    img_df = pd.DataFrame(img_data).sort_values("timestamp")
                    df = df.sort_values("timestamp")
                    df = pd.merge_asof(df, img_df, on="timestamp", direction="nearest", tolerance=0.1)
                    df["frame_name"] = df["actual_frame"]
        
        df = df.dropna(subset=["frame_name"])
        max_val = df["motor_speed"].max()
        max_indices = df.index[df["motor_speed"] == max_val].tolist()
        
        if not max_indices: 
            print("  -> 'max_indices' is empty. Motor speed might be entirely NaN.")
            return None, None
            
        mid_idx = max_indices[len(max_indices) // 2]
        df["direction"] = "Increasing"
        df.loc[mid_idx + 1:, "direction"] = "Decreasing"
        df["grouped_speed"] = df["motor_speed"].round(SPEED_ROUNDING).astype(int)
        
    except Exception as e: 
        print(f"  -> ERROR reading CSV: {e}")
        return None, None

    unique_speeds = sorted(df["grouped_speed"].unique())
    unique_speeds = [s for s in unique_speeds if s >= 1]
    heatmap_store = {}
    
    for speed in unique_speeds:
        heatmap_store[speed] = {}
        for direction in ["Increasing", "Decreasing"]:
            subset = df[(df["grouped_speed"] == speed) & (df["direction"] == direction)]
            if subset.empty: continue
            
            frames = subset["frame_name"].unique()
            img_dir = os.path.join(target_root, IMAGE_SUBFOLDER)
            if not os.path.exists(img_dir):
                if os.path.exists(os.path.join(target_root, "frames")): img_dir = os.path.join(target_root, "frames")
                else: continue
            
            accumulator = None; count = 0
            for fname in frames:
                fpath = os.path.join(img_dir, str(fname).strip())
                if not os.path.exists(fpath): continue
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = apply_circular_mask(img) 
                if accumulator is None: accumulator = np.zeros_like(img, dtype=np.float32)
                
                _, mask = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)
                accumulator += mask
                count += 1
            
            if accumulator is not None and count > 0:
                heatmap_store[speed][direction] = (accumulator / count) * 100.0

    save_heatmap_grids(heatmap_store, unique_speeds, material_name, subfolder_key, output_dir)

    if BASELINE_SPEED not in heatmap_store or BASELINE_DIR not in heatmap_store[BASELINE_SPEED]:
        print(f"  Baseline missing in {subfolder_key}")
        return None, None

    base_map = heatmap_store[BASELINE_SPEED][BASELINE_DIR]
    base_cy, base_cx = center_of_mass(base_map)
    base_flat = base_map.flatten()

    results = []
    for speed in unique_speeds:
        for direction in ["Increasing", "Decreasing"]:
            if direction in heatmap_store.get(speed, {}):
                curr_map = heatmap_store[speed][direction]
                curr_cy, curr_cx = center_of_mass(curr_map)
                dx = curr_cx - base_cx
                dy = base_cy - curr_cy
                corr = np.corrcoef(base_flat, curr_map.flatten())[0, 1]
                
                results.append({
                    'speed': speed, 
                    'dir': direction, 
                    'x': dx, 
                    'y': dy,
                    'correlation': corr
                })
                
    return results, unique_speeds

def process_multi_volume(parent_dir):
    if CALIBRATE_MODE: run_calibration(parent_dir)

    normalized_path = parent_dir.replace('\\', '/').rstrip('/')
    material_name = os.path.basename(normalized_path)
    print(f"--- Starting Analysis (Displacement + Deformation) for: {material_name} ---")
    
    output_dir = os.path.join(parent_dir, OUTPUT_FOLDER_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    all_subs = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    target_folders = {}
    for folder in all_subs:
        fname = os.path.basename(folder)
        for key in ORDER_KEYS:
            if key.lower() in fname.lower() and "clean" not in fname.lower():
                target_folders[key] = folder

    dataset = {} 
    all_speeds = set()
    for key in ORDER_KEYS:
        if key in target_folders:
            print(f"\nProcessing Group: {key}...")
            data, speeds = get_experiment_data(target_folders[key], output_dir, material_name, key)
            if data:
                dataset[key] = data
                all_speeds.update(speeds)

    if not dataset:
        print("No valid data found.")
        return

    sorted_speeds = sorted(list(all_speeds))
    speed_color_map = {s: SPEED_COLORS[i % len(SPEED_COLORS)] for i, s in enumerate(sorted_speeds)}

    # =========================================================
    # PLOT 1: DISPLACEMENT MAP
    # =========================================================
    print("\nGenerating Displacement Grid...")
    all_x, all_y = [0], [0]
    for key, points in dataset.items():
        for p in points: all_x.append(p['x']); all_y.append(p['y'])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    buff = max(x_max-x_min, y_max-y_min) * 0.1 if x_max!=x_min else 10
    FIXED_XLIM = (x_min - buff, x_max + buff)
    FIXED_YLIM = (y_min - buff, y_max + buff)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    if SHOW_TITLES:
        fig.suptitle(f"{material_name} - Center of Mass Shifting", fontsize=FONT_TITLE, fontweight='bold')

    for i, key in enumerate(ORDER_KEYS):
        ax = axes[i]
        if key not in dataset:
            ax.axis('off'); continue

        points = dataset[key]
        ax.axhline(0, c='k', lw=1, ls='-', alpha=0.3)
        ax.axvline(0, c='k', lw=1, ls='-', alpha=0.3)

        for p in points:
            face_c, edge_c = speed_color_map.get(p['speed'], ('#000000', '#000000'))
            marker = 's' if p['dir'] == "Increasing" else '^'
            ax.scatter(p['x'], p['y'], facecolors=face_c, edgecolors=edge_c, 
                       marker=marker, s=120, linewidth=1.5, alpha=0.8, zorder=3)
            
            partner = next((item for item in points if item['speed'] == p['speed'] and item['dir'] != p['dir']), None)
            if partner and p['dir'] == "Increasing":
                ax.plot([p['x'], partner['x']], [p['y'], partner['y']], color=edge_c, linestyle=':', alpha=0.5, zorder=2)

        if SHOW_TITLES:
            ax.set_title(f"Volume: {key}", fontsize=FONT_LABEL, fontweight='bold')
            
        ax.set_xlim(FIXED_XLIM); ax.set_ylim(FIXED_YLIM)
        apply_academic_axes(ax, "Horizontal Shift (px)" if i == 1 else "", "Vert. Shift (px)" if i == 0 else "")

    if SHOW_LEGEND:
        legend_shape = [
            Line2D([0],[0], marker='s', color='w', label='Accelerating', markerfacecolor='grey', markersize=12, markeredgecolor='k'),
            Line2D([0],[0], marker='^', color='w', label='Decelerating', markerfacecolor='grey', markersize=12, markeredgecolor='k')
        ]
        legend_color = [
            Line2D([0],[0], marker='o', color='w', label=f"{s} RPM", markerfacecolor=speed_color_map[s][0], 
                   markeredgecolor=speed_color_map[s][1], markersize=12) for s in sorted_speeds
        ]
        
        fig.legend(handles=legend_shape, loc='center left', bbox_to_anchor=(1.01, 0.6), title="Phase", fontsize=12, title_fontsize=14)
        fig.legend(handles=legend_color, loc='center left', bbox_to_anchor=(1.01, 0.4), title="Speed", fontsize=12, title_fontsize=14)

    plt.savefig(os.path.join(output_dir, "Grid_1x3_Displacement.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================
    # PLOT 2: DEFORMATION ANALYSIS
    # =========================================================
    print("Generating Deformation Grid...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    if SHOW_TITLES:
        fig2.suptitle(f"{material_name} - Shape Deformation (Correlation vs Baseline)", fontsize=FONT_TITLE, fontweight='bold')

    for i, key in enumerate(ORDER_KEYS):
        ax = axes2[i]
        if key not in dataset:
            ax.axis('off'); continue

        points = dataset[key]
        inc_points = sorted([p for p in points if p['dir'] == "Increasing"], key=lambda x: x['speed'])
        dec_points = sorted([p for p in points if p['dir'] == "Decreasing"], key=lambda x: x['speed'])

        if inc_points:
            speeds = [p['speed'] for p in inc_points]
            corrs = [p['correlation'] for p in inc_points]
            face_c, edge_c = DIR_COLORS['Accelerating']
            ax.plot(speeds, corrs, marker='s', markersize=10, linewidth=2, 
                    color=edge_c, markerfacecolor=face_c, markeredgecolor=edge_c, label='Accelerating')

        if dec_points:
            speeds = [p['speed'] for p in dec_points]
            corrs = [p['correlation'] for p in dec_points]
            face_c, edge_c = DIR_COLORS['Decelerating']
            ax.plot(speeds, corrs, marker='^', markersize=10, linewidth=2, 
                    color=edge_c, markerfacecolor=face_c, markeredgecolor=edge_c, label='Decelerating')

        if SHOW_TITLES:
            ax.set_title(f"Volume: {key}", fontsize=FONT_LABEL, fontweight='bold')
            
        ax.set_ylim(0.6, 1.05)
        apply_academic_axes(ax, "Motor Speed (RPM)" if i == 1 else "", "Shape Consistency" if i == 0 else "")
        
        if i == 2 and SHOW_LEGEND: 
            ax.legend(loc='lower left', fontsize=12)

    plt.savefig(os.path.join(output_dir, "Grid_1x3_Deformation.png"), dpi=300, bbox_inches='tight')
    print(f"Success! Saved Grid_1x3_Displacement.png and Grid_1x3_Deformation.png")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 heatmaps.py <path_to_PARENT_folder>")
    else:
        process_multi_volume(sys.argv[1])