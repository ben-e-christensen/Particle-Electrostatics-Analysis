import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from matplotlib.lines import Line2D
import sys
import os
import math

# ================= CONFIGURATION =================
CALIBRATE_MODE = False  
ROI_CENTER = (753, 485)  # (X, Y)
ROI_RADIUS = 505

IMAGE_SUBFOLDER = "images" 
OUTPUT_FOLDER_NAME = "Ensemble_Multi_Volume_Results"
SPEED_ROUNDING = 0 
BASELINE_SPEED = 1
BASELINE_DIR = "Increasing"
COLS_PER_PAGE = 3 
ORDER_KEYS = ["500", "750", "1000"]
# =================================================

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
    total_pages = math.ceil(len(unique_speeds) / COLS_PER_PAGE)
    for page_idx in range(total_pages):
        start_idx = page_idx * COLS_PER_PAGE
        end_idx = start_idx + COLS_PER_PAGE
        current_speeds = unique_speeds[start_idx:end_idx]
        
        title_text = f"{material_name} {subfolder_name} Heatmap Part {page_idx + 1}"
        fig, axes = plt.subplots(nrows=2, ncols=len(current_speeds), 
                                 figsize=(len(current_speeds)*5, 8), 
                                 constrained_layout=True, squeeze=False)
        fig.suptitle(title_text, fontsize=22)
        
        shared_im = None
        for col_i, speed in enumerate(current_speeds):
            for row_i, direction in enumerate(["Increasing", "Decreasing"]):
                ax = axes[row_i, col_i]
                if direction in heatmap_store.get(speed, {}):
                    im = ax.imshow(heatmap_store[speed][direction], cmap='inferno', vmin=0, vmax=100)
                    if row_i == 0: 
                        ax.set_title(f"{speed} RPM", fontsize=18, fontweight='bold')
                        shared_im = im
                else: 
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)

        axes[0, 0].set_ylabel("INCREASING \u2191", fontsize=20, fontweight='bold', labelpad=15)
        axes[1, 0].set_ylabel("DECREASING \u2193", fontsize=20, fontweight='bold', labelpad=15)
        
        if shared_im:
            cbar = fig.colorbar(shared_im, ax=axes, shrink=0.8, location='right', aspect=30)
            cbar.set_label("Frequency (%)", fontsize=14)

        safe_name = f"{material_name}_{subfolder_name}_Part{page_idx+1}.png".replace(" ", "_")
        plt.savefig(os.path.join(output_dir, safe_name), dpi=300)
        plt.close()
        print(f"    Saved Heatmap Grid: {safe_name}")

def get_experiment_data(folder_path, output_dir, material_name, subfolder_key):
    # 1. FIND CSV (Restored original search logic)
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

    # 2. LOAD DATA
    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed", 
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg", 
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt", 
            "ch2_flag", "ch3_flag"]
    try:
        df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip", engine="python")
        
        # --- NEW TIMESTAMP MATCHING LOGIC ---
        # If the CSV failed to log the image names, reconstruct them from the actual files!
        if df["frame_name"].isnull().all():
            image_dir = os.path.join(target_root, "images")
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
                print(f"  -> CSV missing frame names. Matching {len(image_files)} images by timestamp...")
                
                # Extract timestamp from image names (e.g., "1771281983581_frame_002235.jpg")
                img_data = []
                for img in image_files:
                    try:
                        ts_ms = int(img.split('_')[0])
                        img_data.append({"actual_frame": img, "timestamp": ts_ms / 1000.0})
                    except ValueError:
                        continue # Skip files that don't match the naming convention
                        
                if img_data:
                    img_df = pd.DataFrame(img_data).sort_values("timestamp")
                    df = df.sort_values("timestamp")
                    
                    # Snap each physical image to the closest CSV timestamp (within 0.1 seconds)
                    df = pd.merge_asof(df, img_df, on="timestamp", direction="nearest", tolerance=0.1)
                    df["frame_name"] = df["actual_frame"]
        # ------------------------------------
        
        # Now it will only drop rows that truly don't have a matching picture
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

    # 3. GENERATE HEATMAPS
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
                img = apply_circular_mask(img) # Apply ROI
                if accumulator is None: accumulator = np.zeros_like(img, dtype=np.float32)
                _, mask = cv2.threshold(img, 135, 1, cv2.THRESH_BINARY)
                accumulator += mask
                count += 1
            
            if accumulator is not None and count > 0:
                heatmap_store[speed][direction] = (accumulator / count) * 100.0

    save_heatmap_grids(heatmap_store, unique_speeds, material_name, subfolder_key, output_dir)

    # 4. CALCULATE METRICS (SHIFT + CORRELATION)
    if BASELINE_SPEED not in heatmap_store or BASELINE_DIR not in heatmap_store[BASELINE_SPEED]:
        print(f"  Baseline missing in {subfolder_key}")
        return None, None

    base_map = heatmap_store[BASELINE_SPEED][BASELINE_DIR]
    base_cy, base_cx = center_of_mass(base_map)
    base_flat = base_map.flatten() # For correlation

    results = []
    for speed in unique_speeds:
        for direction in ["Increasing", "Decreasing"]:
            if direction in heatmap_store.get(speed, {}):
                curr_map = heatmap_store[speed][direction]
                
                # A. Displacement
                curr_cy, curr_cx = center_of_mass(curr_map)
                dx = curr_cx - base_cx
                dy = base_cy - curr_cy
                
                # B. Deformation (Correlation)
                curr_flat = curr_map.flatten()
                corr = np.corrcoef(base_flat, curr_flat)[0, 1]
                
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

    # 1. IDENTIFY SUBFOLDERS
    all_subs = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    target_folders = {}
    for folder in all_subs:
        fname = os.path.basename(folder)
        for key in ORDER_KEYS:
            if key.lower() in fname.lower() and "clean" not in fname.lower():
                target_folders[key] = folder

    # 2. PROCESS DATA
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
    cmap = plt.get_cmap('tab10')
    if len(sorted_speeds) > 10: cmap = plt.get_cmap('jet')
    speed_color_map = {s: cmap(i/len(sorted_speeds)) for i, s in enumerate(sorted_speeds)}

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
    fig.suptitle(f"{material_name} - Center of Mass Shifting", fontsize=22, fontweight='bold')

    for i, key in enumerate(ORDER_KEYS):
        ax = axes[i]
        if key not in dataset:
            ax.axis('off'); continue

        points = dataset[key]
        ax.axhline(0, c='k', lw=1, ls='-', alpha=0.3)
        ax.axvline(0, c='k', lw=1, ls='-', alpha=0.3)

        for p in points:
            c = speed_color_map.get(p['speed'], 'black')
            marker = 's' if p['dir'] == "Increasing" else '^'
            ax.scatter(p['x'], p['y'], color=c, marker=marker, s=100, edgecolor='black', alpha=0.8)
            
            partner = next((item for item in points if item['speed'] == p['speed'] and item['dir'] != p['dir']), None)
            if partner and p['dir'] == "Increasing":
                ax.plot([p['x'], partner['x']], [p['y'], partner['y']], color=c, linestyle=':', alpha=0.4)

        ax.set_title(f"Volume: {key}", fontsize=16, fontweight='bold')
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_xlim(FIXED_XLIM); ax.set_ylim(FIXED_YLIM)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i == 0: ax.set_ylabel("Vert. Shift (px)", fontsize=14)
        else: ax.tick_params(labelleft=False)

    fig.supxlabel("Horizontal Shift (px)", fontsize=14, fontweight='bold')
    
    # Legends
    legend_shape = [Line2D([0],[0], marker='s', color='w', label='Increasing', markerfacecolor='grey', markersize=10, markeredgecolor='k'),
                    Line2D([0],[0], marker='^', color='w', label='Decreasing', markerfacecolor='grey', markersize=10, markeredgecolor='k')]
    legend_color = [Line2D([0],[0], marker='o', color='w', label=f"{s} RPM", markerfacecolor=speed_color_map[s], markersize=10) for s in sorted_speeds]
    
    fig.legend(handles=legend_shape, loc='center left', bbox_to_anchor=(1.01, 0.6), title="Phase")
    fig.legend(handles=legend_color, loc='center left', bbox_to_anchor=(1.01, 0.4), title="Speed")
    plt.savefig(os.path.join(output_dir, "Grid_1x3_Displacement.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================
    # PLOT 2: DEFORMATION ANALYSIS
    # =========================================================
    print("Generating Deformation Grid...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig2.suptitle(f"{material_name} - Shape Deformation (Correlation vs Baseline)", fontsize=22, fontweight='bold')

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
            ax.plot(speeds, corrs, marker='s', markersize=8, linewidth=2, color='grey', label='Increasing', alpha=0.7)

        if dec_points:
            speeds = [p['speed'] for p in dec_points]
            corrs = [p['correlation'] for p in dec_points]
            ax.plot(speeds, corrs, marker='^', markersize=8, linewidth=2, color='black', label='Decreasing')

        ax.set_title(f"Volume: {key}", fontsize=16, fontweight='bold')
        ax.grid(True, ls='--', alpha=0.3)
        
        # --- MODIFIED Y-LIMIT (0.6 to 1.05) ---
        ax.set_ylim(0.6, 1.05)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if i == 0: ax.set_ylabel("Shape Consistency (1.0 = Baseline)", fontsize=14)
        else: ax.tick_params(labelleft=False)
        
        if i == 2: ax.legend(loc='lower left', fontsize=12)

    fig2.supxlabel("Motor Speed (RPM)", fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(output_dir, "Grid_1x3_Deformation.png"), dpi=300, bbox_inches='tight')
    print(f"Success! Saved Grid_1x3_Displacement.png and Grid_1x3_Deformation.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_1x3_displacement_and_deformation.py <path_to_PARENT_folder>")
    else:
        process_multi_volume(sys.argv[1])