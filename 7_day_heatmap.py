import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib import rcParams

# ================= THE "KNOBS" =================
DAYS = [f"day{i}" for i in range(1, 8)]
TARGET_SPEED = 16

# --- SPACING CONTROLS ---
FIG_W = 16.0  
COL_GAP = 0.05    
ROW_GAP = 0.15   

# --- FONTS & TOGGLES ---
SHOW_TITLES = False
FONT_TICK = 18
FONT_LABEL = 18
FONT_TITLE = 20
FONT_DAY = 22

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
# ===============================================

def assemble_longitudinal(parent_dir):
    normalized_path = parent_dir.replace('\\', '/').rstrip('/')
    material = os.path.basename(normalized_path)
    
    cache_dir = os.path.join(normalized_path, "heatmap_cache_longitudinal")
    output_dir = os.path.join(normalized_path, "heatmaps_longitudinal")
    
    if not os.path.exists(cache_dir):
        print(f"Error: Could not find cache folder at {cache_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Building 7-Day Heatmap Progression for: {material} ===")

    # Aspect ratio math
    sample_file = next((f for f in os.listdir(cache_dir) if f.endswith(".npy")), None)
    if not sample_file:
        print(f"No .npy files found in {cache_dir}!")
        return
        
    sample_data = np.load(os.path.join(cache_dir, sample_file))
    img_h, img_w = sample_data.shape
    img_aspect_ratio = img_h / img_w 
    
    plot_width = FIG_W * 0.9 
    W_cell = plot_width / (4 + 3 * COL_GAP)
    H_cell = W_cell * img_aspect_ratio
    plot_height = (H_cell * 2) + (H_cell * ROW_GAP)
    dynamic_fig_h = plot_height / 0.85 
    
    fig = plt.figure(figsize=(FIG_W, dynamic_fig_h))
    
    if SHOW_TITLES:
        fig.suptitle(f"{material.capitalize()} - 7-Day Evolution ({TARGET_SPEED} RPM)", 
                     fontsize=FONT_TITLE, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 4, wspace=COL_GAP, hspace=ROW_GAP, 
                          left=0.05, right=0.95, top=0.90, bottom=0.05)
    
    shared_im = None
    
    for i, day in enumerate(DAYS):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        path = os.path.join(cache_dir, f"{day}_{TARGET_SPEED}RPM.npy")
        
        if os.path.exists(path):
            data = np.load(path)
            im = ax.imshow(data, cmap='inferno', vmin=0, vmax=100)
            shared_im = im
        else:
            ax.text(0.5, 0.5, f"No Data\n{day}", ha='center', va='center', color='gray', fontsize=FONT_LABEL)
            ax.set_facecolor('#f0f0f0') 
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): 
            spine.set_visible(False)
            
        ax.set_title(f"Day {i+1}", fontsize=FONT_DAY, fontweight='normal', pad=10)

    # Colorbar in slot 8
    if shared_im:
        cbar_ax = fig.add_subplot(gs[1, 3])
        cbar_ax.axis('off') 
        cax = cbar_ax.inset_axes([0.3, 0.1, 0.15, 0.8]) 
        
        cbar = fig.colorbar(shared_im, cax=cax)
        cbar.set_label("Frequency (%)", fontsize=FONT_LABEL, fontweight='bold', labelpad=15)
        cbar.ax.tick_params(labelsize=FONT_TICK)

    out_name = os.path.join(output_dir, f"{material}_7Day_Evolution.png")
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"=== Report finished! Saved to: {out_name} ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python longitudinal_designer.py <path_to_material_folder>")
    else:
        assemble_longitudinal(sys.argv[1])