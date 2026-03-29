import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib import rcParams

# ================= THE "KNOBS" =================
VOL_KEYS = ["500", "750", "1000"] 

# --- SPACING CONTROLS ---
FIG_W = 12.0
COL_GAP = 0.05    # Gap between RPM columns
ROW_GAP = 0.05    # Gap between Accel/Decel
SPACER_VAL = 0.1  # Gap between the Top block and Bottom block


GRID_LEFT = 0.08
GRID_RIGHT = 0.82 
GRID_TOP = 0.92
GRID_BOT = 0.05
CBAR_X = 0.86

# --- FONTS & TOGGLES ---
SHOW_TITLES = False
FONT_TICK = 18
FONT_LABEL = 18
FONT_TITLE = 20

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"

SPEEDS = [1, 6, 11, 16, 21, 26]
# ===============================================

def assemble(parent_dir):
    # --- DYNAMIC PATH RESOLUTION ---
    # Strip trailing slashes and grab the folder name as the material
    normalized_path = parent_dir.replace('\\', '/').rstrip('/')
    material = os.path.basename(normalized_path)
    
    # Auto-generate the correct target directories
    cache_dir = os.path.join(normalized_path, "heatmap_cache")
    output_dir = os.path.join(normalized_path, "heatmaps")
    
    if not os.path.exists(cache_dir):
        print(f"Error: Could not find the cache folder at {cache_dir}")
        print("Please run extract_heatmaps.py on this folder first!")
        return
        
    # Ensure the output folder exists before we try to save to it
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Building Heatmap Reports for: {material} ===")

    # --- 1. SNEAK PEEK AT THE DATA ---
    sample_file = next((f for f in os.listdir(cache_dir) if f.endswith(".npy")), None)
    if not sample_file:
        print(f"No .npy files found in {cache_dir}!")
        return
        
    sample_data = np.load(os.path.join(cache_dir, sample_file))
    img_h, img_w = sample_data.shape
    img_aspect_ratio = img_h / img_w 
    
    # --- 2. EXACT ASPECT RATIO MATH ---
    w_span = GRID_RIGHT - GRID_LEFT
    h_span = GRID_TOP - GRID_BOT
    
    plot_width = FIG_W * w_span 
    W_cell = plot_width / (3 + 2 * COL_GAP)
    H_cell = W_cell * img_aspect_ratio
    
    block_height = H_cell * (2 + ROW_GAP)
    plot_height = block_height * (2 + SPACER_VAL)
    dynamic_fig_h = plot_height / h_span
    # ---------------------------------

    for vol_key in VOL_KEYS:
        print(f"  -> Assembling Volume {vol_key}...")
        
        fig = plt.figure(figsize=(FIG_W, dynamic_fig_h))
        
        if SHOW_TITLES:
            fig.suptitle(f"{material.capitalize()} {vol_key} Consolidated Heatmap", fontsize=FONT_TITLE, fontweight='bold', y=0.96)

        fig.subplots_adjust(left=GRID_LEFT, right=GRID_RIGHT, top=GRID_TOP, bottom=GRID_BOT)
        
        # --- NESTED GRIDSPEC SETUP ---
        gs_outer = fig.add_gridspec(2, 1, hspace=SPACER_VAL)
        gs_top = gs_outer[0].subgridspec(2, 3, wspace=COL_GAP, hspace=ROW_GAP)
        gs_bot = gs_outer[1].subgridspec(2, 3, wspace=COL_GAP, hspace=ROW_GAP)
        gs_blocks = [gs_top, gs_bot]
        
        shared_im = None
        directions = ["Increasing", "Decreasing"]
        y_labels = ["ACCELERATING", "DECELERATING"] # Arrows removed
        
        for i, speed in enumerate(SPEEDS):
            col = i % 3
            block_idx = 0 if i < 3 else 1
            gs_current = gs_blocks[block_idx]
            
            for phase_idx in range(2):
                d_name = directions[phase_idx]
                ax = fig.add_subplot(gs_current[phase_idx, col])
                
                # LOAD DATA INSTANTLY
                path = os.path.join(cache_dir, f"{vol_key}_{speed}_{d_name}.npy")
                if os.path.exists(path):
                    data = np.load(path)
                    im = ax.imshow(data, cmap='inferno', vmin=0, vmax=100)
                    shared_im = im
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')
                
                # --- FORMATTING ---
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                
                if phase_idx == 0:
                    ax.set_title(f"{speed} RPM", fontsize=FONT_LABEL, fontweight='bold', pad=10)
                    
                if col == 0:
                    ax.set_ylabel(y_labels[phase_idx], fontsize=FONT_LABEL, fontweight='bold', labelpad=10)

        # Add Colorbar safely in the reserved space
        if shared_im:
            cbar_ax = fig.add_axes([CBAR_X, 0.25, 0.03, 0.5])
            cbar = fig.colorbar(shared_im, cax=cbar_ax)
            cbar.set_label("Frequency (%)", fontsize=FONT_LABEL)
            cbar.ax.tick_params(labelsize=FONT_TICK)

        out_name = os.path.join(output_dir, f"{material}_{vol_key}_Consolidated_Report.png")
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        
        plt.close(fig)

    print(f"=== All reports finished! Saved to: {output_dir} ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python heatmap_graph_builder.py <path_to_material_folder>")
    else:
        assemble(sys.argv[1])