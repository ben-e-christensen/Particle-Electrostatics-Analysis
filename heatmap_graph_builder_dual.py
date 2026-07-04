import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib import rcParams

# ================= THE "KNOBS" =================
VOL_KEYS = ["500", "750", "1000"] 

# --- LAYOUT TOGGLE ---
# Options: "split" (original 2 blocks of 3) OR "2x6" (one long block of 6)
LAYOUT_MODE = "2x6" 

# --- SPACING CONTROLS ---
FIG_W_SPLIT = 12.0  # Figure width for the split layout
FIG_W_2X6 = 30.0    # Figure width for the 2x6 layout

COL_GAP = 0.05    # Gap between RPM columns
ROW_GAP = 0.05    # Gap between Accel/Decel
SPACER_VAL = 0.1  # Gap between different blocks/materials

GRID_LEFT = 0.08
GRID_RIGHT = 0.82 
GRID_TOP = 0.92
GRID_BOT = 0.05

# Colorbar positioning and width depending on layout
CBAR_X = 0.86
CBAR_W_SPLIT = 0.03  # 3% of the 12-inch width
CBAR_W_2X6 = 0.01    # 1% of the 30-inch width (keeps absolute thickness ~the same)

# --- FONTS & TOGGLES ---
SHOW_TITLES = False
FONT_TICK = 18
FONT_LABEL = 18
FONT_TITLE = 20

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"

SPEEDS = [1, 6, 11, 16, 21, 26]
# ===============================================

def assemble(dir1, dir2):
    # --- 1. RESOLVE PATHS & NAMES FOR BOTH INPUTS ---
    dirs = [dir1, dir2]
    materials = []
    cache_dirs = []
    
    for d in dirs:
        norm_path = d.replace('\\', '/').rstrip('/')
        materials.append(os.path.basename(norm_path))
        
        c_dir = os.path.join(norm_path, "heatmap_cache")
        if not os.path.exists(c_dir):
            print(f"Error: Could not find cache folder at {c_dir}")
            print("Please run extract_heatmaps.py on this folder first!")
            return
        cache_dirs.append(c_dir)
        
    output_dir = os.path.join(os.getcwd(), f"Combined_{materials[0]}_vs_{materials[1]}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Building Stacked Reports for: {materials[0]} vs {materials[1]} ===")

    # --- 2. SNEAK PEEK AT THE DATA (Using Mat 1) ---
    sample_file = next((f for f in os.listdir(cache_dirs[0]) if f.endswith(".npy")), None)
    if not sample_file:
        print(f"No .npy files found in {cache_dirs[0]}!")
        return
        
    sample_data = np.load(os.path.join(cache_dirs[0], sample_file))
    img_h, img_w = sample_data.shape
    img_aspect_ratio = img_h / img_w 
    
    # --- 3. EXACT ASPECT RATIO MATH (Scaled for 2 Materials) ---
    current_fig_w = FIG_W_2X6 if LAYOUT_MODE == "2x6" else FIG_W_SPLIT
    current_cbar_w = CBAR_W_2X6 if LAYOUT_MODE == "2x6" else CBAR_W_SPLIT
    
    w_span = GRID_RIGHT - GRID_LEFT
    h_span = GRID_TOP - GRID_BOT
    plot_width = current_fig_w * w_span 
    
    if LAYOUT_MODE == "2x6":
        W_cell = plot_width / (6 + 5 * COL_GAP)
        H_cell = W_cell * img_aspect_ratio
        block_height = H_cell * (2 + ROW_GAP)
        # We now have 2 full material blocks + 1 spacer between them
        plot_height = block_height * (2 + SPACER_VAL) 
        dynamic_fig_h = plot_height / h_span
    else: 
        W_cell = plot_width / (3 + 2 * COL_GAP)
        H_cell = W_cell * img_aspect_ratio
        sub_block_height = H_cell * (2 + ROW_GAP)
        # We now have 4 sub-blocks total + 3 spacers
        plot_height = sub_block_height * (4 + 3 * SPACER_VAL)
        dynamic_fig_h = plot_height / h_span
    # ---------------------------------

    for vol_key in VOL_KEYS:
        print(f"  -> Assembling Stacked Volume {vol_key}...")
        
        fig = plt.figure(figsize=(current_fig_w, dynamic_fig_h))
        
        if SHOW_TITLES:
            title_str = f"{materials[0]} vs {materials[1]} - Vol {vol_key}"
            fig.suptitle(title_str, fontsize=FONT_TITLE, fontweight='bold', y=0.96)

        fig.subplots_adjust(left=GRID_LEFT, right=GRID_RIGHT, top=GRID_TOP, bottom=GRID_BOT)
        
        # --- 4. GRIDSPEC SETUP (Nested Master Grids) ---
        if LAYOUT_MODE == "2x6":
            gs_outer = fig.add_gridspec(2, 1, hspace=SPACER_VAL)
            gs_m1 = gs_outer[0].subgridspec(2, 6, wspace=COL_GAP, hspace=ROW_GAP)
            gs_m2 = gs_outer[1].subgridspec(2, 6, wspace=COL_GAP, hspace=ROW_GAP)
            
            # Pack them so looping is generic
            gs_blocks_all = [[gs_m1, gs_m1], [gs_m2, gs_m2]] 
        else:
            gs_outer = fig.add_gridspec(4, 1, hspace=SPACER_VAL)
            gs_m1_top = gs_outer[0].subgridspec(2, 3, wspace=COL_GAP, hspace=ROW_GAP)
            gs_m1_bot = gs_outer[1].subgridspec(2, 3, wspace=COL_GAP, hspace=ROW_GAP)
            gs_m2_top = gs_outer[2].subgridspec(2, 3, wspace=COL_GAP, hspace=ROW_GAP)
            gs_m2_bot = gs_outer[3].subgridspec(2, 3, wspace=COL_GAP, hspace=ROW_GAP)
            
            gs_blocks_all = [[gs_m1_top, gs_m1_bot], [gs_m2_top, gs_m2_bot]]
        
        shared_im = None
        directions = ["Increasing", "Decreasing"]
        
        # Static original Y-labels
        y_labels = ["ACCELERATING", "DECELERATING"]
        
        # --- 5. DRAW LOOP ---
        for m_idx, mat_name in enumerate(materials):
            cache_dir = cache_dirs[m_idx]
            gs_blocks = gs_blocks_all[m_idx]
            
            for i, speed in enumerate(SPEEDS):
                if LAYOUT_MODE == "2x6":
                    col = i
                    block_idx = 0
                else:
                    col = i % 3
                    block_idx = 0 if i < 3 else 1
                    
                gs_current = gs_blocks[block_idx]
                
                for phase_idx in range(2):
                    d_name = directions[phase_idx]
                    ax = fig.add_subplot(gs_current[phase_idx, col])
                    
                    # LOAD DATA
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
                    
                    # Only show titles (RPM) on the very top row (Material 1, phase 0)
                    if m_idx == 0 and phase_idx == 0:
                        ax.set_title(f"{speed} RPM", fontsize=FONT_LABEL, fontweight='bold', pad=10)
                        
                    # Show Y labels on the far left columns
                    if col == 0:
                        ax.set_ylabel(y_labels[phase_idx], fontsize=FONT_LABEL, fontweight='bold', labelpad=10)

        # --- 6. ONE SHARED COLORBAR ---
        if shared_im:
            # Placed right in the middle vertically
            cbar_ax = fig.add_axes([CBAR_X, 0.25, current_cbar_w, 0.5])
            cbar = fig.colorbar(shared_im, cax=cbar_ax)
            cbar.set_label("Frequency (%)", fontsize=FONT_LABEL)
            cbar.ax.tick_params(labelsize=FONT_TICK)

        out_name = os.path.join(output_dir, f"{materials[0]}_vs_{materials[1]}_{vol_key}_Consolidated_Report.png")
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"=== All reports finished! Saved to: {output_dir} ===")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python heatmap_graph_builder.py <path_to_material_1> <path_to_material_2>")
    else:
        assemble(sys.argv[1], sys.argv[2])