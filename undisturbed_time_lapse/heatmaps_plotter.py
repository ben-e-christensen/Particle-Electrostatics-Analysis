import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib import rcParams

# ================= THE "KNOBS" =================
TARGET_SPEED = 16  # Must match TARGET_SPEED used in extract_longitudinal_heatmaps.py
NUM_DAYS = 7

# --- LAYOUT: 2 rows x 4 cols. Days 1-4 top row, Days 5-7 + colorbar bottom row ---
GRID_ROWS = 2
GRID_COLS = 4

# --- SPACING CONTROLS ---
FIG_W = 14.0        # Overall figure width (inches)

COL_GAP = 0.05      # Gap between day columns
ROW_GAP = 0.05      # Gap between the two rows

GRID_LEFT = 0.03
GRID_RIGHT = 0.97
GRID_TOP = 0.90
GRID_BOT = 0.03

# --- FONTS & TOGGLES ---
SHOW_TITLES = True   # "Day N" labels above each panel
FONT_TICK = 16
FONT_LABEL = 18
FONT_TITLE = 22

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
# ===============================================


def assemble(parent_dir):
    normalized_path = parent_dir.replace('\\', '/').rstrip('/')
    material = os.path.basename(normalized_path)

    cache_dir = os.path.join(normalized_path, "heatmap_cache_longitudinal")
    output_dir = os.path.join(normalized_path, "heatmaps")

    if not os.path.exists(cache_dir):
        print(f"Error: Could not find the cache folder at {cache_dir}")
        print("Please run extract_longitudinal_heatmaps.py on this folder first!")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Building 7-Day Evolution Report for: {material} ===")

    # --- 1. SNEAK PEEK AT THE DATA (for aspect ratio math) ---
    sample_path = None
    for day in range(1, NUM_DAYS + 1):
        candidate = os.path.join(cache_dir, f"day{day}_{TARGET_SPEED}RPM.npy")
        if os.path.exists(candidate):
            sample_path = candidate
            break

    if sample_path is None:
        print(f"No day*_{TARGET_SPEED}RPM.npy files found in {cache_dir}!")
        return

    sample_data = np.load(sample_path)
    img_h, img_w = sample_data.shape
    img_aspect_ratio = img_h / img_w

    # --- 2. EXACT ASPECT RATIO MATH (same approach as heatmap_graph_builder.py) ---
    w_span = GRID_RIGHT - GRID_LEFT
    h_span = GRID_TOP - GRID_BOT
    plot_width = FIG_W * w_span

    W_cell = plot_width / (GRID_COLS + (GRID_COLS - 1) * COL_GAP)
    H_cell = W_cell * img_aspect_ratio
    plot_height = H_cell * (GRID_ROWS + (GRID_ROWS - 1) * ROW_GAP)
    dynamic_fig_h = plot_height / h_span
    # ---------------------------------

    fig = plt.figure(figsize=(FIG_W, dynamic_fig_h))
    fig.subplots_adjust(left=GRID_LEFT, right=GRID_RIGHT, top=GRID_TOP, bottom=GRID_BOT)

    gs = fig.add_gridspec(GRID_ROWS, GRID_COLS, wspace=COL_GAP, hspace=ROW_GAP)

    shared_im = None

    for day in range(1, NUM_DAYS + 1):
        idx = day - 1
        row = idx // GRID_COLS
        col = idx % GRID_COLS

        ax = fig.add_subplot(gs[row, col])

        path = os.path.join(cache_dir, f"day{day}_{TARGET_SPEED}RPM.npy")
        if os.path.exists(path):
            data = np.load(path)
            im = ax.imshow(data, cmap='inferno', vmin=0, vmax=100)
            shared_im = im
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if SHOW_TITLES:
            ax.set_title(f"Day {day}", fontsize=FONT_TITLE, pad=10)

    # --- COLORBAR: lives in the leftover grid cell (bottom-right) ---
    last_row = (NUM_DAYS - 1) // GRID_COLS
    last_col = (NUM_DAYS - 1) % GRID_COLS

    if shared_im and (last_row < GRID_ROWS - 1 or last_col < GRID_COLS - 1):
        # Find the first unused cell after the last day panel
        cbar_row, cbar_col = last_row, last_col + 1
        if cbar_col >= GRID_COLS:
            cbar_row += 1
            cbar_col = 0

        cell_pos = gs[cbar_row, cbar_col].get_position(fig)

        cbar_w = 0.22 * cell_pos.width
        cbar_h = 0.75 * cell_pos.height
        cbar_x = cell_pos.x0 + 0.15 * cell_pos.width
        cbar_y = cell_pos.y0 + 0.5 * (cell_pos.height - cbar_h)

        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
        cbar = fig.colorbar(shared_im, cax=cbar_ax)
        cbar.set_label("Frequency (%)", fontsize=FONT_LABEL, fontweight='bold')
        cbar.ax.tick_params(labelsize=FONT_TICK)

    out_name = os.path.join(output_dir, f"{material}_7Day_Evolution.png")
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"=== Report saved to: {out_name} ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assemble_longitudinal_heatmaps.py <path_to_material_folder>")
    else:
        assemble(sys.argv[1])