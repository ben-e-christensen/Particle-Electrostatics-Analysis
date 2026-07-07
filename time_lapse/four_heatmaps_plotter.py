import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib import rcParams

# ================= THE "KNOBS" =================
TARGET_SPEED = 16  # Must match TARGET_SPEED used in extract_longitudinal_heatmaps.py
DAYS_TO_SHOW = [1, 7]  # Rows, top-to-bottom

# --- Material folders (folder_name, display_title) -> columns, left-to-right ---
MATERIAL_FOLDERS = [
    ("Dirty",          "As-Is"),
    ("cleaned-once",   "Cleaned Once"),
    ("Cleaned_Thrice", "Cleaned Thrice"),
    ("undisturbed",    "Undisturbed"),
]

GRID_ROWS = len(DAYS_TO_SHOW)
GRID_COLS = len(MATERIAL_FOLDERS)

# --- SPACING CONTROLS ---
FIG_W = 16.0        # Overall figure width (inches)

COL_GAP = 0.05      # Gap between material columns
ROW_GAP = 0.05      # Gap between day rows

GRID_LEFT = 0.06
GRID_RIGHT = 0.90    # Leaves room on the right for the shared colorbar
GRID_TOP = 0.88
GRID_BOT = 0.05

# Colorbar (lives outside the grid, to the right)
CBAR_X = 0.92
CBAR_W = 0.02
CBAR_Y = 0.25
CBAR_H = 0.5

# --- FONTS & TOGGLES ---
FONT_TICK = 16
FONT_LABEL = 20
FONT_TITLE = 22

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
# ===============================================


def find_npy(parent_dir, folder_name, day):
    """Checks folder_name/day{N}_{SPEED}RPM.npy first, then falls back to
    folder_name/heatmap_cache_longitudinal/day{N}_{SPEED}RPM.npy."""
    direct = os.path.join(parent_dir, folder_name, f"day{day}_{TARGET_SPEED}RPM.npy")
    if os.path.exists(direct):
        return direct

    nested = os.path.join(parent_dir, folder_name, "heatmap_cache_longitudinal",
                           f"day{day}_{TARGET_SPEED}RPM.npy")
    if os.path.exists(nested):
        return nested

    return None


def assemble(parent_dir):
    normalized_path = parent_dir.replace('\\', '/').rstrip('/')

    output_dir = os.path.join(normalized_path, "comparison_reports")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Building Material Comparison Report ===")

    # --- 1. SNEAK PEEK AT THE DATA (for aspect ratio math) ---
    sample_path = None
    for folder_name, _ in MATERIAL_FOLDERS:
        for day in DAYS_TO_SHOW:
            candidate = find_npy(normalized_path, folder_name, day)
            if candidate:
                sample_path = candidate
                break
        if sample_path:
            break

    if sample_path is None:
        print(f"No day*_{TARGET_SPEED}RPM.npy files found under {normalized_path}!")
        return

    sample_data = np.load(sample_path)
    img_h, img_w = sample_data.shape
    img_aspect_ratio = img_h / img_w

    # --- 2. EXACT ASPECT RATIO MATH ---
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

    for row, day in enumerate(DAYS_TO_SHOW):
        for col, (folder_name, display_title) in enumerate(MATERIAL_FOLDERS):
            ax = fig.add_subplot(gs[row, col])

            path = find_npy(normalized_path, folder_name, day)
            if path:
                data = np.load(path)
                im = ax.imshow(data, cmap='inferno', vmin=0, vmax=100)
                shared_im = im
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column titles: only on the top row
            if row == 0:
                ax.set_title(display_title, fontsize=FONT_TITLE, fontweight='bold', pad=12)

            # Row labels: only on the leftmost column
            if col == 0:
                ax.set_ylabel(f"Day {day}", fontsize=FONT_LABEL, fontweight='bold', labelpad=10)

    # --- SHARED COLORBAR (outside the grid, on the right) ---
    if shared_im:
        cbar_ax = fig.add_axes([CBAR_X, CBAR_Y, CBAR_W, CBAR_H])
        cbar = fig.colorbar(shared_im, cax=cbar_ax)
        cbar.set_label("Frequency (%)", fontsize=FONT_LABEL, fontweight='bold')
        cbar.ax.tick_params(labelsize=FONT_TICK)

    out_name = os.path.join(output_dir, f"Material_Comparison_Day{DAYS_TO_SHOW[0]}_vs_Day{DAYS_TO_SHOW[-1]}.png")
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"=== Report saved to: {out_name} ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assemble_material_comparison.py <path_to_heatmap_caches_folder>")
    else:
        assemble(sys.argv[1])