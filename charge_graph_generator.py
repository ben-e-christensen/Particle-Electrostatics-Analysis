import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import random
import os

# ================= CONFIGURATION =================
# File Paths
CSV_FILE_PATH = "experiment_log.csv"
IMAGE_FOLDER_PATH = "./images"

# Output Filenames
OUTPUT_GRAPH_PATH = "charge_graph_snapshot.png"
OUTPUT_HEATMAP_PATH = "average_shape_heatmap.png"

# Analysis Settings
SAMPLE_RATE_HZ = 100
WINDOW_SECONDS = 15
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)
THRESHOLD_VAL = 50  # For heatmap generation (matches your live feed)
# =================================================


# ---------------------------------------------------------
# 1. Synchronized Data Loading
# ---------------------------------------------------------
def load_synchronized_window(csv_path, samples_needed):
    """
    Reads the CSV and returns concurrent slices of CH2 data
    and frame filenames for a random time window.
    """
    all_ch2 = []
    all_frames = []
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None, None, 0

    print("Reading CSV data...")
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                try:
                    # Ensure row has enough columns
                    if len(row) < 12: continue
                        
                    # Index 7 = CH2_volts, Index 11 = frame_name
                    val = float(row[7])
                    fname = row[11].strip()
                    
                    all_ch2.append(val)
                    all_frames.append(fname)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None, 0

    # Validate data length
    total_points = len(all_ch2)
    if total_points < samples_needed:
        print(f"Not enough data found ({total_points} points). Need {samples_needed}.")
        return None, None, 0

    # Pick ONE random start index for both datasets
    max_start = total_points - samples_needed
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + samples_needed

    print(f"Selected 5s Window: Data Index {start_idx} to {end_idx}")

    # Return the synchronized slices
    return (all_ch2[start_idx:end_idx], 
            all_frames[start_idx:end_idx], 
            start_idx)


# ---------------------------------------------------------
# 2. Generate Image A: Charge Graph
# ---------------------------------------------------------
def plot_charge_graph(ch2_data, start_idx):
    # Create Time-axis (0 to 5 seconds)
    t_vals = [i / SAMPLE_RATE_HZ for i in range(len(ch2_data))]

    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, ch2_data, color='black', linewidth=1, label="Charge")

    # Styling
    plt.title(f"Charge vs Time")
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Voltage (V)", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=14) 
    plt.tight_layout()

    print(f"Saving graph to {OUTPUT_GRAPH_PATH}...")
    plt.savefig(OUTPUT_GRAPH_PATH, dpi=300, bbox_inches='tight')
    # plt.close() # Close plot to free memory if running in a loop


# ---------------------------------------------------------
# 3. Generate Image B: Shape Heatmap
# ---------------------------------------------------------
def plot_shape_heatmap(frame_names, start_idx):
    accumulator = None
    count = 0
    
    print(f"Processing images for heatmap...")
    
    for fname in frame_names:
        if not fname: continue
        full_path = os.path.join(IMAGE_FOLDER_PATH, fname)
        
        if not os.path.exists(full_path): continue

        # Load as grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # Initialize accumulator on first valid frame
        if accumulator is None:
            accumulator = np.zeros_like(img, dtype=np.float32)

        # Binarize based on threshold (creates the 'shape' mask)
        _, mask = cv2.threshold(img, THRESHOLD_VAL, 1, cv2.THRESH_BINARY)

        accumulator += mask
        count += 1

    if accumulator is None or count == 0:
        print("Could not generate heatmap (no valid images found in window).")
        return

    # Normalize
    heatmap = accumulator / count

    # Plotting
    plt.figure(figsize=(10, 8))
    # Using 'inferno' colormap for a nice glowing effect
    plt.imshow(heatmap, cmap='inferno', vmin=0, vmax=np.max(heatmap))
    
    cbar = plt.colorbar()
    cbar.set_label("Occupancy Frequency")
    
    plt.title(f"Average Particle Shape (Source Index: {start_idx})")
    plt.axis('off')

    print(f"Saving heatmap to {OUTPUT_HEATMAP_PATH}...")
    plt.savefig(OUTPUT_HEATMAP_PATH, dpi=300, bbox_inches='tight')
    # plt.close()


# ================= MAIN =================
if __name__ == "__main__":
    # 1. Load Data Once
    ch2_slice, frame_slice, start_index = load_synchronized_window(CSV_FILE_PATH, WINDOW_SAMPLES)

    if ch2_slice is not None:
        # 2. Generate Graph
        plot_charge_graph(ch2_slice, start_index)
        
        # 3. Generate Heatmap
        plot_shape_heatmap(frame_slice, start_index)

        print("\nDone. Showing generated plots...")
        plt.show() # Show both windows at the end