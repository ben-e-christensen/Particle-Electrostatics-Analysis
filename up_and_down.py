#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import medfilt, find_peaks

# === CONFIG ===
BASE_DIR = "F:/particle-data"
DERIVATIVE_THRESHOLD = -1
SLIP_THRESHOLD = DERIVATIVE_THRESHOLD / 4
SAMPLE_RATE = 100  # Hz
BASELINE_WINDOW_SEC = 4  # smoothing window duration in seconds

# --- Parse argument like "Acrylic/Acrylic-200" ---
if len(sys.argv) < 2:
    print("Usage: python3 main.py <Material/RunFolder>")
    print("Example: python3 main.py Acrylic/Acrylic-200")
    sys.exit(1)

rel_path = sys.argv[1]
run_dir = os.path.join(BASE_DIR, rel_path)
input_csv = os.path.join(run_dir, "experiment_log.csv")

if not os.path.isfile(input_csv):
    print(f"❌ Error: {input_csv} not found")
    sys.exit(1)

# --- Output paths (same folder as input) ---
peaks_csv = os.path.join(run_dir, "fall_local_maxima.csv")
summary_csv = os.path.join(run_dir, "speed_run_summary.csv")
plot_dir = run_dir

run_name = os.path.basename(run_dir)
material_name = run_name.replace("-", " ")

# --- Load CSV ---
cols = [
    "index", "timestamp", "seq", "ms",
    "motor_angle_deg", "motor_speed", "CH0_volts", "CH2_volts", "CH3_volts",
    "ellipse_angle_deg", "ellipse_area_px2", "frame_name",
    "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"
]
df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
df["ellipse_angle_derivative"] = df["ellipse_angle_deg"].diff()

# === STEP 1: Detect local maxima (Using find_peaks) ===

# --- PARAMETERS TO TUNE ---
# MIN_HEIGHT: The absolute minimum angle to be considered a peak (filters out noise near 0)
MIN_HEIGHT = 20  

# PROMINENCE: The vertical distance a peak must drop on either side to be counted.
# This is the most important setting. Since you had DERIVATIVE_THRESHOLD = -1, 
# a prominence of 1.5 or 2.0 degrees is a good starting point.
PROMINENCE = 3.5

# Find the indices of the peaks
peak_indices, properties = find_peaks(
    df["ellipse_angle_deg"], 
    height=MIN_HEIGHT, 
    prominence=PROMINENCE
)

# Create the result DataFrame using the found indices
result_df = df.iloc[peak_indices].copy()

# Apply your upper limit filter
result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

# Save to CSV
result_df[["index", "ellipse_angle_deg", "ellipse_angle_derivative", "motor_speed"]].to_csv(peaks_csv, index=False)
print(f"✅ Saved → {peaks_csv} ({len(result_df)} local maxima detected)")
# === STEP 2: Summarize runs ===
summary = None
if len(result_df) > 0:

    # Default rule for all speeds != 26
    result_df["speed_change"] = result_df["motor_speed"].ne(result_df["motor_speed"].shift())
    result_df["run_id"] = result_df["speed_change"].cumsum()

    # --- SPECIAL handling for speed == 26 ---
    mask26 = (result_df["motor_speed"] == 26)
    count26 = mask26.sum()

    if count26 > 0:
        print(f"🔧 Splitting {count26} peaks at 26 RPM into forward/backward halves")

        # row numbers of all 26-speed peaks
        idxs_26 = result_df.index[mask26].to_list()

        # split point
        half = count26 // 2

        # assign run IDs:
        # forward-26 = first half
        # backward-26 = second half
        run_forward = idxs_26[:half]
        run_backward = idxs_26[half:]

        # forward-26 → same run as previous speed
        forward_run_id = result_df.loc[run_forward[0], "run_id"]
        result_df.loc[run_forward, "run_id"] = forward_run_id

        # backward-26 → NEW run after forward
        backward_run_id = forward_run_id + 1
        result_df.loc[run_backward, "run_id"] = backward_run_id

        # increment all later run_ids by 1 to keep order intact
        result_df.loc[result_df.index > run_backward[-1], "run_id"] += 1

    # Now group by corrected run_id
    summary = (
        result_df.groupby("run_id")
        .agg(
            motor_speed=("motor_speed", "first"),
            num_points=("ellipse_angle_deg", "size"),
            angle_max=("ellipse_angle_deg", "max"),
            angle_min=("ellipse_angle_deg", "min"),
            angle_median=("ellipse_angle_deg", "median"),
            angle_mean=("ellipse_angle_deg", "mean"),
            angle_std=("ellipse_angle_deg", "std"),   # <— add this
            start_index=("index", "min"),
            end_index=("index", "max"),
        )
        .reset_index(drop=True)
    )

    summary.to_csv(summary_csv, index=False)
    print(f"✅ Saved → {summary_csv} ({len(summary)} speed runs summarized)")

else:
    print("⚠️ No valid peaks found — skipping summary.")


# === STEP 3: Plots (angle trace, sweep comparison) ===
if len(result_df) > 0:
    # 1️⃣ Scatter: angle vs speed
    plt.figure(figsize=(8, 5))
    plt.scatter(result_df["motor_speed"], result_df["ellipse_angle_deg"], alpha=0.7)
    plt.xlabel("Motor Speed (RPM)")
    plt.ylabel("Ellipse Angle (°)")
    plt.title(f"{material_name} — Angle of Repose vs Motor Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_name}_angle_vs_speed.png"))
    plt.close()

# 2️⃣ Trace: index vs angle (with best-fit line over the detected peaks)
plt.figure(figsize=(10, 5))

# Raw angle trace
plt.plot(df["index"], df["ellipse_angle_deg"], color="gray", lw=0.8, label="Raw Angle")

# Detected peaks
px = result_df["index"].values
py = result_df["ellipse_angle_deg"].values
plt.scatter(px, py, color="red", s=25, label="Detected Peaks")

# === Quadratic fit through detected peaks ===
if len(result_df) > 2:
    a, b, c = np.polyfit(px, py, 2)
    x_fit = np.linspace(px.min(), px.max(), 500)
    y_fit = a*x_fit**2 + b*x_fit + c

    fit_label = f"Quadratic Fit: y = {a:.3e}x² + {b:.3e}x + {c:.3f}"
    plt.plot(x_fit, y_fit, "--", lw=2, color="blue", label=fit_label)


plt.xlabel("Index")
plt.ylabel("Ellipse Angle (°)")
plt.title(f"{material_name} — Detected Angles of Repose")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{run_name}_angle_trace.png"))
plt.close()


# 3️⃣ Overlay: first vs second sweep
if summary is not None and len(summary) > 0:
    half = len(summary) // 2
    first_half = summary.iloc[:half]
    second_half = summary.iloc[half:]

    plt.figure(figsize=(8, 5))

    # ===== First sweep error bars =====
    plt.errorbar(
        first_half["motor_speed"],
        first_half["angle_mean"],
        yerr=first_half["angle_std"],
        marker="o",
        lw=2,
        capsize=4,
        label="First Sweep (1→26)",
    )

    # ===== Second sweep error bars =====
    plt.errorbar(
        second_half["motor_speed"],
        second_half["angle_mean"],
        yerr=second_half["angle_std"],
        marker="o",
        lw=2,
        capsize=4,
        label="Second Sweep (26→1)",
    )

    # ===== Linear fits =====
    import numpy as np

    # First sweep fit
    m1, b1 = np.polyfit(first_half["motor_speed"], first_half["angle_mean"], 1)
    x_fit1 = np.linspace(first_half["motor_speed"].min(),
                         first_half["motor_speed"].max(), 200)
    y_fit1 = m1 * x_fit1 + b1
    plt.plot(x_fit1, y_fit1, "--", lw=1.8,
             label=f"Fit 1→26: y = {m1:.3f}x + {b1:.3f}")

    # Second sweep fit
    m2, b2 = np.polyfit(second_half["motor_speed"], second_half["angle_mean"], 1)
    x_fit2 = np.linspace(second_half["motor_speed"].min(),
                         second_half["motor_speed"].max(), 200)
    y_fit2 = m2 * x_fit2 + b2
    plt.plot(x_fit2, y_fit2, "--", lw=1.8,
             label=f"Fit 26→1: y = {m2:.3f}x + {b2:.3f}")

    plt.xlabel("Motor Speed (RPM)")
    plt.ylabel("Mean Angle of Repose (°)")
    plt.title(material_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    overlay_path = os.path.join(plot_dir, f"{run_name}_sweep_comparison.png")
    plt.savefig(overlay_path)
    plt.close()
    print(f"📊 Saved → {overlay_path}")


# === STEP 4: Robust Peak Count (Split Top Speed) ===
if len(result_df) > 0:
    # 1. Define "Runs" based on raw data
    df["speed_change"] = df["motor_speed"].ne(df["motor_speed"].shift())
    df["run_id"] = df["speed_change"].cumsum()
    
    run_ranges = (
        df.groupby("run_id")
        .agg(
            motor_speed=("motor_speed", "first"),
            start_index=("index", "min"),
            end_index=("index", "max")
        )
        .reset_index()
    )

    # --- THE FIX: Force Split the Max Speed Run (26 RPM) ---
    # Find the run with the highest speed (assuming this is the turning point)
    max_speed_idx = run_ranges["motor_speed"].idxmax()
    max_speed_row = run_ranges.loc[max_speed_idx]
    
    # Only split if it exists as a single block (count == 1 for that speed in the summary)
    # This handles the case where the motor sits at 26 for a while
    if run_ranges["motor_speed"].value_counts()[max_speed_row["motor_speed"]] == 1:
        print(f"🔧 Splitting Run {max_speed_idx} (Speed {max_speed_row['motor_speed']}) into two halves...")
        
        # Calculate the midpoint index
        midpoint = int((max_speed_row["start_index"] + max_speed_row["end_index"]) / 2)
        
        # Create the Forward-26 half
        run_fwd = max_speed_row.copy()
        run_fwd["end_index"] = midpoint
        
        # Create the Backward-26 half
        run_bwd = max_speed_row.copy()
        run_bwd["start_index"] = midpoint + 1
        
        # Remove the original big run and insert the two halves
        run_ranges = run_ranges.drop(max_speed_idx)
        run_ranges = pd.concat([run_ranges, pd.DataFrame([run_fwd, run_bwd])], ignore_index=True)
        
        # Sort by start_index so the timeline is correct again
        run_ranges = run_ranges.sort_values("start_index").reset_index(drop=True)
        
        # Re-assign run_id to be sequential (1, 2, 3...)
        run_ranges["run_id"] = range(1, len(run_ranges) + 1)

    # 2. Bin the Peaks (The Red Dots)
    peak_indices = result_df["index"].values
    
    # Use searchsorted to find which time-bucket every peak falls into
    # (This works perfectly now because we have two distinct 26 buckets)
    bins = np.searchsorted(run_ranges["end_index"].values, peak_indices, side="left")
    
    # Map peaks to runs
    # Note: we use clip just in case a peak falls exactly on the last edge
    bins = np.clip(bins, 0, len(run_ranges) - 1)
    result_df["mapped_run_id"] = run_ranges.loc[bins, "run_id"].values
    
    # Count peaks per run
    counts = result_df["mapped_run_id"].value_counts().reindex(run_ranges["run_id"], fill_value=0)
    run_ranges["num_peaks"] = counts.values

    # 3. Split into Sweep 1 (Forward) and Sweep 2 (Backward)
    # Now that 26 is split, we should have an EVEN number of runs (e.g. 1-26 and 26-1)
    half_point = len(run_ranges) // 2
    
    first_sweep = run_ranges.iloc[:half_point].copy()
    second_sweep = run_ranges.iloc[half_point:].copy()
    
    # Merge for plotting
    merged = pd.merge(
        first_sweep[["motor_speed", "num_peaks"]],
        second_sweep[["motor_speed", "num_peaks"]],
        on="motor_speed",
        how="outer",
        suffixes=("_first", "_second")
    ).fillna(0).sort_values("motor_speed")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.4
    x = np.arange(len(merged))
    
    ax.bar(x - bar_width/2, merged["num_peaks_first"], width=bar_width, color="tab:cyan", alpha=0.8, label="Peaks (Forward)")
    ax.bar(x + bar_width/2, merged["num_peaks_second"], width=bar_width, color="tab:orange", alpha=0.8, label="Peaks (Backward)")
    
    # Add text labels
    for i, v in enumerate(merged["num_peaks_first"]):
        if v > 0: ax.text(i - bar_width/2, v + 0.5, int(v), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(merged["num_peaks_second"]):
        if v > 0: ax.text(i + bar_width/2, v + 0.5, int(v), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(merged["motor_speed"].astype(int))
    ax.set_xlabel("Motor Speed (RPM)")
    ax.set_ylabel("Count of Peaks")
    ax.set_title(f"{material_name} — Peak Count per Speed Run")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    peak_plot_path = os.path.join(plot_dir, f"{run_name}_peak_counts.png")
    plt.savefig(peak_plot_path)
    plt.close()
    print(f"📊 Saved → {peak_plot_path}")

# === STEP 5: Advanced Charge Analysis (Metrics per Run) ===
if 'run_ranges' in locals() and len(run_ranges) > 0:
    print(f"⚡ Analyzing Charge Data across {len(run_ranges)} runs...")

    # 1. Establish Baseline (The "Empty Space" Voltage)
    # We use a median filter because the probe spends most time looking at "nothing".
    # The median ignores the particle spikes and finds the true DC offset.
    kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE)
    if kernel_size % 2 == 0: kernel_size += 1
    
    print(f"   → Applying rolling median baseline (window={kernel_size})")
    df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
    df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
    
    # Create Clean Signals
    df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
    df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

    # 2. Define a "Noise Floor" for the Active Fraction
    # Any signal below this absolute voltage is considered "Empty Space"
    # You can tune this based on your sensor noise. 0.005V (5mV) is a safe guess.
    NOISE_THRESHOLD = 0.005 

    charge_stats = []

    # 3. Iterate through the SAME runs we defined in Step 4
    for _, row in run_ranges.iterrows():
        # Slice the clean data for this specific speed run
        mask = (df["index"] >= row["start_index"]) & (df["index"] <= row["end_index"])
        segment = df.loc[mask]
        
        # If segment is empty, skip
        if len(segment) == 0: continue

        # --- METRIC 1: Standard Deviation (The "Energy") ---
        # High STD = Lots of large particles or very frequent hits
        ch2_std = segment["CH2_clean"].std()
        ch3_std = segment["CH3_clean"].std()

        # --- METRIC 2: Robust Peak-to-Peak (99th - 1st Percentile) ---
        # We use percentiles to ignore single cosmic ray/spark outliers
        ch2_range = np.percentile(segment["CH2_clean"], 99) - np.percentile(segment["CH2_clean"], 1)
        ch3_range = np.percentile(segment["CH3_clean"], 99) - np.percentile(segment["CH3_clean"], 1)

        # --- METRIC 3: Active Fraction (Time Spent Reading Particles) ---
        # What % of data points were louder than the noise floor?
        ch2_active_count = (segment["CH2_clean"].abs() > NOISE_THRESHOLD).sum()
        ch2_active_pct = (ch2_active_count / len(segment)) * 100
        
        ch3_active_count = (segment["CH3_clean"].abs() > NOISE_THRESHOLD).sum()
        ch3_active_pct = (ch3_active_count / len(segment)) * 100

        charge_stats.append({
            "motor_speed": row["motor_speed"],
            "run_id": row["run_id"],
            "ch2_std": ch2_std,
            "ch3_std": ch3_std,
            "ch2_p2p": ch2_range,
            "ch3_p2p": ch3_range,
            "ch2_active_pct": ch2_active_pct,
            "ch3_active_pct": ch3_active_pct
        })

    charge_df = pd.DataFrame(charge_stats)
    
    # Save detailed stats
    charge_csv = os.path.join(run_dir, "charge_analysis.csv")
    charge_df.to_csv(charge_csv, index=False)
    print(f"✅ Saved → {charge_csv}")

# === STEP 5: Advanced Charge Analysis (Two Plots) ===
if 'run_ranges' in locals() and len(run_ranges) > 0:
    print(f"⚡ Analyzing Charge Data across {len(run_ranges)} runs...")

    # 1. Establish Baseline & Clean Data
    kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE)
    if kernel_size % 2 == 0: kernel_size += 1
    
    df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
    df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
    df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
    df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

    # 2. Calculate Stats per Run
    NOISE_THRESHOLD = 0.005 
    charge_stats = []

    for _, row in run_ranges.iterrows():
        mask = (df["index"] >= row["start_index"]) & (df["index"] <= row["end_index"])
        segment = df.loc[mask]
        if len(segment) == 0: continue

        # Calculate Metrics
        ch2_std = segment["CH2_clean"].std()
        ch3_std = segment["CH3_clean"].std()
        
        # Active Fraction
        ch2_active_pct = ((segment["CH2_clean"].abs() > NOISE_THRESHOLD).sum() / len(segment)) * 100
        ch3_active_pct = ((segment["CH3_clean"].abs() > NOISE_THRESHOLD).sum() / len(segment)) * 100

        charge_stats.append({
            "motor_speed": row["motor_speed"],
            "run_id": row["run_id"],
            "ch2_std": ch2_std,
            "ch3_std": ch3_std,
            "ch2_active_pct": ch2_active_pct,
            "ch3_active_pct": ch3_active_pct
        })

    charge_df = pd.DataFrame(charge_stats)
    charge_csv = os.path.join(run_dir, "charge_analysis.csv")
    charge_df.to_csv(charge_csv, index=False)
    print(f"✅ Saved → {charge_csv}")

    # Split sweeps
    half = len(charge_df) // 2
    first_sweep = charge_df.iloc[:half]
    second_sweep = charge_df.iloc[half:]

    # ==========================================
    # PLOT 1: RAW DATA (Visual Trends Only)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # CH2 (Blue/Cyan)
    ax1.plot(first_sweep["motor_speed"], first_sweep["ch2_std"], 
            marker='o', color='tab:blue', label="CH2 Forward")
    ax1.plot(second_sweep["motor_speed"], second_sweep["ch2_std"], 
            marker='o', linestyle='--', color='tab:cyan', label="CH2 Backward")
    
    # CH3 (Red/Orange)
    ax1.plot(first_sweep["motor_speed"], first_sweep["ch3_std"], 
            marker='s', color='tab:red', label="CH3 Forward")
    ax1.plot(second_sweep["motor_speed"], second_sweep["ch3_std"], 
            marker='s', linestyle='--', color='tab:orange', label="CH3 Backward")

    ax1.set_xlabel("Motor Speed (RPM)")
    ax1.set_ylabel("Signal Energy (Std Dev)")
    ax1.set_title(f"{material_name} — Charge Activity (Raw Data)")
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    plt.tight_layout()
    
    raw_path = os.path.join(plot_dir, f"{run_name}_charge_raw.png")
    plt.savefig(raw_path)
    plt.close(fig1)
    print(f"📊 Saved → {raw_path}")

    # ==========================================
    # PLOT 2: LINES OF BEST FIT (Equations)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Helper to plot ONLY the line
    def plot_trend(x, y, color, label_name):
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = m * x_fit + b
            
            sign = "+" if b >= 0 else "-"
            eq = f"{label_name}: y = {m:.5f}x {sign} {abs(b):.4f}"
            
            ax2.plot(x_fit, y_fit, linewidth=2, color=color, label=eq)
            # Optional: Faint scatter dots so you can see what the line is fitting to
            ax2.scatter(x, y, color=color, alpha=0.15, s=10)

    # Plot the 4 lines
    plot_trend(first_sweep["motor_speed"], first_sweep["ch2_std"], 'tab:blue', "CH2 Fwd")
    plot_trend(second_sweep["motor_speed"], second_sweep["ch2_std"], 'tab:cyan', "CH2 Bwd")
    plot_trend(first_sweep["motor_speed"], first_sweep["ch3_std"], 'tab:red', "CH3 Fwd")
    plot_trend(second_sweep["motor_speed"], second_sweep["ch3_std"], 'tab:orange', "CH3 Bwd")

    ax2.set_xlabel("Motor Speed (RPM)")
    ax2.set_ylabel("Linear Trend (Std Dev)")
    ax2.set_title(f"{material_name} — Charge Trends & Equations")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.5)
    plt.tight_layout()
    
    fit_path = os.path.join(plot_dir, f"{run_name}_charge_fits.png")
    plt.savefig(fit_path)
    plt.close(fig2)
    print(f"📊 Saved → {fit_path}")

else:
    print("⚠️ Skipping Charge Analysis (Run Ranges not defined in Step 4)")