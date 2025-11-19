#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.signal import medfilt

# === CONFIG ===
BASE_DIR = "/media/ben/SANDISK/particle-data"
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

# === STEP 1: Detect local maxima ===
peaks = []
i = 1
while i < len(df):
    if df.loc[i, "ellipse_angle_derivative"] <= DERIVATIVE_THRESHOLD:
        j = i - 1
        while j > 0 and df.loc[j - 1, "ellipse_angle_deg"] >= df.loc[j, "ellipse_angle_deg"]:
            j -= 1
        peaks.append(df.loc[j])
        while i < len(df) - 1 and df.loc[i + 1, "ellipse_angle_deg"] <= df.loc[i, "ellipse_angle_deg"]:
            i += 1
    i += 1

result_df = pd.DataFrame(peaks).drop_duplicates(subset="index")
result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

result_df[["index", "ellipse_angle_deg", "ellipse_angle_derivative", "motor_speed"]].to_csv(peaks_csv, index=False)
print(f"✅ Saved → {peaks_csv} ({len(result_df)} local maxima)")

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

    # 2️⃣ Trace: index vs angle
    plt.figure(figsize=(10, 5))
    plt.plot(df["index"], df["ellipse_angle_deg"], color="gray", lw=0.8, label="Raw Angle")
    plt.scatter(result_df["index"], result_df["ellipse_angle_deg"], color="red", s=25, label="Detected Peaks")
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


# === STEP 4: Peak Count per Run ===
if summary is not None and len(summary) > 0:
    peaks_df = pd.read_csv(peaks_csv)
    peaks_df = peaks_df[peaks_df["ellipse_angle_deg"] <= 70]
    summary = summary.copy()
    summary["run_id"] = np.arange(1, len(summary) + 1)
    peaks_df["run_id"] = np.searchsorted(summary["end_index"].values, peaks_df["index"].values, side="right") + 1

    peak_counts = (
        peaks_df.groupby("run_id")["ellipse_angle_deg"]
        .count()
        .reset_index(name="num_peaks")
    )
    peak_summary = pd.merge(summary, peak_counts, on="run_id", how="left").fillna(0)

    half = len(peak_summary) // 2
    first_sweep = peak_summary.iloc[:half].copy()
    second_sweep = peak_summary.iloc[half:].copy()
    merged = pd.merge(
        first_sweep[["motor_speed", "num_peaks"]],
        second_sweep[["motor_speed", "num_peaks"]],
        on="motor_speed",
        how="outer",
        suffixes=("_first", "_second")
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.4
    x = np.arange(len(merged))
    ax.bar(x - bar_width/2, merged["num_peaks_first"], width=bar_width, color="tab:cyan", alpha=0.8, label="Peaks (1→25)")
    ax.bar(x + bar_width/2, merged["num_peaks_second"], width=bar_width, color="tab:orange", alpha=0.8, label="Peaks (25→1)")
    for i, v in enumerate(merged["num_peaks_first"]):
        ax.text(i - bar_width/2, v + 2, int(v), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(merged["num_peaks_second"]):
        ax.text(i + bar_width/2, v + 2, int(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["motor_speed"].astype(int))
    ax.set_xlabel("Motor Speed (RPM)")
    ax.set_ylabel("Peak Count")
    ax.set_title(f"{material_name} — Peak Count per Speed")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    peak_plot_path = os.path.join(plot_dir, f"{run_name}_peak_counts.png")
    plt.savefig(peak_plot_path)
    plt.close()
    print(f"📊 Saved → {peak_plot_path}")

# === STEP 5: Charge Sum per Run (with baseline removal) ===
if summary is not None and len(summary) > 0:
    # Estimate baseline using rolling median filter
    kernel_size = int(BASELINE_WINDOW_SEC * SAMPLE_RATE)
    if kernel_size % 2 == 0:
        kernel_size += 1
    print(f"🧮 Using baseline window = {kernel_size} samples (~{BASELINE_WINDOW_SEC}s)")

    df["CH2_baseline"] = medfilt(df["CH2_volts"], kernel_size)
    df["CH3_baseline"] = medfilt(df["CH3_volts"], kernel_size)
    df["CH2_clean"] = df["CH2_volts"] - df["CH2_baseline"]
    df["CH3_clean"] = df["CH3_volts"] - df["CH3_baseline"]

    charge_summary = []
    for _, row in summary.iterrows():
        mask = (df["index"] >= row["start_index"]) & (df["index"] <= row["end_index"])
        subset = df.loc[mask]
        total_charge = subset["CH2_clean"].abs().sum() + subset["CH3_clean"].abs().sum()
        charge_summary.append({
            "motor_speed": row["motor_speed"],
            "total_charge": total_charge
        })

    charge_df = pd.DataFrame(charge_summary)
    half = len(charge_df) // 2
    first_charge = charge_df.iloc[:half].copy()
    second_charge = charge_df.iloc[half:].copy()
    merged_charge = pd.merge(
        first_charge, second_charge, on="motor_speed", how="outer", suffixes=("_first", "_second")
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.4
    x = np.arange(len(merged_charge))
    ax.bar(x - bar_width/2, merged_charge["total_charge_first"], width=bar_width, color="tab:blue", alpha=0.8, label="Charge (1→25)")
    ax.bar(x + bar_width/2, merged_charge["total_charge_second"], width=bar_width, color="tab:orange", alpha=0.8, label="Charge (25→1)")
    ax.set_xticks(x)
    ax.set_xticklabels(merged_charge["motor_speed"].astype(int))
    ax.set_xlabel("Motor Speed (RPM)")
    ax.set_ylabel("Σ |Charge| (volts)")
    ax.set_title(f"{material_name} — Total Absolute Charge per Speed (baseline corrected)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    charge_path = os.path.join(plot_dir, f"{run_name}_charge_sum.png")
    plt.savefig(charge_path)
    plt.close()
    print(f"📊 Saved → {charge_path}")

print("✅ Processing complete.")
