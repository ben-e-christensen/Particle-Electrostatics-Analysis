"""
detect_stuck_sensor_events.py

Scans the same day1..day7 folder structure as continuous_ensemble.py and finds
stretches where CH2_volts / CH3_volts got "stuck" (a frozen/near-constant reading
for an extended period) -- the flat horizontal lines you circled on the plot.

This is a DIFFERENT failure mode than the existing "chassis arc blackout" filter
in continuous_ensemble.py (that one catches a sudden voltage FLOOR DROP; this one
catches a value that stops changing entirely).

Usage:
    python detect_stuck_sensor_events.py <path_to_PARENT_folder>

Output:
    - Printed report of detected frozen windows (per channel + merged)
    - A file `exclusion_windows.py` written into the parent folder, containing
      a ready-to-paste EXCLUSION_WINDOWS_HOURS constant.
"""

import pandas as pd
import numpy as np
import sys
import os

# ================= CONFIGURATION =================
VOLTAGE_COL_CH2 = "CH2_volts"
VOLTAGE_COL_CH3 = "CH3_volts"

# A "frozen" sample is one where the rolling std over WINDOW_SAMPLES is below
# FREEZE_STD_THRESHOLD. Tune these if you get too many/few hits.
WINDOW_SAMPLES = 500          # rolling window size, in raw samples
FREEZE_STD_THRESHOLD = 5e-4   # volts; real sensor noise should exceed this easily

# Only keep frozen runs that last at least this long (filters out normal calm
# periods that just happen to be quiet for a few seconds).
MIN_DURATION_HOURS = 0.02     # ~72 seconds

# When merging per-channel windows, gaps smaller than this get bridged together
# into one continuous exclusion window.
MERGE_GAP_HOURS = 0.05        # ~3 minutes

OUTPUT_CONST_FILE = "exclusion_windows.py"
# =================================================


def find_frozen_runs(times_hours, values, window_samples, std_threshold, min_duration_hours):
    """Returns list of (start_hour, end_hour) for stretches where the rolling
    std of `values` stays below `std_threshold` for at least `min_duration_hours`."""
    if len(values) < window_samples:
        return []

    s = pd.Series(values)
    rolling_std = s.rolling(window=window_samples, min_periods=window_samples, center=True).std()
    frozen_mask = (rolling_std < std_threshold).fillna(False).to_numpy()

    runs = []
    in_run = False
    run_start_idx = None

    for idx, flag in enumerate(frozen_mask):
        if flag and not in_run:
            in_run = True
            run_start_idx = idx
        elif not flag and in_run:
            in_run = False
            run_end_idx = idx - 1
            start_hr = times_hours[run_start_idx]
            end_hr = times_hours[run_end_idx]
            if (end_hr - start_hr) >= min_duration_hours:
                runs.append((start_hr, end_hr))

    if in_run:
        run_end_idx = len(frozen_mask) - 1
        start_hr = times_hours[run_start_idx]
        end_hr = times_hours[run_end_idx]
        if (end_hr - start_hr) >= min_duration_hours:
            runs.append((start_hr, end_hr))

    return runs


def merge_windows(windows, gap_hours):
    """Merge overlapping/near-adjacent (start, end) windows."""
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap_hours:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def process(parent_dir):
    print(f"--- Scanning for stuck/frozen sensor events in: {parent_dir} ---")

    cols = ["index", "timestamp", "seq", "ms", "motor_angle_deg", "motor_speed",
            "CH0_volts", "CH2_volts", "CH3_volts", "ellipse_angle_deg",
            "ellipse_area_px2", "frame_name", "ch2_dv/dt", "ch3_dv/dt",
            "ch2_flag", "ch3_flag"]

    cumulative_ms = 0.0
    ch2_runs_all = []
    ch3_runs_all = []
    per_day_report = []

    for i in range(1, 8):
        day_folder = os.path.join(parent_dir, f"day{i}")
        if not os.path.exists(day_folder):
            continue

        csv_files = sorted(f for f in os.listdir(day_folder) if f.endswith('.csv'))
        if not csv_files:
            continue

        print(f"  Reading Day {i}...")

        for csv_file in csv_files:
            csv_path = os.path.join(day_folder, csv_file)
            try:
                df = pd.read_csv(csv_path, names=cols, header=0, on_bad_lines="skip",
                                  dtype={"ms": "float32", "CH2_volts": "float32",
                                         "CH3_volts": "float32", "ellipse_angle_deg": "float32"})
                if df.empty:
                    continue

                # Identical time-stitching logic to continuous_ensemble.py
                start_ms = df["ms"].min()
                df["Continuous_ms"] = (df["ms"] - start_ms) + cumulative_ms
                cumulative_ms = df["Continuous_ms"].max() + 10.0
                df["rel_time_hours"] = df["Continuous_ms"] / 3600000.0
                df.sort_values(by="Continuous_ms", inplace=True)

                df_volt = df.dropna(subset=[VOLTAGE_COL_CH2, VOLTAGE_COL_CH3])
                if df_volt.empty:
                    continue

                # Same CH2 sign flip as the main script, so hour ranges line up
                # with what you see plotted.
                ch2 = pd.to_numeric(df_volt[VOLTAGE_COL_CH2], errors='coerce') * -1.0
                ch3 = pd.to_numeric(df_volt[VOLTAGE_COL_CH3], errors='coerce')
                times = df_volt["rel_time_hours"].to_numpy()

                ch2_runs = find_frozen_runs(times, ch2.to_numpy(), WINDOW_SAMPLES,
                                             FREEZE_STD_THRESHOLD, MIN_DURATION_HOURS)
                ch3_runs = find_frozen_runs(times, ch3.to_numpy(), WINDOW_SAMPLES,
                                             FREEZE_STD_THRESHOLD, MIN_DURATION_HOURS)

                if ch2_runs:
                    per_day_report.append((i, csv_file, "CH2", ch2_runs))
                if ch3_runs:
                    per_day_report.append((i, csv_file, "CH3", ch3_runs))

                ch2_runs_all.extend(ch2_runs)
                ch3_runs_all.extend(ch3_runs)

            except Exception as e:
                print(f"    Skipping {csv_file}: {e}")

    print("\n--- Per-file detections ---")
    if per_day_report:
        for day_num, fname, chan, runs in per_day_report:
            for start_hr, end_hr in runs:
                print(f"  Day {day_num} | {fname} | {chan}: {start_hr:.4f}h -> {end_hr:.4f}h "
                      f"(duration {(end_hr - start_hr) * 60:.1f} min)")
    else:
        print("  No frozen stretches found with current thresholds.")

    merged_ch2 = merge_windows(ch2_runs_all, MERGE_GAP_HOURS)
    merged_ch3 = merge_windows(ch3_runs_all, MERGE_GAP_HOURS)
    # Union across both channels -- if either channel froze, exclude that window
    # from everything (angle + both voltage plots), matching how the blackout
    # filter is applied globally in continuous_ensemble.py.
    merged_all = merge_windows(merged_ch2 + merged_ch3, MERGE_GAP_HOURS)

    print("\n--- Merged CH2 windows (hours) ---")
    for w in merged_ch2:
        print(f"  {w[0]:.4f} -> {w[1]:.4f}")

    print("\n--- Merged CH3 windows (hours) ---")
    for w in merged_ch3:
        print(f"  {w[0]:.4f} -> {w[1]:.4f}")

    print("\n--- FINAL merged exclusion windows (union, hours) ---")
    for w in merged_all:
        print(f"  {w[0]:.4f} -> {w[1]:.4f}")

    # Write out a ready-to-paste constant
    out_path = os.path.join(parent_dir, OUTPUT_CONST_FILE)
    with open(out_path, "w") as f:
        f.write("# Auto-generated by detect_stuck_sensor_events.py\n")
        f.write("# Paste EXCLUSION_WINDOWS_HOURS into continuous_ensemble.py and filter\n")
        f.write("# any row whose rel_time_hours falls inside one of these (start, end) pairs.\n\n")
        f.write("EXCLUSION_WINDOWS_HOURS = [\n")
        for start_hr, end_hr in merged_all:
            f.write(f"    ({start_hr:.4f}, {end_hr:.4f}),\n")
        f.write("]\n")

    print(f"\nWrote constants to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_stuck_sensor_events.py <path_to_PARENT_folder>")
    else:
        process(sys.argv[1])