import pandas as pd

# --- Input/Output ---
input_csv = "fall_local_maxima.csv"   # your detected peaks
output_csv = "speed_run_summary.csv"

# --- Load ---
df = pd.read_csv(input_csv)

# --- Identify contiguous segments where motor_speed stays constant ---
df["speed_change"] = df["motor_speed"].ne(df["motor_speed"].shift())
df["run_id"] = df["speed_change"].cumsum()

# --- Compute stats for each run ---
summary = (
    df.groupby("run_id")
    .agg(
        motor_speed=("motor_speed", "first"),
        num_points=("ellipse_angle_deg", "size"),
        angle_max=("ellipse_angle_deg", "max"),
        angle_min=("ellipse_angle_deg", "min"),
        angle_median=("ellipse_angle_deg", "median"),
        angle_mean=("ellipse_angle_deg", "mean"),
    )
    .reset_index(drop=True)
)

# --- Save results ---
summary.to_csv(output_csv, index=False)

print(f"Saved → {output_csv} ({len(summary)} speed segments summarized)")
