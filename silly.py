import pandas as pd

input_csv = "experiment_log.csv"
output_csv = "ellipse_angle_derivative.csv"

# Define the correct header explicitly
cols = [
    "index", "timestamp", "seq", "ms",
    "motor_angle_deg", "motor_speed", "CH0_volts", "CH2_volts", "CH3_volts",
    "ellipse_angle_deg", "ellipse_area_px2", "frame_name",
    "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"
]

# Read using explicit column names, skip malformed rows
df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")

# Compute derivative of ellipse angle
df["ellipse_angle_derivative"] = df["ellipse_angle_deg"].diff()

# Filter rows where derivative is nonzero and not NaN
df_filtered = df[df["ellipse_angle_derivative"].ne(0) & df["ellipse_angle_derivative"].notna()]

# Select relevant columns
out = df_filtered[["index", "ellipse_angle_deg", "ellipse_angle_derivative"]]
out.to_csv(output_csv, index=False)

print(f"Saved → {output_csv} ({len(out)} rows kept)")
