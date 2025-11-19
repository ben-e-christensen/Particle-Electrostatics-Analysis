import pandas as pd

# --- Config ---
input_csv = "ellipse_angle_derivative.csv"
output_csv = "derivative_less_than_neg1.csv"

df = pd.read_csv(input_csv, header=0, on_bad_lines="skip", engine="python")

# --- Compute derivative of ellipse angle ---
df["ellipse_angle_derivative"] = df["ellipse_angle_deg"].diff()

# --- Filter where derivative < -1 ---
df_filtered = df[df["ellipse_angle_derivative"] < -1]

# --- Show results ---
print(df_filtered[["index", "ellipse_angle_deg", "ellipse_angle_derivative"]])

# --- Save to file ---
df_filtered.to_csv(output_csv, index=False)
print(f"Saved → {output_csv}")
