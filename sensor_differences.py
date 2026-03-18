import os
import pandas as pd

# Updated parent directory
parent_dir = "D:/particle-data/sensor-readings" 

folders = [
    "Just-acrylic-again",
    "acetal-and-acrylic",
    "nylon-and-acrylic",
    "teflon-and-acrylic"
]

# Map your raw folder names to the clean display names for the LaTeX table
# Map your raw folder names to the clean display names for the LaTeX table
name_map = {
    "Just-acrylic-again": "Acrylic",
    "acetal-and-acrylic": "Acetal and Acrylic",
    "nylon-and-acrylic": "Nylon and Acrylic",
    "teflon-and-acrylic": "Teflon and Acrylic"
}

results = []

for folder in folders:
    file_path = os.path.join(parent_dir, folder, "experiment_log.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        if 'CH2_volts' in df.columns:
            # Drop NaNs and convert ms to seconds
            clean_data = df[['ms', 'CH2_volts']].dropna().copy()
            clean_data['time_s'] = clean_data['ms'] / 1000.0
            
            # Create 1-second bins (1Hz wave) across the ENTIRE recording
            clean_data['second_bin'] = (clean_data['time_s'] // 1).astype(int)
            
            # Find max and min for each 1-second window
            peaks = clean_data.groupby('second_bin')['CH2_volts'].agg(['max', 'min']).reset_index()
            peaks['p2p'] = peaks['max'] - peaks['min']
            
            # Calculate Average and Stability (Standard Deviation %)
            avg_p2p = peaks['p2p'].mean()
            std_p2p = peaks['p2p'].std()
            cv_percent = (std_p2p / avg_p2p) * 100 if avg_p2p != 0 else 0
            
            results.append({
                'Material': folder, 
                'Avg P2P (V)': avg_p2p,
                'Stability Error (%)': cv_percent
            })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Calculate Attenuation (Percentage Difference from Baseline)
baseline_row = results_df[results_df['Material'] == 'Just-acrylic-again'] # <-- Updated this line!

if not baseline_row.empty:
    baseline_val = baseline_row['Avg P2P (V)'].values[0]
    # Calculate % difference
    results_df['Attenuation (%)'] = ((results_df['Avg P2P (V)'] - baseline_val) / baseline_val) * 100
else:
    results_df['Attenuation (%)'] = 0.0

# Print raw terminal table for quick review
print("\n--- Final Peak-to-Peak Analysis ---")
pd.set_option('display.float_format', lambda x: '%.5f' % x)
print(results_df.to_string(index=False))

# Generate the exact LaTeX rows you requested
print("\n\n--- LaTeX Data Rows ---")
for index, row in results_df.iterrows():
    # Fetch clean name from map, or format it nicely if not in map
    mat = name_map.get(row['Material'], row['Material'].replace('-', ' ').title())
    
    # 5 decimal places for voltage, 2 for percentages
    p2p = f"{row['Avg P2P (V)']:.5f}"
    stab = f"{row['Stability Error (%)']:.2f}"
    atten = f"{row['Attenuation (%)']:.2f}"
    
    print(f"        {mat} & {p2p} & {stab} & {atten} \\\\")