import pandas as pd

# 1. Define your file and column names
file_path = 'experiment_log.csv'   # Change this to your CSV file path
value_column = 'ellipse_angle_deg'       # Change this to the name of the column you want the derivative of
new_column_name = 'dA/dt'

# 2. Read the CSV file, selecting only the columns you need for efficiency
try:
    df = pd.read_csv(file_path, usecols=[value_column])
except ValueError as e:
    print(f"Error: Could not find one or both columns. Please check your column names and file path.")
    print(f"Original error: {e}")
    exit()

# 3. Calculate the change (Delta) for both the value and time columns
delta_value = df[value_column].diff()
delta_time = 1.0/100.0

# 4. Calculate the Derivative (Rate of Change)
# Derivative = Delta_Value / Delta_Time
df[new_column_name] = delta_value 

# 5. Display the result
print("--- DataFrame with Calculated Derivative ---")
print(df)

# Optional: Save the results to a new CSV file
# df.to_csv('data_with_derivative.csv', index=False)