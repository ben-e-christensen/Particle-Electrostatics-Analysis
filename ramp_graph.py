import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# --- FONTS & STYLING ---
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"

# --- DATA CONFIGURATION ---
speeds_up = [1, 6, 11, 16, 21, 26]
speeds_down = [26, 21, 16, 11, 6, 1]
rpm_sequence = speeds_up + speeds_down

step_duration = 5  # minutes per RPM stage

# We need the time points to define the boundaries of each 5-minute block.
# (12 blocks = 13 time boundaries, from 0 to 60 minutes)
time_points = [i * step_duration for i in range(len(rpm_sequence) + 1)]

# We append a '0' to the end of the sequence so the plot drops off after the final 5 mins
y_vals = rpm_sequence + [0]

# --- PLOT SETUP ---
fig, ax = plt.subplots(figsize=(12, 5))

# The 'post' argument tells it to hold the value *after* the x-coordinate starts, 
# creating the flat top before dropping/jumping at the next x-coordinate.
ax.step(time_points, y_vals, where='post', color='black', linewidth=2.5)

# --- BACKGROUND SHADING (Visual Polish) ---
# Highlight the Ramp Up (0 to 30 mins) and Ramp Down (30 to 60 mins)
ax.axvspan(0, 30, color='#4A90E2', alpha=0.15, label="Acceleration Phase")
ax.axvspan(30, 60, color='#F39C12', alpha=0.15, label="Deceleration Phase")

# Adding a dashed line to show the split point clearly
ax.axvline(30, color='gray', linestyle='--', linewidth=1.5)

# --- FORMATTING ---
# ax.set_title("System Operational Speed Profile", fontsize=18, fontweight='bold', pad=15)
ax.set_xlabel("Time (Minutes)", fontsize=14, fontweight='bold')
ax.set_ylabel("Speed (RPM)", fontsize=14, fontweight='bold')

# Set ticks to exactly match your data points for maximum clarity
ax.set_xticks(time_points)

# Removed the 0 from this list!
ax.set_yticks([1, 6, 11, 16, 21, 26])

ax.tick_params(axis='both', labelsize=12)
ax.grid(True, axis='y', linestyle=':', alpha=0.7)

# Ensure the plot limits hug the data cleanly (keeping y_ylim at 0 keeps the baseline visible)
ax.set_xlim(0, 60)
ax.set_ylim(0, 29)

# Clean up the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add Legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig("RPM_Step_Profile.png", dpi=300, bbox_inches='tight')
plt.show()