import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================
# CYBERSNIFF
# Visualize Normal Industrial Sensor Data
# ============================================

# Find project root
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Dataset path
data_path = os.path.join(
    project_root,
    "data",
    "raw",
    "normal_sensor_data.csv"
)

# Load dataset
data = pd.read_csv(data_path)

# Convert timestamp
data["timestamp"] = pd.to_datetime(data["timestamp"])


# ============================================
# Create one dashboard-style figure
# ============================================

fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# Water Level
axes[0, 0].plot(
    data["timestamp"],
    data["water_level"]
)
axes[0, 0].set_title("Water Level")
axes[0, 0].set_ylabel("Level (%)")
axes[0, 0].grid(True)

# Flow Rate
axes[0, 1].plot(
    data["timestamp"],
    data["flow_rate"]
)
axes[0, 1].set_title("Flow Rate")
axes[0, 1].set_ylabel("Flow (L/min)")
axes[0, 1].grid(True)

# Pressure
axes[1, 0].plot(
    data["timestamp"],
    data["pressure"]
)
axes[1, 0].set_title("Pressure")
axes[1, 0].set_ylabel("Pressure (bar)")
axes[1, 0].grid(True)

# Temperature
axes[1, 1].plot(
    data["timestamp"],
    data["temperature"]
)
axes[1, 1].set_title("Temperature")
axes[1, 1].set_ylabel("Temperature (°C)")
axes[1, 1].grid(True)

# Pump Current
axes[2, 0].plot(
    data["timestamp"],
    data["pump_current"]
)
axes[2, 0].set_title("Pump Current")
axes[2, 0].set_ylabel("Current (A)")
axes[2, 0].grid(True)

# Vibration
axes[2, 1].plot(
    data["timestamp"],
    data["vibration"]
)
axes[2, 1].set_title("Pump Vibration")
axes[2, 1].set_ylabel("Vibration")
axes[2, 1].grid(True)

# Overall title
fig.suptitle(
    "CyberSniff - Normal Industrial Pump Operation",
    fontsize=16
)

plt.tight_layout()

plt.show()

print("Visualization completed successfully!")