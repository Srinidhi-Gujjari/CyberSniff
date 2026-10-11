import numpy as np
import pandas as pd
import os


# ============================================================
# CYBERSNIFF - INDUSTRIAL SENSOR SIMULATOR
# ============================================================
#
# Simulates an industrial water pumping system.
#
# Scenarios:
#   1. Normal Operation
#   2. Physical Pump Fault
#   3. False Data Injection (FDI) Cyber Attack
#
# Later we can add:
#   4. Sensor Spoofing
#   5. Replay Attack
#   6. Communication Tampering
#
# ============================================================


# ============================================================
# 1. NORMAL DATA GENERATION
# ============================================================

def generate_normal_data(num_samples=10000, seed=42):

    np.random.seed(seed)

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    timestamps = pd.date_range(
        start="2026-01-01 10:00:00",
        periods=num_samples,
        freq="s"
    )

    time = np.arange(num_samples)

    # --------------------------------------------------------
    # 1. Water Level
    # --------------------------------------------------------

    water_level = (
        70
        + 5 * np.sin(2 * np.pi * time / 1000)
        + np.random.normal(0, 0.3, num_samples)
    )

    water_level = np.clip(
        water_level,
        0,
        100
    )

    # --------------------------------------------------------
    # 2. Flow Rate
    # --------------------------------------------------------

    flow_rate = (
        1.8
        + 0.05 * np.sin(2 * np.pi * time / 500)
        + np.random.normal(0, 0.02, num_samples)
    )

    flow_rate = np.clip(
        flow_rate,
        0,
        None
    )

    # --------------------------------------------------------
    # 3. Pressure
    # --------------------------------------------------------

    pressure = (
        4.2
        + 0.25 * (flow_rate - 1.8)
        + np.random.normal(0, 0.03, num_samples)
    )

    # --------------------------------------------------------
    # 4. Temperature
    # --------------------------------------------------------

    temperature = (
        32
        + 1.5 * np.sin(2 * np.pi * time / 2000)
        + np.random.normal(0, 0.1, num_samples)
    )

    # --------------------------------------------------------
    # 5. Pump Current
    # --------------------------------------------------------

    pump_current = (
        0.72
        + 0.03 * (flow_rate - 1.8)
        + np.random.normal(0, 0.015, num_samples)
    )

    pump_current = np.clip(
        pump_current,
        0,
        None
    )

    # --------------------------------------------------------
    # 6. Vibration
    # --------------------------------------------------------

    vibration = (
        0.12
        + np.random.normal(0, 0.015, num_samples)
    )

    vibration = np.clip(
        vibration,
        0,
        None
    )

    # --------------------------------------------------------
    # Pump Status
    # --------------------------------------------------------

    pump_status = np.ones(
        num_samples,
        dtype=int
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    data = pd.DataFrame({

        "timestamp": timestamps,

        "water_level": water_level,

        "flow_rate": flow_rate,

        "pressure": pressure,

        "temperature": temperature,

        "pump_current": pump_current,

        "vibration": vibration,

        "pump_status": pump_status
    })

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    data["label"] = "Normal"

    return data


# ============================================================
# 2. PHYSICAL FAULT DATA GENERATION
# ============================================================

def generate_physical_fault_data(
    num_samples=5000,
    seed=42
):

    np.random.seed(seed)

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    timestamps = pd.date_range(
        start="2026-01-02 10:00:00",
        periods=num_samples,
        freq="s"
    )

    time = np.arange(num_samples)

    # --------------------------------------------------------
    # Healthy baseline
    # --------------------------------------------------------

    water_level = (
        70
        + 5 * np.sin(2 * np.pi * time / 1000)
        + np.random.normal(0, 0.3, num_samples)
    )

    flow_rate = (
        1.8
        + 0.05 * np.sin(2 * np.pi * time / 500)
        + np.random.normal(0, 0.02, num_samples)
    )

    pressure = (
        4.2
        + 0.25 * (flow_rate - 1.8)
        + np.random.normal(0, 0.03, num_samples)
    )

    temperature = (
        32
        + 1.5 * np.sin(2 * np.pi * time / 2000)
        + np.random.normal(0, 0.1, num_samples)
    )

    pump_current = (
        0.72
        + 0.03 * (flow_rate - 1.8)
        + np.random.normal(0, 0.015, num_samples)
    )

    vibration = (
        0.12
        + np.random.normal(0, 0.015, num_samples)
    )

    # --------------------------------------------------------
    # Physical fault progression
    # --------------------------------------------------------
    #
    # Simulate gradual pump degradation / flow restriction.
    #
    # Flow        ↓
    # Pressure    ↓
    # Temperature ↑
    # Current     ↑
    # Vibration   ↑
    #
    # --------------------------------------------------------

    fault_progress = np.linspace(
        0,
        1,
        num_samples
    )

    flow_rate -= (
        0.7 * fault_progress
    )

    pressure -= (
        0.8 * fault_progress
    )

    temperature += (
        8 * fault_progress
    )

    pump_current += (
        0.25 * fault_progress
    )

    vibration += (
        0.35 * fault_progress
    )

    # --------------------------------------------------------
    # Keep values realistic
    # --------------------------------------------------------

    flow_rate = np.clip(
        flow_rate,
        0.1,
        None
    )

    pressure = np.clip(
        pressure,
        0.5,
        None
    )

    temperature = np.clip(
        temperature,
        None,
        50
    )

    pump_current = np.clip(
        pump_current,
        0,
        None
    )

    vibration = np.clip(
        vibration,
        0,
        None
    )

    water_level = np.clip(
        water_level,
        0,
        100
    )

    # --------------------------------------------------------
    # Pump status
    # --------------------------------------------------------

    pump_status = np.ones(
        num_samples,
        dtype=int
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    data = pd.DataFrame({

        "timestamp": timestamps,

        "water_level": water_level,

        "flow_rate": flow_rate,

        "pressure": pressure,

        "temperature": temperature,

        "pump_current": pump_current,

        "vibration": vibration,

        "pump_status": pump_status
    })

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    data["label"] = "Physical_Fault"

    return data


# ============================================================
# 3. FALSE DATA INJECTION (FDI)
# ============================================================
#
# IMPORTANT:
#
# The physical pump is HEALTHY.
#
# The attacker modifies the values being reported
# by the sensors.
#
# This is different from a physical fault.
#
# ============================================================

def generate_fdi_data(
    num_samples=5000,
    seed=42
):

    np.random.seed(seed)

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    timestamps = pd.date_range(
        start="2026-01-03 10:00:00",
        periods=num_samples,
        freq="s"
    )

    time = np.arange(num_samples)

    # --------------------------------------------------------
    # Healthy physical system
    # --------------------------------------------------------

    water_level = (
        70
        + 5 * np.sin(2 * np.pi * time / 1000)
        + np.random.normal(0, 0.3, num_samples)
    )

    flow_rate = (
        1.8
        + 0.05 * np.sin(2 * np.pi * time / 500)
        + np.random.normal(0, 0.02, num_samples)
    )

    pressure = (
        4.2
        + 0.25 * (flow_rate - 1.8)
        + np.random.normal(0, 0.03, num_samples)
    )

    temperature = (
        32
        + 1.5 * np.sin(2 * np.pi * time / 2000)
        + np.random.normal(0, 0.1, num_samples)
    )

    pump_current = (
        0.72
        + 0.03 * (flow_rate - 1.8)
        + np.random.normal(0, 0.015, num_samples)
    )

    vibration = (
        0.12
        + np.random.normal(0, 0.015, num_samples)
    )

    # --------------------------------------------------------
    # Cyber attack begins
    # --------------------------------------------------------

    attack_start = 1000

    # The attacker injects false values.

    flow_rate[attack_start:] -= 0.8

    pressure[attack_start:] -= 1.0

    temperature[attack_start:] += 6.0

    vibration[attack_start:] += 0.4

    # --------------------------------------------------------
    # Keep values realistic
    # --------------------------------------------------------

    water_level = np.clip(
        water_level,
        0,
        100
    )

    flow_rate = np.clip(
        flow_rate,
        0.1,
        None
    )

    pressure = np.clip(
        pressure,
        0.5,
        None
    )

    temperature = np.clip(
        temperature,
        None,
        50
    )

    pump_current = np.clip(
        pump_current,
        0,
        None
    )

    vibration = np.clip(
        vibration,
        0,
        None
    )

    # --------------------------------------------------------
    # Physical pump remains healthy
    # --------------------------------------------------------

    pump_status = np.ones(
        num_samples,
        dtype=int
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    data = pd.DataFrame({

        "timestamp": timestamps,

        "water_level": water_level,

        "flow_rate": flow_rate,

        "pressure": pressure,

        "temperature": temperature,

        "pump_current": pump_current,

        "vibration": vibration,

        "pump_status": pump_status
    })

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    data["label"] = "Cyber_Induced_Anomaly"

    return data


# ============================================================
# 4. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CYBERSNIFF - INDUSTRIAL SENSOR SIMULATOR")
    print("=" * 60)

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_directory = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
        "data",
        "raw"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    print("\nOutput directory:")
    print(output_directory)

    # ========================================================
    # GENERATE NORMAL DATA
    # ========================================================

    print("\nGenerating normal sensor data...")

    normal_data = generate_normal_data(
        num_samples=10000
    )

    normal_output_file = os.path.join(
        output_directory,
        "normal_sensor_data.csv"
    )

    normal_data.to_csv(
        normal_output_file,
        index=False
    )

    print(
        f"Normal dataset saved to:\n"
        f"{normal_output_file}"
    )

    print(
        f"Normal samples: "
        f"{len(normal_data)}"
    )

    # ========================================================
    # GENERATE PHYSICAL FAULT DATA
    # ========================================================

    print("\nGenerating physical fault data...")

    fault_data = generate_physical_fault_data(
        num_samples=5000
    )

    fault_output_file = os.path.join(
        output_directory,
        "physical_fault_data.csv"
    )

    fault_data.to_csv(
        fault_output_file,
        index=False
    )

    print(
        f"Physical fault dataset saved to:\n"
        f"{fault_output_file}"
    )

    print(
        f"Physical fault samples: "
        f"{len(fault_data)}"
    )

    # ========================================================
    # GENERATE FDI DATA
    # ========================================================

    print("\nGenerating cyber-induced anomaly data...")

    fdi_data = generate_fdi_data(
        num_samples=5000
    )

    fdi_output_file = os.path.join(
        output_directory,
        "fdi_sensor_data.csv"
    )

    fdi_data.to_csv(
        fdi_output_file,
        index=False
    )

    print(
        f"FDI dataset saved to:\n"
        f"{fdi_output_file}"
    )

    print(
        f"FDI samples: "
        f"{len(fdi_data)}"
    )

    # ========================================================
    # DISPLAY NORMAL DATA
    # ========================================================

    print("\n" + "=" * 60)
    print("NORMAL DATA - FIRST 5 RECORDS")
    print("=" * 60)

    print(
        normal_data.head()
    )

    # ========================================================
    # DISPLAY PHYSICAL FAULT DATA
    # ========================================================

    print("\n" + "=" * 60)
    print("PHYSICAL FAULT DATA - FIRST 5 RECORDS")
    print("=" * 60)

    print(
        fault_data.head()
    )

    # ========================================================
    # DISPLAY FDI DATA
    # ========================================================

    print("\n" + "=" * 60)
    print("FDI DATA - FIRST 5 RECORDS")
    print("=" * 60)

    print(
        fdi_data.head()
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("\nNormal dataset:")
    print(
        normal_data.describe()
    )

    print("\nPhysical fault dataset:")
    print(
        fault_data.describe()
    )

    print("\nFDI dataset:")
    print(
        fdi_data.describe()
    )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)

    print("\nNormal:")
    print(
        normal_data["label"].value_counts()
    )

    print("\nPhysical Fault:")
    print(
        fault_data["label"].value_counts()
    )

    print("\nCyber-Induced Anomaly:")
    print(
        fdi_data["label"].value_counts()
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nGenerated files:")

    print("1. normal_sensor_data.csv")
    print("2. physical_fault_data.csv")
    print("3. fdi_sensor_data.csv")

    print("\nAll files are located inside:")
    print(output_directory)

    print("\nNext step:")
    print("Combine all three datasets into one ML dataset.")