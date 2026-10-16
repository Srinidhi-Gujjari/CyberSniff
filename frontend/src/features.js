
export const featureGroups = [
  {
    title: 'Force',
    features: [
      'force_kurtosis',
      'force_max',
      'force_mean',
      'force_min',
      'force_peak_to_peak',
      'force_rms',
      'force_std',
    ],
  },
  {
    title: 'Phase Current 1',
    features: [
      'phase_current_1_kurtosis',
      'phase_current_1_max',
      'phase_current_1_mean',
      'phase_current_1_min',
      'phase_current_1_peak_to_peak',
      'phase_current_1_rms',
      'phase_current_1_std',
    ],
  },
  {
    title: 'Phase Current 2',
    features: [
      'phase_current_2_kurtosis',
      'phase_current_2_max',
      'phase_current_2_mean',
      'phase_current_2_min',
      'phase_current_2_peak_to_peak',
      'phase_current_2_rms',
      'phase_current_2_std',
    ],
  },
  {
    title: 'Speed',
    features: [
      'speed_kurtosis',
      'speed_max',
      'speed_mean',
      'speed_min',
      'speed_peak_to_peak',
      'speed_rms',
      'speed_std',
    ],
  },
  {
    title: 'Temperature — Bearing Module 2',
    features: [
      'temp_2_bearing_module_kurtosis',
      'temp_2_bearing_module_max',
      'temp_2_bearing_module_mean',
      'temp_2_bearing_module_min',
      'temp_2_bearing_module_peak_to_peak',
      'temp_2_bearing_module_rms',
      'temp_2_bearing_module_std',
    ],
  },
  {
    title: 'Torque',
    features: [
      'torque_kurtosis',
      'torque_max',
      'torque_mean',
      'torque_min',
      'torque_peak_to_peak',
      'torque_rms',
      'torque_std',
    ],
  },
  {
    title: 'Vibration 1',
    features: [
      'vibration_1_kurtosis',
      'vibration_1_max',
      'vibration_1_mean',
      'vibration_1_min',
      'vibration_1_peak_to_peak',
      'vibration_1_rms',
      'vibration_1_std',
    ],
  },
]

export const featureNames = featureGroups.flatMap(
  (group) => group.features
)
