
import React from "react";
import "./Explainability.css";

const featureInfo = {
  force: ["Force", "Measures the force applied to the equipment."],
  phase_current_1: ["Phase Current 1", "Measures electrical current in motor phase 1."],
  phase_current_2: ["Phase Current 2", "Measures electrical current in motor phase 2."],
  speed: ["Speed", "Measures the motor's operating speed."],
  temp_2_bearing_module: ["Bearing Temperature", "Measures temperature around the bearing."],
  torque: ["Torque", "Measures the turning force produced by the motor."],
  vibration_1: ["Vibration", "Measures vibration in the equipment."],
};

const metrics = {
  mean: "average reading",
  max: "highest reading",
  min: "lowest reading",
  rms: "overall signal level",
  std: "variation in readings",
  kurtosis: "sharp peaks in the signal",
  peak_to_peak: "difference between highest and lowest readings",
};

function describeFeature(name) {
  const metric = Object.keys(metrics).find((m) =>
    name.endsWith(`_${m}`)
  );

  const group = metric
    ? name.slice(0, -(metric.length + 1))
    : name;

  const info = featureInfo[group] || [
    group.replaceAll("_", " "),
    "Measures an equipment operating condition.",
  ];

  return {
    name: info[0],
    meaning: info[1],
    metric: metrics[metric] || "measurement",
  };
}

function getRecommendations(prediction, features) {
  if (prediction === "Normal") {
    return [
      "Continue normal equipment monitoring.",
      "Perform regular preventive maintenance and sensor calibration.",
      "Monitor future readings for unusual changes.",
    ];
  }

  if (prediction === "Physical_Fault") {
    const groups = [...new Set(features.map((f) =>
      describeFeature(f.name).name
    ))];

    const actions = [];

    if (groups.includes("Torque")) {
      actions.push("Inspect motor loading, shaft coupling and gearbox condition.");
    }

    if (groups.includes("Vibration")) {
      actions.push("Check bearings, shaft alignment and mechanical looseness.");
    }

    if (groups.includes("Bearing Temperature")) {
      actions.push("Inspect bearing lubrication, friction and cooling.");
    }

    if (groups.includes("Phase Current 1") ||
        groups.includes("Phase Current 2")) {
      actions.push("Check motor current, electrical connections and possible overload.");
    }

    if (groups.includes("Speed")) {
      actions.push("Verify motor speed and check for unexpected variations.");
    }

    if (groups.includes("Force")) {
      actions.push("Inspect applied load and force sensor calibration.");
    }

    actions.push(
      "Compare the readings with historical healthy operating data."
    );

    return actions;
  }

  if (prediction === "Cyber_Induced_Anomaly") {
    return [
      "Verify sensor readings using trusted or redundant sources.",
      "Inspect communication logs for unusual messages or timing changes.",
      "Check device authentication and unauthorized access.",
      "Compare sensor data with trusted PLC or SCADA records.",
      "Preserve logs and investigate suspicious activity.",
    ];
  }

  return ["Verify sensor readings and inspect the equipment."];
}

function ShapExplanation({ explanation }) {
  if (!explanation?.available) {
    return <p>SHAP explanation is currently unavailable.</p>;
  }

  const features = [...(explanation.features || [])].sort(
    (a, b) =>
      Math.abs(Number(b.shap_value)) -
      Math.abs(Number(a.shap_value))
  ).slice(0, 3);

  const recommendations = getRecommendations(
    explanation.class_name,
    features
  );

  return (
    <div className="xai-explanation">
      <div className="xai-section">
        <h3>Top factors influencing the result</h3>
        <p className="xai-subtitle">
          These sensor measurements had the strongest influence
          on the model's prediction.
        </p>

        {features.map((feature, index) => {
          const details = describeFeature(feature.name);

          return (
            <div className="xai-factor" key={feature.name}>
              <h4>
                {index + 1}. {details.name} – {details.metric}
              </h4>

              <p>{details.meaning}</p>

              <span className="xai-reading">
                Sensor reading:{" "}
                <strong>{Number(feature.value).toFixed(3)}</strong>
              </span>

              <span className={
                Number(feature.shap_value) >= 0
                  ? "xai-positive"
                  : "xai-negative"
              }>
                {Number(feature.shap_value) > 0
                  ? "Supports prediction"
                  : Number(feature.shap_value) < 0
                    ? "Opposes prediction"
                    : "Minimal influence"}
              </span>
            </div>
          );
        })}
      </div>

      <div className="xai-recommendations">
        <h3>Recommended action</h3>

        <ul>
          {recommendations.map((action, index) => (
            <li key={index}>{action}</li>
          ))}
        </ul>
      </div>

      <p className="xai-disclaimer">
        Recommendations are suggested inspection steps, not
        confirmed fault diagnoses. SHAP explains model
        contributions, not fault severity or probability.
      </p>
    </div>
  );
}

export default ShapExplanation;
