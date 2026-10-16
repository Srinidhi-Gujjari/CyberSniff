
import { useState } from 'react'
import { featureGroups, featureNames } from './features'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const [activeTab, setActiveTab] = useState('Overview')
  const [values, setValues] = useState({})
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const [apiStatus, setApiStatus] = useState('Not checked')

  async function checkApi() {
    setApiStatus('Checking...')
    try {
      const response = await fetch(`${API_URL}/health`)
      if (!response.ok) throw new Error('API health check failed')
      const data = await response.json()
      setApiStatus(
        data.model_loaded ? 'Connected' : 'API online, model unavailable'
      )
    } catch {
      setApiStatus('Offline')
    }
  }

  function handleChange(name, value) {
    setValues((previous) => ({ ...previous, [name]: value }))
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setResult(null)

    const missing = featureNames.filter(
      (name) =>
        values[name] === undefined ||
        values[name].trim() === '' ||
        !Number.isFinite(Number(values[name]))
    )

    if (missing.length > 0) {
      setError(
        `Please enter valid numeric values for all 49 features. ${missing.length} field(s) still need attention.`
      )
      return
    }

    const features = Object.fromEntries(
      featureNames.map((name) => [name, Number(values[name])])
    )

    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
      })

      const data = await response.json()

      if (!response.ok) {
        const detail =
          typeof data.detail === 'string'
            ? data.detail
            : JSON.stringify(data.detail ?? data)
        throw new Error(detail || 'Prediction request failed')
      }

      setResult(data)
      setActiveTab('Predictions')
    } catch (err) {
      setError(
        `${err.message}. Make sure the CyberSniff backend is running at ${API_URL}.`
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">C</div>
          <div>
            <h2>CyberSniff</h2>
            <span>Threat Detection System</span>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>

        {['Overview', 'Sensor Data', 'Predictions'].map((tab) => (
          <button
            key={tab}
            className={`nav-item ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            <span>
              {tab === 'Overview'
                ? '◫'
                : tab === 'Sensor Data'
                  ? '⌁'
                  : '⌘'}
            </span>
            {tab}
          </button>
        ))}

        <div className="sidebar-footer">
          <div className="status-dot" />
          <span>Detection system</span>
          <strong>{apiStatus}</strong>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">CYBERSECURITY / MONITORING</p>
            <h1>{activeTab}</h1>
          </div>
          <div className="profile">
            <div className="avatar">CS</div>
            <div>
              <strong>Analyst</strong>
              <span>Security dashboard</span>
            </div>
          </div>
        </header>

        <section className="welcome">
          <div>
            <p className="eyebrow">CYBERSNIFF INTELLIGENCE</p>
            <h2>Industrial System Security</h2>
            <p>
              Monitor sensor readings and identify potential
              cyber-induced anomalies and physical faults.
            </p>
          </div>
          <div className="shield">⬡</div>
        </section>

        <section className="metrics">
          <article className="metric-card">
            <span className="metric-icon blue">⌁</span>
            <p>Sensor Features</p>
            <h2>{featureNames.length}</h2>
            <span className="muted">Required model inputs</span>
          </article>

          <article className="metric-card">
            <span className="metric-icon purple">◈</span>
            <p>Detection Classes</p>
            <h2>3</h2>
            <span className="muted">Model-defined categories</span>
          </article>

          <article className="metric-card">
            <span className="metric-icon green">✓</span>
            <p>API Status</p>
            <h2 className="api-status-value">{apiStatus}</h2>
            <button className="text-button" onClick={checkApi}>
              Check connection
            </button>
          </article>
        </section>

        {activeTab === 'Overview' && (
          <section className="content-card">
            <div className="section-heading">
              <div>
                <h2>System Overview</h2>
                <p>Your security monitoring workspace</p>
              </div>
              <span className="demo-badge">MODEL-BASED DETECTION</span>
            </div>

            <div className="overview-grid">
              <div className="info-panel">
                <span className="panel-symbol">◉</span>
                <h3>Sensor Monitoring</h3>
                <p>
                  Enter all 49 sensor measurements and submit them
                  for classification by the trained model.
                </p>
                <button
                  className="primary-button"
                  onClick={() => setActiveTab('Sensor Data')}
                >
                  Enter sensor data
                </button>
              </div>
              <div className="info-panel">
                <span className="panel-symbol purple-text">⌁</span>
                <h3>Prediction Results</h3>
                <p>
                  Review the latest result returned by the
                  CyberSniff prediction API.
                </p>
                <button
                  className="secondary-button"
                  onClick={() => setActiveTab('Predictions')}
                >
                  View predictions
                </button>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'Sensor Data' && (
          <section className="content-card">
            <div className="section-heading">
              <div>
                <h2>Sensor Input Form</h2>
                <p>
                  Enter numeric readings for every required feature.
                  All {featureNames.length} fields are required.
                </p>
              </div>
              <span className="demo-badge">
                {Object.values(values).filter((v) => v !== '').length}/49 ENTERED
              </span>
            </div>

            <form onSubmit={handleSubmit}>
              {featureGroups.map((group) => (
                <fieldset className="sensor-group" key={group.title}>
                  <legend>{group.title}</legend>
                  <div className="sensor-grid">
                    {group.features.map((name) => (
                      <label className="sensor-field" key={name}>
                        <span>{name.replaceAll('_', ' ')}</span>
                        <input
                          type="number"
                          step="any"
                          required
                          value={values[name] ?? ''}
                          onChange={(event) =>
                            handleChange(name, event.target.value)
                          }
                          placeholder="Enter numeric value"
                        />
                      </label>
                    ))}
                  </div>
                </fieldset>
              ))}

              {error && (
                <div className="error-message" role="alert">
                  {error}
                </div>
              )}

              <div className="form-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setValues({})
                    setError('')
                  }}
                >
                  Clear form
                </button>
                <button
                  type="submit"
                  className="primary-button"
                  disabled={loading}
                >
                  {loading ? 'Predicting...' : 'Run prediction'}
                </button>
              </div>
            </form>
          </section>
        )}

        {activeTab === 'Predictions' && (
          <section className="content-card">
            <div className="section-heading">
              <div>
                <h2>Prediction Results</h2>
                <p>Results returned by the backend model</p>
              </div>
              <span className="demo-badge">
                {result ? 'RESULT AVAILABLE' : 'AWAITING PREDICTION'}
              </span>
            </div>

            {result ? (
              <div className="result-panel">
                <p className="eyebrow">PREDICTED CLASS</p>
                <h2 className="result-class">{result.prediction}</h2>
                <p>
                  Model: {result.model_name}
                </p>

                {result.available_classes && (
                  <div className="class-list">
                    <h3>Available classes</h3>
                    {result.available_classes.map((label) => (
                      <span className="class-chip" key={label}>
                        {label}
                      </span>
                    ))}
                  </div>
                )}

                {result.probabilities && (
                  <div className="probability-list">
                    <h3>Class probabilities</h3>
                    {Object.entries(result.probabilities).map(
                      ([label, probability]) => (
                        <div className="probability-row" key={label}>
                          <span>{label}</span>
                          <strong>
                            {(probability * 100).toFixed(1)}%
                          </strong>
                        </div>
                      )
                    )}
                  </div>
                )}

                <p className="muted">
                  This result is a model classification, not a
                  guarantee of real-world system safety.
                </p>
              </div>
            ) : (
              <div className="empty-state">
                <span>◈</span>
                <h3>No prediction yet</h3>
                <p>
                  Enter all 49 sensor values, then run a prediction.
                </p>
                <button
                  className="primary-button"
                  onClick={() => setActiveTab('Sensor Data')}
                >
                  Open sensor form
                </button>
              </div>
            )}
          </section>
        )}

        <footer>
          CyberSniff · Industrial anomaly detection
          <span>Frontend development</span>
        </footer>
      </main>
    </div>
  )
}

export default App
