import { useState, useEffect } from 'react';
import axios from 'axios';
import './Predictor.css';

interface FormData {
  gender: string;
  height: number;
  weight: number;
  bmi: number;
  physical_activity: string;
  smoking_status: string;
  alcohol_consumption: string;
  diet: string;
  blood_pressure: string;
  cholesterol: number;
  diabetes: number;
  hypertension: number;
  heart_disease: number;
  asthma: number;
}

interface PredictionResult {
  success: boolean;
  prediction: number;
  confidence: number;
  profile: {
    bmi: number;
    bmi_category: string;
    cholesterol: number;
    cholesterol_status: string;
    activity: string;
    smoking: string;
    risk_factors: number;
  };
  insights: Array<{ category: string; message: string; severity: string }>;
  recommendations: Array<{ title: string; description: string; priority: string }>;
  model_version: string;
  timestamp: string;
}

const Predictor = () => {
  const [formData, setFormData] = useState<FormData>({
    gender: '',
    height: 170,
    weight: 70,
    bmi: 24.2,
    physical_activity: '',
    smoking_status: '',
    alcohol_consumption: '',
    diet: '',
    blood_pressure: '',
    cholesterol: 200,
    diabetes: 0,
    hypertension: 0,
    heart_disease: 0,
    asthma: 0,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [showResults, setShowResults] = useState(false);

  // Calculate BMI whenever height or weight changes
  useEffect(() => {
    if (formData.height > 0 && formData.weight > 0) {
      const bmi = formData.weight / (formData.height / 100) ** 2;
      setFormData(prev => ({ ...prev, bmi: parseFloat(bmi.toFixed(1)) }));
    }
  }, [formData.height, formData.weight]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    
    if (type === 'checkbox') {
      const checked = (e.target as HTMLInputElement).checked;
      setFormData(prev => ({ ...prev, [name]: checked ? 1 : 0 }));
    } else if (type === 'number' || type === 'range') {
      setFormData(prev => ({ ...prev, [name]: parseFloat(value) }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const validateForm = (): boolean => {
    const requiredFields = [
      'gender', 'physical_activity', 'smoking_status',
      'alcohol_consumption', 'diet', 'blood_pressure'
    ];

    for (const field of requiredFields) {
      if (!formData[field as keyof FormData] || formData[field as keyof FormData] === '') {
        alert(`Please fill in the ${field.replace('_', ' ')} field`);
        return false;
      }
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post<PredictionResult>(
        '/api/v1/predictions/predict',
        formData
      );

      if (response.data.success) {
        setResult(response.data);
        setShowResults(true);
        
        // Scroll to results
        setTimeout(() => {
          const resultsElement = document.getElementById('results');
          if (resultsElement) {
            resultsElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 100);
      } else {
        alert('An error occurred during prediction');
      }
    } catch (error) {
      alert('Error making prediction: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!result) return;

    const payload = {
      formData,
      prediction: result.prediction,
      insights: result.insights.map(i => i.message),
      recommendations: result.recommendations.map(r => `${r.title}: ${r.description}`)
    };

    try {
      setLoading(true);
      const response = await axios.post(
        '/api/v1/report',
        payload,
        { responseType: 'blob' }
      );

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `life-expectancy-report-${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      alert('Error generating PDF: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    window.location.href = '#predictor';
    setShowResults(false);
    setResult(null);
  };

  return (
    <section id="predictor" className="predictor-section">
      <div className="predictor-container">
        <div className="form-header">
          <h2>Health Profile Assessment</h2>
          <p>Fill in your health information for an accurate prediction</p>
        </div>

        <form onSubmit={handleSubmit} className="prediction-form">
          {/* Personal Information */}
          <div className="form-section">
            <h3><i className="fas fa-user"></i> Personal Information</h3>
            
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="gender">Gender</label>
                <select
                  id="gender"
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="height">Height (cm)</label>
                <div className="input-group">
                  <input
                    type="range"
                    id="heightRange"
                    min="140"
                    max="220"
                    value={formData.height}
                    onChange={(e) => setFormData(prev => ({ ...prev, height: parseFloat(e.target.value) }))}
                  />
                  <input
                    type="number"
                    id="height"
                    name="height"
                    min="140"
                    max="220"
                    value={formData.height}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="weight">Weight (kg)</label>
                <div className="input-group">
                  <input
                    type="range"
                    id="weightRange"
                    min="40"
                    max="150"
                    value={formData.weight}
                    onChange={(e) => setFormData(prev => ({ ...prev, weight: parseFloat(e.target.value) }))}
                  />
                  <input
                    type="number"
                    id="weight"
                    name="weight"
                    min="40"
                    max="150"
                    value={formData.weight}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="bmi">BMI</label>
                <input
                  type="number"
                  id="bmi"
                  name="bmi"
                  step="0.1"
                  value={formData.bmi}
                  readOnly
                />
              </div>
            </div>
          </div>

          {/* Lifestyle Factors */}
          <div className="form-section">
            <h3><i className="fas fa-heart"></i> Lifestyle Factors</h3>
            
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="physical_activity">Physical Activity</label>
                <select
                  id="physical_activity"
                  name="physical_activity"
                  value={formData.physical_activity}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select Activity Level</option>
                  <option value="Low">Low</option>
                  <option value="Medium">Medium</option>
                  <option value="High">High</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="smoking_status">Smoking Status</label>
                <select
                  id="smoking_status"
                  name="smoking_status"
                  value={formData.smoking_status}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select Status</option>
                  <option value="Never">Never</option>
                  <option value="Former">Former</option>
                  <option value="Current">Current</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="alcohol_consumption">Alcohol Consumption</label>
                <select
                  id="alcohol_consumption"
                  name="alcohol_consumption"
                  value={formData.alcohol_consumption}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select Level</option>
                  <option value="None">None</option>
                  <option value="Moderate">Moderate</option>
                  <option value="High">High</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="diet">Diet Type</label>
                <select
                  id="diet"
                  name="diet"
                  value={formData.diet}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select Diet</option>
                  <option value="Poor">Poor</option>
                  <option value="Average">Average</option>
                  <option value="Healthy">Healthy</option>
                </select>
              </div>
            </div>
          </div>

          {/* Health Indicators */}
          <div className="form-section">
            <h3><i className="fas fa-stethoscope"></i> Health Indicators</h3>
            
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="blood_pressure">Blood Pressure</label>
                <select
                  id="blood_pressure"
                  name="blood_pressure"
                  value={formData.blood_pressure}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select Level</option>
                  <option value="Low">Low</option>
                  <option value="Normal">Normal</option>
                  <option value="High">High</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="cholesterol">Cholesterol (mg/dL)</label>
                <div className="input-group">
                  <input
                    type="range"
                    id="cholesterolRange"
                    min="100"
                    max="300"
                    value={formData.cholesterol}
                    onChange={(e) => setFormData(prev => ({ ...prev, cholesterol: parseFloat(e.target.value) }))}
                  />
                  <input
                    type="number"
                    id="cholesterol"
                    name="cholesterol"
                    min="100"
                    max="300"
                    value={formData.cholesterol}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Medical History */}
          <div className="form-section">
            <h3><i className="fas fa-hospital"></i> Medical History</h3>
            
            <div className="checkbox-group">
              <div className="checkbox-item">
                <input
                  type="checkbox"
                  id="diabetes"
                  name="diabetes"
                  checked={formData.diabetes === 1}
                  onChange={handleChange}
                />
                <label htmlFor="diabetes">Diabetes</label>
              </div>
              <div className="checkbox-item">
                <input
                  type="checkbox"
                  id="hypertension"
                  name="hypertension"
                  checked={formData.hypertension === 1}
                  onChange={handleChange}
                />
                <label htmlFor="hypertension">Hypertension</label>
              </div>
              <div className="checkbox-item">
                <input
                  type="checkbox"
                  id="heart_disease"
                  name="heart_disease"
                  checked={formData.heart_disease === 1}
                  onChange={handleChange}
                />
                <label htmlFor="heart_disease">Heart Disease</label>
              </div>
              <div className="checkbox-item">
                <input
                  type="checkbox"
                  id="asthma"
                  name="asthma"
                  checked={formData.asthma === 1}
                  onChange={handleChange}
                />
                <label htmlFor="asthma">Asthma</label>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <button type="submit" className={`submit-button ${loading ? 'loading' : ''}`} disabled={loading}>
            <span className="button-text">Get Prediction</span>
            <span className="button-loader"></span>
          </button>
        </form>
      </div>

      {/* Results Section */}
      {showResults && result && (
        <div id="results" className="results-container">
          <div className="results-content">
            {/* Prediction Result */}
            <div className="result-card main-result">
              <div className="result-icon">
                <i className="fas fa-chart-pie"></i>
              </div>
              <div className="result-text">
                <p className="result-label">Predicted Life Expectancy</p>
                <p className="result-value">{result.prediction.toFixed(1)}</p>
                <p className="result-unit">years</p>
              </div>
            </div>

            {/* Profile Summary */}
            <div className="result-card">
              <h3><i className="fas fa-user-check"></i> Your Profile</h3>
              <div className="profile-grid">
                <div className="profile-item">
                  <span className="profile-label">BMI</span>
                  <span className="profile-value">{result.profile.bmi} ({result.profile.bmi_category})</span>
                </div>
                <div className="profile-item">
                  <span className="profile-label">Cholesterol</span>
                  <span className="profile-value">{result.profile.cholesterol} ({result.profile.cholesterol_status})</span>
                </div>
                <div className="profile-item">
                  <span className="profile-label">Activity</span>
                  <span className="profile-value">{result.profile.activity}</span>
                </div>
                <div className="profile-item">
                  <span className="profile-label">Smoking</span>
                  <span className="profile-value">{result.profile.smoking}</span>
                </div>
              </div>
            </div>

            {/* Health Insights */}
            <div className="result-card">
              <h3><i className="fas fa-lightbulb"></i> Health Insights</h3>
              <div className="insights-list">
                {result.insights.map((insight, index) => (
                  <div
                    key={index}
                    className={`insight-item ${insight.severity}`}
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    <strong>{insight.category}:</strong> {insight.message}
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            <div className="result-card">
              <h3><i className="fas fa-bullseye"></i> Recommendations</h3>
              <div className="recommendations-list">
                {result.recommendations.map((recommendation, index) => (
                  <div
                    key={index}
                    className={`recommendation-item ${recommendation.priority}`}
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    <strong>{recommendation.title}:</strong> {recommendation.description}
                  </div>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="result-actions">
              <button className="action-button primary" onClick={resetForm}>
                <i className="fas fa-redo"></i> New Prediction
              </button>
              <button className="action-button secondary" onClick={downloadReport}>
                <i className="fas fa-download"></i> Download Report
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};

export default Predictor;
