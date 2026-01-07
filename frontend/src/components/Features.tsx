import './Features.css';

const Features = () => {
  return (
    <section id="features" className="features">
      <h2>Why Choose Our Predictor?</h2>
      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">
            <i className="fas fa-brain"></i>
          </div>
          <h3>AI Powered</h3>
          <p>Advanced machine learning models trained on 10,000+ health records</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">
            <i className="fas fa-shield-heart"></i>
          </div>
          <h3>Accurate</h3>
          <p>87% accuracy rate with real-time health insights</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">
            <i className="fas fa-chart-line"></i>
          </div>
          <h3>Personalized</h3>
          <p>Customized recommendations based on your health profile</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">
            <i className="fas fa-lock"></i>
          </div>
          <h3>Private</h3>
          <p>Your data is processed locally and never stored</p>
        </div>
      </div>
    </section>
  );
};

export default Features;
