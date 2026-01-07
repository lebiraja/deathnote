import './About.css';

const About = () => {
  return (
    <section id="about" className="about">
      <h2>About This Predictor</h2>
      <div className="about-content">
        <div className="about-text">
          <h3>Advanced AI Technology</h3>
          <p>
            Our life expectancy predictor uses state-of-the-art machine learning algorithms 
            trained on comprehensive health data. The model analyzes multiple factors including 
            physical health, lifestyle choices, and medical history to provide accurate predictions.
          </p>
          <div className="stats">
            <div className="stat">
              <strong>87%</strong>
              <span>Accuracy Rate</span>
            </div>
            <div className="stat">
              <strong>10,000+</strong>
              <span>Training Records</span>
            </div>
            <div className="stat">
              <strong>14</strong>
              <span>Health Factors</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
