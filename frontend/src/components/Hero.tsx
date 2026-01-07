import './Hero.css';

const Hero = () => {
  const scrollToPredictor = () => {
    const element = document.getElementById('predictor');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="hero">
      <div className="hero-content">
        <h1 className="hero-title">
          <span className="gradient-text">Predict Your Life Expectancy</span>
        </h1>
        <p className="hero-subtitle">
          Advanced AI-powered analysis of health and lifestyle factors
        </p>
        <a href="#predictor" className="cta-button" onClick={(e) => { e.preventDefault(); scrollToPredictor(); }}>
          Start Prediction <i className="fas fa-arrow-right"></i>
        </a>
      </div>
      <div className="hero-animation">
        <div className="animated-sphere"></div>
      </div>
    </section>
  );
};

export default Hero;
