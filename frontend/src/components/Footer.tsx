import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <p>&copy; 2025 Life Expectancy Predictor. All rights reserved.</p>
        <p className="disclaimer">
          <i className="fas fa-info-circle"></i>
          This tool is for informational purposes only. Always consult with healthcare professionals for medical advice.
        </p>
      </div>
    </footer>
  );
};

export default Footer;
