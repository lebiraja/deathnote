import './Navbar.css';

const Navbar = () => {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="logo">
          <i className="fas fa-heart-pulse"></i>
          <span>Life Expectancy AI</span>
        </div>
        <ul className="nav-links">
          <li>
            <a href="#features" onClick={(e) => { e.preventDefault(); scrollToSection('features'); }}>
              Features
            </a>
          </li>
          <li>
            <a href="#predictor" onClick={(e) => { e.preventDefault(); scrollToSection('predictor'); }}>
              Predictor
            </a>
          </li>
          <li>
            <a href="#about" onClick={(e) => { e.preventDefault(); scrollToSection('about'); }}>
              About
            </a>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
