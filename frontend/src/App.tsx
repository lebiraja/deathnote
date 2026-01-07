import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Features from './components/Features';
import Predictor from './components/Predictor';
import About from './components/About';
import Footer from './components/Footer';
import './App.css';

function App() {
  return (
    <div className="app">
      <Navbar />
      <Hero />
      <Features />
      <Predictor />
      <About />
      <Footer />
    </div>
  );
}

export default App;
