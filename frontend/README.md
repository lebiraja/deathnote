# Life Expectancy Predictor - Frontend

Modern React + TypeScript + Vite frontend for the Life Expectancy Prediction application.

## 🚀 Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool & dev server
- **Axios** - HTTP client
- **Font Awesome** - Icons
- **CSS3** - Styling with animations

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Navbar.tsx/css       # Navigation bar
│   │   ├── Hero.tsx/css         # Hero section with CTA
│   │   ├── Features.tsx/css     # Features grid
│   │   ├── Predictor.tsx/css    # Main prediction form & results
│   │   ├── About.tsx/css        # About section
│   │   └── Footer.tsx/css       # Footer
│   ├── App.tsx                  # Main app component
│   ├── App.css                  # Global styles & animations
│   ├── main.tsx                 # Entry point
│   └── index.css                # CSS reset
├── index.html                   # HTML template
├── package.json                 # Dependencies
├── tsconfig.json                # TypeScript config
└── vite.config.ts               # Vite config
```

## 🛠️ Installation

```bash
cd frontend
npm install
```

## 🏃 Development

```bash
# Start development server (http://localhost:5173)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🔌 API Integration

The frontend connects to the Flask backend API at `http://localhost:5000`.

### Endpoints Used:
- `POST /api/predict` - Get life expectancy prediction
- `POST /api/report` - Download PDF report

### Environment Variables (optional):
Create a `.env` file:
```
VITE_API_URL=http://localhost:5000
```

## 🎨 Design Features

### Exact Replica of Original Design
- ✅ Same color scheme (purple gradient theme)
- ✅ Same layout and sections
- ✅ Same animations (fadeIn, slideUp, float, etc.)
- ✅ Same form structure with all 14 health factors
- ✅ Same results display with insights & recommendations
- ✅ Responsive design (mobile-friendly)

### Components
1. **Navbar** - Fixed navigation with smooth scroll
2. **Hero** - Title, subtitle, CTA button, animated sphere
3. **Features** - 4-column grid (AI, Accurate, Personalized, Private)
4. **Predictor** - Multi-section form with validation
5. **Results** - Prediction display, profile, insights, recommendations
6. **About** - Technology info with statistics
7. **Footer** - Copyright and disclaimer

## 🔧 TypeScript Features

- Full type safety for form data
- Interface definitions for API responses
- Proper event typing
- Type-safe state management

## 📱 Responsive Design

- Desktop: Full layout with all features
- Tablet (< 768px): Single column grid, adjusted spacing
- Mobile (< 480px): Simplified navigation, stacked layout

## ⚡ Performance

- Code splitting with React lazy loading
- Optimized CSS with minimal specificity
- Efficient re-renders with React hooks
- Fast development with Vite HMR

## 🚢 Deployment

### Build for production:
```bash
npm run build
```

Outputs to `dist/` directory.

### Serve with:
- **Vercel**: `vercel deploy`
- **Netlify**: `netlify deploy --prod`
- **Static server**: `npx serve dist`

## 🔗 Backend Integration

Ensure Flask backend is running:
```bash
cd ..
python flask_app.py
```

Backend should be available at `http://localhost:5000`.

## 🎯 Features Implemented

- ✅ Form validation
- ✅ BMI auto-calculation
- ✅ Range/number input synchronization
- ✅ Checkbox handling for medical conditions
- ✅ Loading states
- ✅ Error handling
- ✅ Smooth scrolling
- ✅ Results display with animations
- ✅ PDF report download
- ✅ Fully responsive

## 📝 Notes

- Font Awesome icons loaded from CDN
- No dark mode (as per original design)
- CSS animations match original exactly
- All form fields validated before submission

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit PR

## 📄 License

Same as parent project
