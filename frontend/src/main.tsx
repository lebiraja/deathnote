import '@fontsource/caveat/500.css'
import '@fontsource/cormorant-garamond/500.css'
import '@fontsource/cormorant-garamond/700.css'
import '@fontsource/inter/400.css'
import '@fontsource/inter/600.css'
import { MotionConfig } from 'motion/react'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './index.css'

const root = document.getElementById('root')
if (!root) throw new Error('Missing #root element')

createRoot(root).render(
  <StrictMode>
    <MotionConfig reducedMotion="user">
      <App />
    </MotionConfig>
  </StrictMode>,
)
