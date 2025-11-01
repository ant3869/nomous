import React from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

declare global {
  interface Window {
    ReactReduxContext?: React.Context<unknown>
  }
}

if (typeof window !== 'undefined' && !window.ReactReduxContext) {
  window.ReactReduxContext = React.createContext(undefined)
}

createRoot(document.getElementById('root')!).render(<App />)
