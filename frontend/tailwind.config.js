/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          main: '#080B14',
          secondary: '#0D1220',
          primary: '#101727',
          elevated: '#151E31',
        },
        accent: {
          primary: '#00E5CC',
          secondary: '#3987FF',
          tertiary: '#9B6DFF',
        },
        text: {
          primary: '#F1F5FF',
          secondary: '#91A0B8',
        },
        border: {
          DEFAULT: '#243149',
        },
        success: '#39D98A',
        warning: '#FFB84D',
        error: '#FF5263',
      },
      fontFamily: {
        sans: ['Inter', 'Vazirmatn', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        slideUp: { '0%': { opacity: 0, transform: 'translateY(20px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
        pulseGlow: { '0%,100%': { opacity: 0.6 }, '50%': { opacity: 1 } },
      },
    },
  },
  plugins: [],
}
