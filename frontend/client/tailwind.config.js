/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
        },
        blue: {
          600: '#2563eb',
          700: '#1d4ed8',
        },
        green: {
          600: '#16a34a',
        },
        red: {
          600: '#dc2626',
        },
        yellow: {
          600: '#ca8a04',
        },
        purple: {
          600: '#9333ea',
        },
      },
      animation: {
        'spin': 'spin 1s linear infinite',
      },
    },
  },
  plugins: [],
} 