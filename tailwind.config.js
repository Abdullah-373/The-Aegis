/** Build the production CSS for The Aegis dashboard.
 *
 * The previous build shipped tailwindcss from a CDN, which meant the
 * full JIT compiler ran in the browser on every page load. This config
 * pre-compiles the stylesheet at build time, scanning templates/index.html
 * for the exact classes in use so the resulting CSS is small.
 */
module.exports = {
  content: ['./templates/**/*.html'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        ink: {
          900: '#0b1220', 800: '#1a2236', 700: '#2a3349',
          600: '#475569', 500: '#64748b', 400: '#94a3b8',
          300: '#cbd5e1', 200: '#e2e8f0', 100: '#f1f5f9', 50: '#f8fafc',
        },
      },
    },
  },
  plugins: [],
  // Safelist classes the template builds at runtime (verdict colours, etc.).
  // Tailwind's static scanner cannot see them so we list them here.
  safelist: [
    'hover:border-emerald-200', 'hover:border-emerald-300',
    'hover:border-amber-200',   'hover:border-amber-300',
    'hover:border-rose-200',    'hover:border-rose-300',
    'bg-emerald-50', 'bg-amber-50', 'bg-rose-50',
    'text-emerald-700', 'text-amber-700', 'text-rose-700',
    'border-emerald-200', 'border-amber-200', 'border-rose-200',
    'ring-emerald-300', 'ring-amber-300', 'ring-rose-300',
  ],
};
