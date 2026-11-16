/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#1c1915",
        paper: "#f3eee4",
        card: "#faf7f1",
        copper: "#8d4528",
        line: "#e3d8c8",
        moss: "#3f5c4b",
      },
      fontFamily: {
        serif: ["Fraunces", "ui-serif", "Georgia", "serif"],
        sans: ["Outfit", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        sheet: "0 18px 50px rgba(48, 32, 16, 0.08)",
      },
    },
  },
  plugins: [],
};
