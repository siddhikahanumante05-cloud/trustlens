import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react(), tailwindcss()],
    server: {
        host: true, // Allows access from any device on the network
        port: 5173, // You can change this to your preferred port
        open: true, // Automatically opens the app in the browser
    },
})
