# GoPredict Frontend Setup Guide

## Quick Start

### 1. Prerequisites

- Node.js 18+ installed
- Google Maps API key

### 2. Setup Frontend

```bash
# Run the setup script
setup-frontend.bat

# Or manually:
cd frontend
npm install
copy env.example .env
```

### 3. Configure Environment

Edit `frontend/.env` and add your Google Maps API key:

```env
VITE_GOOGLE_MAPS_API_KEY=your_actual_api_key_here
VITE_API_URL=http://localhost:8000
```

### 4. Start Development Server

```bash
cd frontend
npm run dev
```

The app will be available at `http://localhost:3000`

## Google Maps API Setup

### 1. Get API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable these APIs:
   - Maps JavaScript API
   - Directions API
   - Places API

### 2. Create API Key

1. Go to "Credentials" → "Create Credentials" → "API Key"
2. Copy the API key
3. Add it to your `.env` file

### 3. Secure Your API Key

1. Click on your API key in Google Cloud Console
2. Under "Application restrictions":
   - Select "HTTP referrers"
   - Add: `localhost:3000/*`, `127.0.0.1:3000/*`
3. Under "API restrictions":
   - Select "Restrict key"
   - Choose: Maps JavaScript API, Directions API, Places API

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── LocationSearch.tsx    # Smart location search
│   │   ├── GoogleMap.tsx         # Google Maps integration
│   │   └── ui/                   # Reusable UI components
│   ├── pages/
│   │   ├── Home.tsx              # Main application page
│   │   └── NotFound.tsx          # 404 page
│   ├── lib/
│   │   ├── api.ts                # Backend API integration
│   │   └── utils.ts              # Utility functions
│   └── App.tsx                   # Main app component
├── package.json                  # Dependencies
├── vite.config.ts               # Vite configuration
├── tailwind.config.js           # Tailwind CSS config
└── env.example                  # Environment template
```

## Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run typecheck    # TypeScript type checking
```

## Features

- 🗺️ **Google Maps Integration** - Interactive maps with route visualization
- 🔍 **Smart Location Search** - Autocomplete for NY and SF locations
- 🌙 **Dark/Light Mode** - System preference detection
- 📱 **Responsive Design** - Works on desktop and mobile
- ⚡ **Real-time Predictions** - ML model integration ready

## Troubleshooting

### Google Maps not loading?

- Check API key is correct in `.env`
- Verify API key restrictions in Google Cloud Console
- Check browser console for errors

### CORS errors?

- Ensure backend is running on port 8000
- Check `VITE_API_URL` in `.env` file

### Build errors?

- Run `npm install` to ensure dependencies are installed
- Check Node.js version (requires 18+)
- Clear node_modules: `rm -rf node_modules && npm install`

## Deployment

### Netlify (Recommended)

1. Connect your GitHub repository
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Add environment variables:
   - `VITE_GOOGLE_MAPS_API_KEY`
   - `VITE_API_URL`

### Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel`
3. Add environment variables in dashboard

### Manual Build

```bash
npm run build
# Upload dist/ folder to your hosting provider
```

## Next Steps

1. **Get Google Maps API Key** - Follow the setup guide above
2. **Start Backend** - Run the backend server (see BACKEND_SETUP.md)
3. **Test Integration** - Make sure frontend can connect to backend
4. **Deploy** - Choose your deployment option

## Support

If you encounter issues:

1. Check the browser console for errors
2. Verify all environment variables are set
3. Ensure both frontend and backend are running
4. Check the API endpoints are accessible
