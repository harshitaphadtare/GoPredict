# GoPredict Full-Stack Setup Guide

## 🚀 Complete Project Setup

Your GoPredict project now has both frontend and backend code set up! Here's how to get everything running:

## 📁 Project Structure

```
GoPredict/
├── frontend/                 # React Frontend
│   ├── src/
│   │   ├── components/       # UI Components
│   │   ├── pages/           # App Pages
│   │   └── lib/             # Utilities & API
│   ├── package.json         # Frontend dependencies
│   └── env.example          # Environment template
├── backend/                  # Express Backend
│   ├── routes/              # API Routes
│   ├── server.js           # Main server
│   ├── package.json        # Backend dependencies
│   └── env.example         # Environment template
├── src/                     # Your ML Pipeline (existing)
├── saved_models/           # Your trained models (existing)
└── setup-*.bat            # Setup scripts
```

## 🛠️ Quick Setup (Windows)

### 1. Setup Frontend

```bash
# Run the setup script
setup-frontend.bat

# Or manually:
cd frontend
npm install
copy env.example .env
```

### 2. Setup Backend

```bash
# Run the setup script
setup-backend.bat

# Or manually:
cd backend
npm install
copy env.example .env
```

### 3. Get Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable: Maps JavaScript API, Directions API, Places API
3. Create API key
4. Edit `frontend/.env` and add your key:
   ```env
   VITE_GOOGLE_MAPS_API_KEY=your_actual_api_key_here
   ```

## 🚀 Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
npm start
```

Backend will run on `http://localhost:8000`

### Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

Frontend will run on `http://localhost:3000`

## 🔧 Configuration

### Frontend Environment (`frontend/.env`)

```env
VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
VITE_API_URL=http://localhost:8000
```

### Backend Environment (`backend/.env`)

```env
PORT=8000
NODE_ENV=development
FRONTEND_URL=http://localhost:3000
MODEL_PATH=../saved_models/xgboost_20250930_103824.pkl
```

## 🎯 Features Implemented

### Frontend Features

- ✅ **Google Maps Integration** - Interactive maps with route visualization
- ✅ **Smart Location Search** - Autocomplete for NY and SF locations
- ✅ **Dark/Light Mode** - System preference detection
- ✅ **Responsive Design** - Works on desktop and mobile
- ✅ **Real-time Predictions** - Connected to backend API
- ✅ **Location Validation** - Prevents cross-city travel
- ✅ **Car Animation** - Animated car along the route

### Backend Features

- ✅ **Express API Server** - RESTful API endpoints
- ✅ **CORS Configuration** - Frontend integration ready
- ✅ **Prediction Endpoint** - ML model integration ready
- ✅ **Health Checks** - Server monitoring
- ✅ **Error Handling** - Robust error management
- ✅ **Fallback Logic** - Works without ML model

## 🔗 API Endpoints

### Health Check

```http
GET http://localhost:8000/api/health
```

### Prediction

```http
POST http://localhost:8000/api/predict
Content-Type: application/json

{
  "from": {
    "id": "ny_times_square",
    "name": "Times Square, NYC",
    "lat": 40.7580,
    "lon": -73.9855
  },
  "to": {
    "id": "ny_central_park",
    "name": "Central Park, NYC",
    "lat": 40.7829,
    "lon": -73.9654
  },
  "startTime": "2024-01-15T14:30:00.000Z",
  "city": "new_york"
}
```

## 🧠 ML Model Integration

### Current Status

The backend currently uses a distance-based prediction algorithm. To integrate your trained ML model:

### Option 1: Python Integration

1. Create a Python script that loads your model
2. Call it from the Node.js backend
3. Update `backend/routes/predict.js`

### Option 2: Direct Model Loading

1. Use a Node.js ML library (TensorFlow.js, etc.)
2. Load your model directly in the backend
3. Replace the prediction logic

### Option 3: Microservice

1. Deploy your ML model as a separate service
2. Call it from the backend API
3. Handle failures gracefully

## 🚀 Deployment Options

### Frontend Deployment

- **Netlify** (Recommended) - Easy setup, automatic deployments
- **Vercel** - Great for React applications
- **AWS S3 + CloudFront** - Enterprise solution

### Backend Deployment

- **Railway** - Simple Node.js deployment
- **Heroku** - Popular platform
- **AWS EC2** - Full control
- **DigitalOcean** - Cost-effective

## 📊 Testing the Application

### 1. Test Backend

```bash
# Health check
curl http://localhost:8000/api/health

# Test prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"from":{"id":"ny_times_square","name":"Times Square, NYC","lat":40.7580,"lon":-73.9855},"to":{"id":"ny_central_park","name":"Central Park, NYC","lat":40.7829,"lon":-73.9654},"startTime":"2024-01-15T14:30:00.000Z","city":"new_york"}'
```

### 2. Test Frontend

1. Open `http://localhost:3000`
2. Select start and end locations
3. Choose date and time
4. Click "Predict Travel Time"
5. Verify map shows route and prediction

## 🔍 Troubleshooting

### Common Issues

1. **Google Maps not loading**

   - Check API key in `frontend/.env`
   - Verify API key restrictions in Google Cloud Console
   - Check browser console for errors

2. **CORS errors**

   - Ensure backend is running on port 8000
   - Check `VITE_API_URL` in frontend `.env`
   - Verify `FRONTEND_URL` in backend `.env`

3. **Module not found**

   ```bash
   # Frontend
   cd frontend && npm install

   # Backend
   cd backend && npm install
   ```

4. **Port already in use**
   ```bash
   # Find process using port
   netstat -ano | findstr :8000
   # Kill the process
   taskkill /PID <PID> /F
   ```

## 📚 Documentation

- **Frontend Setup**: `FRONTEND_SETUP.md`
- **Backend Setup**: `BACKEND_SETUP.md`
- **API Documentation**: See backend routes
- **Deployment Guide**: See individual setup files

## 🎉 Next Steps

1. **Get Google Maps API Key** - Follow the setup guide
2. **Test Both Services** - Ensure frontend and backend work together
3. **Integrate Your ML Model** - Replace the demo prediction logic
4. **Deploy to Production** - Choose your deployment platform
5. **Monitor and Scale** - Set up monitoring and scaling

## 🆘 Support

If you encounter issues:

1. Check the browser console for frontend errors
2. Check the terminal for backend errors
3. Verify all environment variables are set
4. Ensure both services are running
5. Test API endpoints individually

## 🏆 Success Criteria

Your setup is complete when:

- ✅ Frontend loads at `http://localhost:3000`
- ✅ Backend responds at `http://localhost:8000`
- ✅ Google Maps displays correctly
- ✅ Location search works for NY and SF
- ✅ Predictions are returned
- ✅ Map shows route with car animation

**You're all set! 🎉**
