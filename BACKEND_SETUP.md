# GoPredict Backend Setup Guide

## Quick Start

### 1. Prerequisites

- Node.js 18+ installed
- Your trained ML model (optional for demo)

### 2. Setup Backend

```bash
# Run the setup script
setup-backend.bat

# Or manually:
cd backend
npm install
copy env.example .env
```

### 3. Start Server

```bash
cd backend
npm start
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

```http
GET /api/health
```

Response:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "v1.0-demo",
  "uptime": 123.45
}
```

### Prediction

```http
POST /api/predict
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

Response:

```json
{
  "minutes": 25.5,
  "confidence": 0.85,
  "model_version": "v1.0-demo",
  "distance_km": 2.3,
  "city": "new_york"
}
```

## Project Structure

```
backend/
├── routes/
│   └── predict.js           # Prediction API logic
├── server.js                # Main server file
├── package.json             # Dependencies
└── env.example             # Environment template
```

## Environment Configuration

Edit `backend/.env`:

```env
PORT=8000
NODE_ENV=development
FRONTEND_URL=http://localhost:3000
MODEL_PATH=../saved_models/xgboost_20250930_103824.pkl
```

## Integration with Your ML Model

### Current Implementation

The backend currently uses a simple distance-based prediction algorithm. To integrate your trained ML model:

### 1. Update Prediction Logic

Edit `backend/routes/predict.js`:

```javascript
import { loadModel, predict } from "../ml/model.js";

export async function predictRoute(req, res) {
  try {
    const { from, to, startTime, city } = req.body;

    // Load your trained model
    const model = await loadModel();

    // Prepare features for your model
    const features = prepareFeatures(from, to, startTime, city);

    // Get prediction from your ML model
    const prediction = await model.predict(features);

    res.json({
      minutes: prediction.minutes,
      confidence: prediction.confidence,
      model_version: "v1.0-trained",
      distance_km: calculateDistance(from, to),
      city: city,
    });
  } catch (error) {
    // Fallback to simple calculation
    const fallbackResult = calculateFallback(req.body);
    res.json(fallbackResult);
  }
}
```

### 2. Create ML Integration Module

Create `backend/ml/model.js`:

```javascript
import { readFileSync } from "fs";
import { join } from "path";

let model = null;

export async function loadModel() {
  if (!model) {
    // Load your trained model
    const modelPath =
      process.env.MODEL_PATH || "../saved_models/xgboost_20250930_103824.pkl";
    model = await loadPickleModel(modelPath);
  }
  return model;
}

export function prepareFeatures(from, to, startTime, city) {
  const date = new Date(startTime);

  return {
    start_lat: from.lat,
    start_lon: from.lon,
    end_lat: to.lat,
    end_lon: to.lon,
    hour: date.getHours(),
    day_of_week: date.getDay(),
    month: date.getMonth(),
    city_encoded: city === "new_york" ? 0 : 1,
    distance: calculateDistance(from, to),
    // Add more features as needed
  };
}
```

### 3. Python Integration (Alternative)

If you prefer to keep your ML model in Python:

```javascript
import { spawn } from "child_process";

export async function predictWithPython(features) {
  return new Promise((resolve, reject) => {
    const python = spawn("python", ["../src/predict.py"], {
      cwd: process.cwd(),
    });

    python.stdin.write(JSON.stringify(features));
    python.stdin.end();

    let result = "";
    python.stdout.on("data", (data) => {
      result += data.toString();
    });

    python.on("close", (code) => {
      if (code === 0) {
        resolve(JSON.parse(result));
      } else {
        reject(new Error("Python script failed"));
      }
    });
  });
}
```

## Testing the API

### 1. Health Check

```bash
curl http://localhost:8000/api/health
```

### 2. Test Prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "from": {"id": "ny_times_square", "name": "Times Square, NYC", "lat": 40.7580, "lon": -73.9855},
    "to": {"id": "ny_central_park", "name": "Central Park, NYC", "lat": 40.7829, "lon": -73.9654},
    "startTime": "2024-01-15T14:30:00.000Z",
    "city": "new_york"
  }'
```

### 3. Frontend Integration

Make sure your frontend is configured to connect to `http://localhost:8000` in the `.env` file.

## Deployment Options

### 1. Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### 2. Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: node server.js" > Procfile

# Deploy
heroku create gopredict-api
git push heroku main
```

### 3. DigitalOcean App Platform

1. Connect your GitHub repository
2. Set build command: `npm install`
3. Set run command: `npm start`
4. Add environment variables

### 4. AWS EC2

```bash
# Install PM2 for process management
npm install -g pm2

# Start with PM2
pm2 start server.js --name gopredict-api
pm2 save
pm2 startup
```

## Environment Variables

| Variable       | Description      | Default               |
| -------------- | ---------------- | --------------------- |
| `PORT`         | Server port      | 8000                  |
| `NODE_ENV`     | Environment      | development           |
| `FRONTEND_URL` | CORS origin      | http://localhost:3000 |
| `MODEL_PATH`   | Path to ML model | ../saved_models/      |

## Monitoring

### Health Monitoring

```bash
# Check if server is running
curl http://localhost:8000/api/health

# Monitor logs
tail -f logs/app.log
```

### Performance Monitoring

Consider adding:

- Request logging
- Response time monitoring
- Error tracking (Sentry)
- Uptime monitoring

## Security

### CORS Configuration

```javascript
app.use(
  cors({
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    credentials: true,
  })
);
```

### Rate Limiting

```bash
npm install express-rate-limit
```

```javascript
import rateLimit from "express-rate-limit";

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
});

app.use("/api/", limiter);
```

## Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   # Find process using port 8000
   netstat -ano | findstr :8000
   # Kill the process
   taskkill /PID <PID> /F
   ```

2. **CORS errors**

   - Check `FRONTEND_URL` in `.env`
   - Ensure frontend is running on correct port

3. **Module not found**

   ```bash
   npm install
   ```

4. **ML model integration**
   - Ensure model file exists
   - Check file permissions
   - Verify Python dependencies if using Python integration

## Next Steps

1. **Test API** - Verify all endpoints work
2. **Integrate ML Model** - Connect your trained model
3. **Deploy** - Choose your deployment option
4. **Monitor** - Set up monitoring and logging
5. **Scale** - Consider load balancing for production
