import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { predictRoute, healthCheck } from './routes/predict.js';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 8000;

const allowedOrigins = (process.env.FRONTEND_URL || 'http://localhost:3000')
  .split(',')
  .map(origin => origin.trim());

console.log(`[server.js]:${allowedOrigins}`)
// Middleware
app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/api/health', healthCheck);
app.get('/api/status', healthCheck);
app.post('/api/predict', predictRoute);

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'GoPredict API Server',
    version: '1.0.0',
    status: 'running',
    endpoints: {
      health: '/api/health',
      predict: '/api/predict',
      status: '/api/status'
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  // Respect known status codes (e.g., body-parser JSON errors set status=400)
  const status = err?.status || err?.statusCode || (err?.type === 'entity.parse.failed' ? 400 : 500);
  const isServerError = status >= 500;
  const errorName = isServerError ? 'Internal Server Error' : 'Bad Request';

  // Avoid noisy logs during tests
  if (process.env.NODE_ENV !== 'test') {
    console.error('Error:', err);
  }

  res.status(status).json({
    error: errorName,
    message: err?.message || errorName
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.originalUrl} not found`
  });
});

// Start server only if not in test mode
if (process.env.NODE_ENV !== 'test') {
  app.listen(PORT, () => {
    console.log(`🚀 GoPredict API Server running on port ${PORT}`);
    console.log(`✅ CORS enabled for origins: ${allowedOrigins.join(', ')}`);
    console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
    console.log(`🔮 Prediction endpoint: http://localhost:${PORT}/api/predict`);
  });
}

export default app;
