import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { predictRoute, healthCheck } from './routes/predict.js';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
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
  console.error('Error:', err);
  res.status(500).json({
    error: 'Internal Server Error',
    message: err.message
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.originalUrl} not found`
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 GoPredict API Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`🔮 Prediction endpoint: http://localhost:${PORT}/api/predict`);
});

export default app;
