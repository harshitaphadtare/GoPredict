import express, { Express, Request, Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import connectDB from './db';
import '../config/firebaseAdmin'; // <-- CORRECTED PATH HERE
// --- Import Routes ---
import profileRoutes from './routes/profile';
import tripRoutes from './routes/trips';
import userRoutes from './routes/users';
// -------------------

dotenv.config();
connectDB(); // Connect to DB

const app: Express = express();
const port = process.env.PORT || 5001;

app.use(cors({ origin: process.env.FRONTEND_URL || 'http://localhost:5173', credentials: true }));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// --- Health Check Route ---
app.get('/api/health', (req: Request, res: Response) => {
  res.json({ status: 'UP', timestamp: new Date().toISOString() });
});

// --- API Routes ---
app.use('/api/profile', profileRoutes);
app.use('/api/trips', tripRoutes);
app.use('/api/users', userRoutes);
// ------------------

app.listen(port, () => {
  console.log(` Backend server listening on http://localhost:${port}`);
});

