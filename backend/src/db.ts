import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config(); // Load environment variables from .env

const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
    console.error('❌ FATAL ERROR: MONGO_URI is not defined in .env file');
    process.exit(1); // Exit if connection string is missing
}

const connectDB = async () => {
    try {
        // Added options recommended by Mongoose for latest versions
        await mongoose.connect(MONGO_URI);
        console.log('✅ MongoDB Connected Successfully');
    } catch (error: any) { // Added type annotation for error
        console.error('❌ MongoDB Connection Error:', error.message); // Log only the message
        process.exit(1); // Exit process with failure
    }
};

// Handle connection events (optional but good practice)
mongoose.connection.on('disconnected', () => {
    console.log('MongoDB disconnected.');
});

mongoose.connection.on('error', (err) => {
    console.error('MongoDB connection error:', err);
});


export default connectDB;

