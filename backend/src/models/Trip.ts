import mongoose, { Schema, Document } from 'mongoose';

// Interface for Location subdocument
interface ILocation {
    name: string;
    lat: number;
    lon: number;
}

export interface ITrip extends Document {
    firebaseUid: string;
    timestamp: Date;
    travelDateTime: Date;
    startLocation: ILocation;
    endLocation: ILocation;
    city: 'new_york' | 'san_francisco';
    predictedDuration: number;
}

const LocationSchema: Schema = new Schema<ILocation>({ // Use typed Schema
    name: { type: String, required: true },
    lat: { type: Number, required: true },
    lon: { type: Number, required: true },
}, { _id: false });

const TripSchema: Schema = new Schema<ITrip>({ // Use typed Schema
    firebaseUid: { type: String, required: true, index: true },
    timestamp: { type: Date, default: Date.now }, // When saved
    travelDateTime: { type: Date, required: true }, // For the trip
    startLocation: { type: LocationSchema, required: true },
    endLocation: { type: LocationSchema, required: true },
    city: { type: String, enum: ['new_york', 'san_francisco'], required: true },
    predictedDuration: { type: Number, required: true }, // In minutes
}, {
    timestamps: { createdAt: 'timestamp' } // Alternative way to manage creation time if needed
});

TripSchema.index({ firebaseUid: 1, travelDateTime: -1 }); // Index for user history queries

// Ensure virtuals are included if you use them later
TripSchema.set('toJSON', { virtuals: true });
TripSchema.set('toObject', { virtuals: true });

export default mongoose.model<ITrip>('Trip', TripSchema);

