// API integration for GoPredict backend

export interface PredictionRequest {
  from: {
    id: string;
    name: string;
    lat: number;
    lon: number;
  };
  to: {
    id: string;
    name: string;
    lat: number;
    lon: number;
  };
  startTime: string;
  city: 'new_york' | 'san_francisco';
}

export interface PredictionResponse {
  minutes: number;
  confidence?: number;
  model_version?: string;
  distance_km?: number;
  city?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function predictTravelTime(request: PredictionRequest): Promise<PredictionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Prediction API error:', error);
    throw error;
  }
}

export async function getModelStatus(): Promise<{ status: string; version?: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/status`);
    
    if (!response.ok) {
      throw new Error(`Status check failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Status check error:', error);
    return { status: 'offline' };
  }
}
