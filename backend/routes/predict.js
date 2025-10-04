const NodeCache = require('node-cache');

// Initialize cache with 30 minute TTL (time to live)
const predictionCache = new NodeCache({ 
  stdTTL: 1800,
  checkperiod: 120
});

// Create cache key from request parameters
function createCacheKey(from, to, startTime, city) {
  return `${from.lat},${from.lon}|${to.lat},${to.lon}|${startTime}|${city}`;
}

// Simple distance-based prediction (replace with your ML model)
function calculateDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth's radius in kilometers
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

function estimateTravelTime(distanceKm, startTime, city) {
  const date = new Date(startTime);
  const hour = date.getHours();
  const dayOfWeek = date.getDay();
  
  // Base speed factors
  let baseSpeed = 25; // km/h
  
  // Rush hour adjustments
  const isRushHour = (hour >= 7 && hour <= 10) || (hour >= 16 && hour <= 19);
  const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
  
  if (isRushHour && !isWeekend) {
    baseSpeed *= 0.6; // 40% slower during rush hour
  } else if (isWeekend) {
    baseSpeed *= 0.8; // 20% slower on weekends
  }
  
  // City-specific adjustments
  if (city === 'new_york') {
    baseSpeed *= 0.9; // NYC is generally slower
  } else if (city === 'san_francisco') {
    baseSpeed *= 0.85; // SF has more hills and traffic
  }
  
  // Calculate time in minutes
  const timeHours = distanceKm / baseSpeed;
  const timeMinutes = timeHours * 60;
  
  // Add some randomness for realism (±10%)
  const variation = (Math.random() - 0.5) * 0.2;
  return Math.max(5, timeMinutes * (1 + variation));
}

export function predictRoute(req, res) {
  try {
    const { from, to, startTime, city } = req.body;
    
    // Validate input
    if (!from || !to || !startTime || !city) {
      return res.status(400).json({ 
        error: 'Missing required fields: from, to, startTime, city' 
      });
    }
    
    // Validate coordinates
    if (typeof from.lat !== 'number' || typeof from.lon !== 'number' ||
        typeof to.lat !== 'number' || typeof to.lon !== 'number') {
      return res.status(400).json({ 
        error: 'Invalid coordinates' 
      });
    }

    // Check cache first
    const cacheKey = createCacheKey(from, to, startTime, city);
    const cachedResult = predictionCache.get(cacheKey);
    
    if (cachedResult) {
      return res.json({
        ...cachedResult,
        cached: true
      });
    }
    
    // Calculate distance
    const distance = calculateDistance(from.lat, from.lon, to.lat, to.lon);
    
    // Estimate travel time
    const minutes = estimateTravelTime(distance, startTime, city);
    
    // Create prediction result
    const prediction = {
      minutes: Math.round(minutes * 10) / 10,
      confidence: 0.75 + Math.random() * 0.2,
      model_version: 'v1.0-demo',
      distance_km: Math.round(distance * 10) / 10,
      city: city,
      timestamp: new Date().toISOString()
    };

    // Store in cache
    predictionCache.set(cacheKey, prediction);
    
    // Return prediction
    res.json({
      ...prediction,
      cached: false
    });
    
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message
    });
  }
}

// Add cache stats to health check
export function healthCheck(req, res) {
  const stats = predictionCache.getStats();
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: 'v1.0-demo',
    uptime: process.uptime(),
    cache: {
      keys: predictionCache.keys().length,
      hits: stats.hits,
      misses: stats.misses,
      ksize: stats.ksize,
      vsize: stats.vsize
    }
  });
}
