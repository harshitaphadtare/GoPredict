import NodeCache from 'node-cache';

const RAD = Math.PI / 180;
const EARTH_RADIUS_KM = 6371;
const ONE_DECIMAL = n => Math.round(n * 10) / 10;

const predictionCache = new NodeCache({
  stdTTL: 1800,
  checkperiod: 120,
  useClones: false,
  deleteOnExpire: true,
});

const citySpeedFactor = Object.freeze({
  new_york: 0.9,
  san_francisco: 0.85,
});

function createCacheKey(from, to, startTime, city) {
  return `${from.lat},${from.lon}|${to.lat},${to.lon}|${startTime}|${city}`;
}

function calculateDistance(lat1, lon1, lat2, lon2) {
  const dLat = (lat2 - lat1) * RAD;
  const dLon = (lon2 - lon1) * RAD;
  const lat1r = lat1 * RAD;
  const lat2r = lat2 * RAD;

  const s1 = Math.sin(dLat * 0.5);
  const s2 = Math.sin(dLon * 0.5);

  const a = s1 * s1 + Math.cos(lat1r) * Math.cos(lat2r) * (s2 * s2);
  return EARTH_RADIUS_KM * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
}

function estimateTravelTime(distanceKm, startTime, city) {
  const date = new Date(startTime);
  const t = Number.isNaN(date.getTime()) ? new Date() : date; 
  const hour = t.getHours();
  const dow = t.getDay();

  let baseSpeed = 25; 

  const isWeekend = (dow === 0 || dow === 6);
  const isRushHour = (!isWeekend) && ((hour >= 7 && hour <= 10) || (hour >= 16 && hour <= 19));

  if (isRushHour) {
    baseSpeed *= 0.6; 
  } else if (isWeekend) {
    baseSpeed *= 0.8; 
  }

  const cityFactor = citySpeedFactor[city] ?? 1;
  baseSpeed *= cityFactor;

  const timeMinutes = (distanceKm / baseSpeed) * 60;
  const variation = (Math.random() - 0.5) * 0.2;
  const minutes = timeMinutes * (1 + variation);
  return minutes > 5 ? minutes : 5;
}

export const predictRoute = (req, res) => {
  try {
    const { from, to, startTime, city } = req.body || {};

    if (!from || !to || !startTime || !city) {
      return res.status(400).json({ error: 'Missing required fields: from, to, startTime, city' });
    }

    if (
      !Number.isFinite(from.lat) || !Number.isFinite(from.lon) ||
      !Number.isFinite(to.lat)   || !Number.isFinite(to.lon)   ||
      from.lat < -90 || from.lat > 90 || to.lat < -90 || to.lat > 90 ||
      from.lon < -180 || from.lon > 180 || to.lon < -180 || to.lon > 180
    ) {
      return res.status(400).json({ error: 'Invalid coordinates' });
    }

    const cityKey = String(city).toLowerCase();

    const cacheKey = createCacheKey(from, to, startTime, cityKey);
    const cached = predictionCache.get(cacheKey);
    if (cached !== undefined) {
      return res.json({ ...cached, cached: true });
    }

    const distanceKm = calculateDistance(from.lat, from.lon, to.lat, to.lon);
    const minutes = estimateTravelTime(distanceKm, startTime, cityKey);

    const nowISO = new Date().toISOString();
    const prediction = {
      minutes: ONE_DECIMAL(minutes),
      confidence: 0.75 + Math.random() * 0.2,
      model_version: 'v1.0-demo',
      distance_km: ONE_DECIMAL(distanceKm),
      city: cityKey,
      timestamp: nowISO,
    };

    predictionCache.set(cacheKey, prediction);
    return res.json({ ...prediction, cached: false });
  } catch (error) {
    console.error('Prediction error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error?.message ?? String(error),
    });
  }
};

export const healthCheck = (_req, res) => {
  const stats = predictionCache.getStats();
  return res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: 'v1.0-demo',
    uptime: process.uptime(),
    cache: {
      keys: stats.keys,
      hits: stats.hits,
      misses: stats.misses,
      ksize: stats.ksize,
      vsize: stats.vsize,
    },
  });
};
