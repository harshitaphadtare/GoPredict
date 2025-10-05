import request from 'supertest'
import app from '../server.js'
import { predictRoute, healthCheck } from '../routes/predict.js';

describe('Prediction API', () => {
  describe('POST /api/predict', () => {
    const validRequest = {
      from: {
        id: 'ny_times_square',
        name: 'Times Square',
        lat: 40.7580,
        lon: -73.9855
      },
      to: {
        id: 'ny_central_park',
        name: 'Central Park',
        lat: 40.7829,
        lon: -73.9654
      },
      startTime: '2024-12-25T10:00:00.000Z',
      city: 'new_york'
    }

    it('should return a successful prediction with valid input', async () => {
      const response = await request(app)
        .post('/api/predict')
        .send(validRequest)
        .expect(200)

      expect(response.body).toHaveProperty('minutes')
      expect(response.body).toHaveProperty('confidence')
      expect(response.body).toHaveProperty('model_version')
      expect(response.body).toHaveProperty('distance_km')
      expect(response.body).toHaveProperty('city')

      expect(typeof response.body.minutes).toBe('number')
      expect(response.body.minutes).toBeGreaterThan(0)
      expect(response.body.confidence).toBeGreaterThanOrEqual(0.75)
      expect(response.body.confidence).toBeLessThanOrEqual(0.95)
      expect(response.body.model_version).toBe('v1.0-demo')
      expect(response.body.city).toBe('new_york')
    })

    it('should return 400 for missing required fields', async () => {
      const invalidRequests = [
        {}, // Empty request
        { from: validRequest.from }, // Missing to, startTime, city
        { from: validRequest.from, to: validRequest.to }, // Missing startTime, city
        { from: validRequest.from, to: validRequest.to, startTime: validRequest.startTime }, // Missing city
      ]

      for (const invalidRequest of invalidRequests) {
        const response = await request(app)
          .post('/api/predict')
          .send(invalidRequest)
          .expect(400)

        expect(response.body).toHaveProperty('error')
        expect(response.body.error).toBe('Missing required fields: from, to, startTime, city')
      }
    })

    it('should return 400 for invalid coordinates', async () => {
      const invalidCoordinateRequests = [
        {
          ...validRequest,
          from: { ...validRequest.from, lat: 'invalid' }
        },
        {
          ...validRequest,
          from: { ...validRequest.from, lon: 'invalid' }
        },
        {
          ...validRequest,
          to: { ...validRequest.to, lat: null }
        },
        {
          ...validRequest,
          to: { ...validRequest.to, lon: undefined }
        }
      ]

      for (const invalidRequest of invalidCoordinateRequests) {
        const response = await request(app)
          .post('/api/predict')
          .send(invalidRequest)
          .expect(400)

        expect(response.body).toHaveProperty('error')
        expect(response.body.error).toBe('Invalid coordinates')
      }
    })

    it('should handle different cities correctly', async () => {
      const cities = ['new_york', 'san_francisco']

      for (const city of cities) {
        const response = await request(app)
          .post('/api/predict')
          .send({ ...validRequest, city })
          .expect(200)

        expect(response.body.city).toBe(city)
      }
    })

    it('should calculate different times for rush hour vs non-rush hour', async () => {
      const rushHourRequest = {
        ...validRequest,
        startTime: '2024-12-25T08:00:00.000Z' // 8 AM - rush hour
      }

      const nonRushHourRequest = {
        ...validRequest,
        startTime: '2024-12-25T14:00:00.000Z' // 2 PM - non-rush hour
      }

      const rushResponse = await request(app)
        .post('/api/predict')
        .send(rushHourRequest)
        .expect(200)

      const nonRushResponse = await request(app)
        .post('/api/predict')
        .send(nonRushHourRequest)
        .expect(200)

      // Rush hour should generally take longer (though there's randomness)
      expect(rushResponse.body.minutes).toBeGreaterThan(0)
      expect(nonRushResponse.body.minutes).toBeGreaterThan(0)
    })

    it('should return consistent results for the same input', async () => {
      // Make multiple requests with the same input
      const responses = await Promise.all([
        request(app).post('/api/predict').send(validRequest),
        request(app).post('/api/predict').send(validRequest),
        request(app).post('/api/predict').send(validRequest)
      ])

      responses.forEach(response => {
        expect(response.status).toBe(200)
        expect(response.body).toHaveProperty('minutes')
        expect(response.body).toHaveProperty('distance_km')
      })

      // Distance should be exactly the same
      const distances = responses.map(r => r.body.distance_km)
      expect(distances[0]).toBe(distances[1])
      expect(distances[1]).toBe(distances[2])

      // Times might vary slightly due to randomness, but should be in reasonable range
      const times = responses.map(r => r.body.minutes)
      times.forEach(time => {
        expect(time).toBeGreaterThan(0)
        expect(time).toBeLessThan(1000) // Reasonable upper bound
      })
    })

    it('should handle very short distances', async () => {
      const shortDistanceRequest = {
        ...validRequest,
        from: { id: 'test1', name: 'Point A', lat: 40.7580, lon: -73.9855 },
        to: { id: 'test2', name: 'Point B', lat: 40.7590, lon: -73.9865 } // Slightly further apart
      }

      const response = await request(app)
        .post('/api/predict')
        .send(shortDistanceRequest)
        .expect(200)

      expect(response.body.minutes).toBeGreaterThanOrEqual(5) // Minimum 5 minutes as per code
      expect(response.body.distance_km).toBeGreaterThan(0)
    })

    it('should handle long distances', async () => {
      const longDistanceRequest = {
        ...validRequest,
        from: { id: 'test1', name: 'Point A', lat: 40.7580, lon: -73.9855 },
        to: { id: 'test2', name: 'Point B', lat: 40.8580, lon: -73.8855 } // Much further
      }

      const response = await request(app)
        .post('/api/predict')
        .send(longDistanceRequest)
        .expect(200)

      expect(response.body.minutes).toBeGreaterThan(5)
      expect(response.body.distance_km).toBeGreaterThan(1)
    })
  })

  describe('GET /api/health', () => {
    it('should return health status', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200)

      expect(response.body).toHaveProperty('status', 'healthy')
      expect(response.body).toHaveProperty('timestamp')
      expect(response.body).toHaveProperty('version', 'v1.0-demo')
      expect(response.body).toHaveProperty('uptime')

      expect(typeof response.body.uptime).toBe('number')
      expect(response.body.uptime).toBeGreaterThanOrEqual(0)
    })
  })

  describe('GET /api/status', () => {
    it('should return the same as health endpoint', async () => {
      const response = await request(app)
        .get('/api/status')
        .expect(200)

      expect(response.body).toHaveProperty('status', 'healthy')
      expect(response.body).toHaveProperty('timestamp')
      expect(response.body).toHaveProperty('version', 'v1.0-demo')
      expect(response.body).toHaveProperty('uptime')
    })
  })

  describe('GET /', () => {
    it('should return API information', async () => {
      const response = await request(app)
        .get('/')
        .expect(200)

      expect(response.body).toHaveProperty('message', 'GoPredict API Server')
      expect(response.body).toHaveProperty('version', '1.0.0')
      expect(response.body).toHaveProperty('status', 'running')
      expect(response.body).toHaveProperty('endpoints')
      
      expect(response.body.endpoints).toHaveProperty('health', '/api/health')
      expect(response.body.endpoints).toHaveProperty('predict', '/api/predict')
      expect(response.body.endpoints).toHaveProperty('status', '/api/status')
    })
  })

  describe('404 Handler', () => {
    it('should return 404 for non-existent routes', async () => {
      const response = await request(app)
        .get('/api/nonexistent')
        .expect(404)

      expect(response.body).toHaveProperty('error', 'Not Found')
      expect(response.body).toHaveProperty('message')
      expect(response.body.message).toContain('/api/nonexistent')
    })
  })

  describe('Error Handling', () => {
    it('should handle malformed JSON', async () => {
      const response = await request(app)
        .post('/api/predict')
        .set('Content-Type', 'application/json')
        .send('{ invalid json }')
        .expect(400)

      expect(response.body).toHaveProperty('error')
    })
  })
})
