import request from 'supertest'
import app from '../server.js'

describe('Cache Functionality', () => {
  const sampleRequest = {
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

  it('should return cache statistics in health endpoint', async () => {
    const response = await request(app)
      .get('/api/health')
      .expect(200)

    expect(response.body).toHaveProperty('cache')
    expect(response.body.cache).toHaveProperty('keys')
    expect(response.body.cache).toHaveProperty('hits')
    expect(response.body.cache).toHaveProperty('misses')
    expect(response.body.cache).toHaveProperty('ksize')
    expect(response.body.cache).toHaveProperty('vsize')

    expect(typeof response.body.cache.keys).toBe('number')
    expect(typeof response.body.cache.hits).toBe('number')
    expect(typeof response.body.cache.misses).toBe('number')
  })

  it('should return cached=false for first request', async () => {
    // Use a unique request to avoid cache hits from previous tests
    const uniqueRequest = {
      ...sampleRequest,
      startTime: new Date().toISOString() // Current timestamp for uniqueness
    }

    const response = await request(app)
      .post('/api/predict')
      .send(uniqueRequest)
      .expect(200)

    expect(response.body).toHaveProperty('cached', false)
    expect(response.body).toHaveProperty('minutes')
    expect(response.body).toHaveProperty('confidence')
  })

  it('should return cached=true for identical subsequent request', async () => {
    const fixedTimestamp = '2024-12-25T15:30:00.000Z'
    const cacheTestRequest = {
      ...sampleRequest,
      startTime: fixedTimestamp
    }

    // First request - should be a cache miss
    const response1 = await request(app)
      .post('/api/predict')
      .send(cacheTestRequest)
      .expect(200)

    expect(response1.body).toHaveProperty('cached', false)
    const firstResult = response1.body

    // Second identical request - should be a cache hit
    const response2 = await request(app)
      .post('/api/predict')
      .send(cacheTestRequest)
      .expect(200)

    expect(response2.body).toHaveProperty('cached', true)
    
    // Results should be identical (from cache)
    expect(response2.body.minutes).toBe(firstResult.minutes)
    expect(response2.body.confidence).toBe(firstResult.confidence)
    expect(response2.body.distance_km).toBe(firstResult.distance_km)
  })

  it('should update cache statistics correctly', async () => {
    // Get initial stats
    const initialHealth = await request(app)
      .get('/api/health')
      .expect(200)

    const initialStats = initialHealth.body.cache

    // Make a new unique request (cache miss)
    const uniqueRequest = {
      ...sampleRequest,
      startTime: `${Date.now()}-unique`,  // Unique timestamp
      from: { ...sampleRequest.from, lat: 40.7500 } // Slightly different coordinates
    }

    await request(app)
      .post('/api/predict')
      .send(uniqueRequest)
      .expect(200)

    // Make the same request again (cache hit)
    await request(app)
      .post('/api/predict')
      .send(uniqueRequest)
      .expect(200)

    // Check updated stats
    const finalHealth = await request(app)
      .get('/api/health')
      .expect(200)

    const finalStats = finalHealth.body.cache

    // Should have more keys, at least one more miss, and at least one more hit
    expect(finalStats.keys).toBeGreaterThanOrEqual(initialStats.keys)
    expect(finalStats.misses).toBeGreaterThan(initialStats.misses)
    expect(finalStats.hits).toBeGreaterThan(initialStats.hits)
  })

  it('should generate different cache keys for different requests', async () => {
    const baseRequest = {
      ...sampleRequest,
      startTime: '2024-12-25T12:00:00.000Z'
    }

    // Different start location
    const request1 = {
      ...baseRequest,
      from: { ...baseRequest.from, lat: 40.7000 }
    }

    // Different end location  
    const request2 = {
      ...baseRequest,
      to: { ...baseRequest.to, lat: 40.8000 }
    }

    // Different time
    const request3 = {
      ...baseRequest,
      startTime: '2024-12-25T14:00:00.000Z'
    }

    // Different city
    const request4 = {
      ...baseRequest,
      city: 'san_francisco'
    }

    // All should return cached=false (different cache keys)
    const responses = await Promise.all([
      request(app).post('/api/predict').send(request1),
      request(app).post('/api/predict').send(request2),
      request(app).post('/api/predict').send(request3),
      request(app).post('/api/predict').send(request4)
    ])

    responses.forEach(response => {
      expect(response.status).toBe(200)
      expect(response.body).toHaveProperty('cached', false)
    })
  })
})