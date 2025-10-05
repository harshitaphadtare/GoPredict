import request from 'supertest';
import app from '../server.js';

describe('Backend API', () => {
  test('GET /api/health returns 200', async () => {
    const response = await request(app)
      .get('/api/health')
      .expect(200);
    
    expect(response.body).toHaveProperty('status', 'healthy');
  });

  test('GET /api/predict endpoint exists', async () => {
    const response = await request(app)
      .post('/api/predict')
      .send({});
    
    // Should not return 404
    expect(response.status).not.toBe(404);
  });
});
