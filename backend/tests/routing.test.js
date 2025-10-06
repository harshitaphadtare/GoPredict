import request from 'supertest'
import app from '../server.js'
import axios from 'axios'

jest.mock('axios')

describe('POST /api/routing', () => {
  const from = { lat: 40.758, lon: -73.9855 }
  const to = { lat: 40.7829, lon: -73.9654 }

  afterEach(() => jest.resetAllMocks())

  it('returns 200 and geojson features when ORS responds with route', async () => {
    const mockGeoJSON = { features: [{ geometry: { coordinates: [[-73.9855,40.758],[ -73.9654,40.7829 ]] } }] }
    axios.get.mockResolvedValueOnce({ status: 200, data: mockGeoJSON })

    const res = await request(app).post('/api/routing').send({ from, to })
    expect(res.status).toBe(200)
    expect(res.body).toHaveProperty('features')
  })

  it('forwards 404/2010 non-routable response when ORS returns non-routable', async () => {
    const error = {
      response: {
        status: 404,
        data: { error: { code: 2010, message: 'Could not find routable point' } }
      }
    }
    axios.get.mockRejectedValueOnce(error)

    const res = await request(app).post('/api/routing').send({ from, to })
    expect(res.status).toBe(404)
    expect(res.body).toHaveProperty('error')
    expect(res.body.error).toHaveProperty('code', 2010)
  })
})
