import express from 'express'
import axios from 'axios'

const router = express.Router()

// POST /api/routing
// Body: { from: { lat, lon }, to: { lat, lon } }
router.post('/', async (req, res, next) => {
  try {
    const { from, to } = req.body
    if (!from || !to) return res.status(400).json({ error: 'Missing from/to in body' })

    // Basic validation: ensure lat and lon are present and numeric
    const isValidCoord = (c) => c && typeof c.lat !== 'undefined' && typeof c.lon !== 'undefined' && Number.isFinite(Number(c.lat)) && Number.isFinite(Number(c.lon))
    if (!isValidCoord(from) || !isValidCoord(to)) {
      return res.status(400).json({ error: 'Invalid from/to coordinates. Expect { lat: number, lon: number }' })
    }

    const orsKey = process.env.ORS_API_KEY
    if (!orsKey) return res.status(500).json({ error: 'Routing key not configured on server' })

    const url = `https://api.openrouteservice.org/v2/directions/driving-car?start=${from.lon},${from.lat}&end=${to.lon},${to.lat}`
    try {
  const resp = await axios.get(url, { headers: { Authorization: orsKey }, timeout: 10000 })
      return res.json(resp.data)
    } catch (err) {
      // Log upstream error for local debugging and proxy the upstream error body/status
      const status = err?.response?.status
      const upstreamError = err?.response?.data?.error
      const upstreamCode = upstreamError?.code

      // ORS returns 404 + code 2010 for non-routable coordinates (expected for some centroids).
      // Return a sanitized error object so tests and clients always receive the expected shape.
      if (status === 404 && upstreamCode === 2010) {
        if (process.env.SHOW_ROUTING_DEBUG === '1') {
          console.debug('Routing proxy upstream non-routable (2010):', err?.response?.data)
        }
        // Always return the expected error object, ignoring upstream message.
        return res.status(404).json({ error: { code: 2010, message: 'Could not find routable point' } })
      }

      // Log other upstream errors and proxy the full upstream body/status when present
      console.error('Routing proxy upstream error:', status, err?.response?.data)
      if (err?.response) {
        return res.status(err.response.status).json(err.response.data)
      }
      throw err
    }
  } catch (err) {
    next(err)
  }
})

export default router
