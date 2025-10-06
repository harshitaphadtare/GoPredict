import express from 'express'
import axios from 'axios'

const router = express.Router()

// POST /api/routing
// Body: { from: { lat, lon }, to: { lat, lon } }
router.post('/', async (req, res, next) => {
  try {
    const { from, to } = req.body
    if (!from || !to) return res.status(400).json({ error: 'Missing from/to in body' })

    const orsKey = process.env.ORS_API_KEY
    if (!orsKey) return res.status(500).json({ error: 'Routing key not configured on server' })

    const url = `https://api.openrouteservice.org/v2/directions/driving-car?start=${from.lon},${from.lat}&end=${to.lon},${to.lat}`
    try {
      const resp = await axios.get(url, { headers: { Authorization: orsKey } })
      return res.json(resp.data)
    } catch (err) {
      // Log upstream error for local debugging and proxy the upstream error body/status
      const status = err?.response?.status
      const upstreamCode = err?.response?.data?.error?.code
      // ORS returns 404 + code 2010 for non-routable coordinates (expected for some centroids).
      // Log those at debug level to avoid noisy server error logs during normal retry/nudge flow.
      if (status === 404 && upstreamCode === 2010) {
        // Non-routable points (ORS code 2010) are expected for some coordinates.
        // Forward the upstream `error` object exactly as the test expects.
        if (process.env.SHOW_ROUTING_DEBUG === '1') {
          console.debug('Routing proxy upstream non-routable (2010):', err?.response?.data)
        }
        return res.status(404).json({ error: err.response.data.error })
      }

      // Log other upstream errors and proxy the full upstream body/status
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
