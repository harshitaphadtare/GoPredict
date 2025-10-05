import { useEffect } from 'react'
import axios from 'axios'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix for default markers not showing in bundled environments
delete (L.Icon.Default.prototype as any)._getIconUrl

// Create custom colored markers
const createCustomIcon = (color: string) => {
  return L.divIcon({
    className: 'custom-marker',
    html: `<div style="
      background-color: ${color};
      width: 20px;
      height: 20px;
      border-radius: 50%;
      border: 3px solid white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    "></div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  })
}

export type GeoPoint = {
  name?: string
  lat: number
  lon: number
}

interface LeafletMapProps {
  from?: GeoPoint | null
  to?: GeoPoint | null
  animateKey?: string | number
}

export default function LeafletMap({ from, to, animateKey }: LeafletMapProps) {
  useEffect(() => {
    const map = L.map('route-map', {
      zoomControl: true,
    })

    const apiKey = import.meta.env.VITE_GEOAPIFY_API_KEY
    const tileUrl = apiKey
      ? `https://maps.geoapify.com/v1/tile/osm-bright/{z}/{x}/{y}.png?apiKey=${apiKey}`
      : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'

    const tiles = L.tileLayer(tileUrl, {
      maxZoom: 19,
      attribution:
        apiKey
          ? '© OpenMapTiles © OpenStreetMap contributors | © Geoapify'
          : '© OpenStreetMap contributors',
    })
    tiles.addTo(map)

    let markers: L.Marker[] = []
    let routeLayer: L.Polyline | L.GeoJSON | null = null
  let debugControl: L.Control | null = null

    const fitBoundsIfNeeded = () => {
      const points: L.LatLngExpression[] = []
      if (from) points.push([from.lat, from.lon])
      if (to) points.push([to.lat, to.lon])
      if (points.length) {
        const bounds = L.latLngBounds(points)
        map.fitBounds(bounds.pad(0.25))
      } else {
        map.setView([40.7128, -74.006], 11)
      }
    }

    const drawMarkers = () => {
      markers.forEach(m => m.remove())
      markers = []
      if (from) {
        const startMarker = L.marker([from.lat, from.lon], {
          icon: createCustomIcon('#10b981') // Green for start
        }).addTo(map)
        startMarker.bindPopup(`<b>Start:</b> ${from.name || 'Start Location'}`)
        markers.push(startMarker)
      }
      if (to) {
        const endMarker = L.marker([to.lat, to.lon], {
          icon: createCustomIcon('#ef4444') // Red for end
        }).addTo(map)
        endMarker.bindPopup(`<b>End:</b> ${to.name || 'End Location'}`)
        markers.push(endMarker)
      }
    }

    const addDebugControl = () => {
      // create a small control in top-right that shows provider / errors
      const ctrl = (L.control as any)({ position: 'topright' })
      ctrl.onAdd = function () {
        const el = L.DomUtil.create('div', 'route-debug-badge') as HTMLDivElement
        el.style.padding = '6px 8px'
        el.style.background = 'rgba(0,0,0,0.65)'
        el.style.color = 'white'
        el.style.fontSize = '12px'
        el.style.borderRadius = '6px'
        el.style.boxShadow = '0 1px 4px rgba(0,0,0,0.3)'
        el.innerText = 'Route: none'
        return el
      }
      ctrl.addTo(map)
      debugControl = ctrl
    }

    const setDebugText = (text: string) => {
      try {
        if (debugControl) {
          const c = (debugControl as any).getContainer?.() || (debugControl as any)._container
          if (c) c.innerText = text
        }
      } catch (err) {
        // Ignore debug UI errors
        console.debug('setDebugText error', err)
      }
    }

    const drawStraight = () => {
      if (routeLayer) routeLayer.remove()
      if (from && to) {
        const straight = L.polyline(
          [
            [from.lat, from.lon],
            [to.lat, to.lon],
          ],
          { color: '#2563eb', weight: 3, opacity: 0.85 }
        ).addTo(map)
        routeLayer = straight
      }
    }

    const fetchRoute = async () => {
      if (!from || !to) return

      const orsKey = import.meta.env.VITE_ORS_API_KEY
      const geoKey = import.meta.env.VITE_GEOAPIFY_API_KEY
      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

      const drawFeatures = (features: any) => {
        const coords = features[0].geometry.coordinates.map((c: [number, number]) => [c[1], c[0]])
        if (routeLayer) routeLayer.remove()
        const route = L.polyline(coords, { color: '#2563eb', weight: 4, opacity: 0.95 }).addTo(map)
        routeLayer = route
        map.fitBounds(route.getBounds().pad ? route.getBounds().pad(0.15) : route.getBounds())
      }

      try {
        // 1) Try ORS client-side, fall back to server proxy
        if (orsKey) {
          const url = `https://api.openrouteservice.org/v2/directions/driving-car?start=${from.lon},${from.lat}&end=${to.lon},${to.lat}`
          setDebugText('Route: ORS (fetching...)')
          try {
            const res = await axios.get(url, { headers: { Authorization: orsKey } })
            if (res.data?.features?.[0]) {
              drawFeatures(res.data.features)
              setDebugText(`Route: ORS (${res.data.features[0].geometry.coordinates.length} pts)`)
              return
            }
          } catch (e) {
            console.debug('LeafletMap: ORS client request failed, will try proxy', e)
            setDebugText('Route: ORS (proxying...)')
            // fallthrough to proxy below
          }
        }

        // 2) Try Geoapify client-side (if configured)
        if (geoKey) {
          const url = `https://api.geoapify.com/v1/routing?waypoints=${from.lon},${from.lat}|${to.lon},${to.lat}&mode=drive&format=geojson&apiKey=${geoKey}`
          setDebugText('Route: Geoapify (fetching...)')
          try {
            const res = await axios.get(url)
            if (res.data?.features?.[0]) {
              routeLayer && routeLayer.remove()
              routeLayer = L.geoJSON(res.data.features[0], { style: { color: '#2563eb', weight: 4, opacity: 0.95 } }).addTo(map)
              map.fitBounds((routeLayer as any).getBounds().pad ? (routeLayer as any).getBounds().pad(0.15) : (routeLayer as any).getBounds())
              setDebugText('Route: Geoapify (done)')
              return
            }
          } catch (e) {
            console.debug('LeafletMap: Geoapify client request failed, will try proxy', e)
            setDebugText('Route: Geoapify (proxying...)')
          }
        }

        // 3) Try server proxy directly
        setDebugText('Route: server (fetching...)')
        try {
          const res = await axios.post(`${API_BASE_URL}/api/routing`, { from, to })
          if (res.data?.features?.[0]) {
            drawFeatures(res.data.features)
            setDebugText('Route: server (done)')
            return
          }
        } catch (err) {
          const upstreamCode = (err as any)?.response?.data?.error?.code ?? (err as any)?.response?.status
          // ORS returns 2010 for non-routable points; this is expected and will be retried via nudging.
          // Avoid noisy console.error for that specific case.
          if (upstreamCode !== 2010 && upstreamCode !== 404) {
            console.debug('LeafletMap: server proxy returned error', err)
          }
          // If ORS returned non-routable error, attempt nudging the destination
          if (upstreamCode === 2010 || upstreamCode === 404) {
            setDebugText('Route: server (nudging dest)')

            const metersToDegrees = (meters: number, lat: number) => {
              const dLat = meters / 111111
              const dLon = meters / (111111 * Math.cos((lat * Math.PI) / 180))
              return { dLat, dLon }
            }

            const tryNudges = async () => {
              const radii = [50, 150, 300]
              const angles = Array.from({ length: 8 }, (_, i) => (i * Math.PI) / 4)
              for (const r of radii) {
                const { dLat, dLon } = metersToDegrees(r, to.lat)
                for (const a of angles) {
                  const nx = to.lon + Math.cos(a) * dLon
                  const ny = to.lat + Math.sin(a) * dLat
                  try {
                    const res2 = await axios.post(`${API_BASE_URL}/api/routing`, { from, to: { ...to, lon: nx, lat: ny } })
                    if (res2.data?.features?.[0]) return { features: res2.data.features, nudged: { lon: nx, lat: ny } }
                  } catch (_) {
                    await new Promise(r => setTimeout(r, 150))
                    continue
                  }
                }
              }
              return null
            }

            try {
              const nudged = await tryNudges()
              if (nudged && nudged.features && nudged.features[0]) {
                drawFeatures(nudged.features)
                setDebugText('Route: server (nudged)')
                return
              }
            } catch (inner) {
              console.debug('LeafletMap: nudging attempts failed', inner)
            }
          }
          // rethrow to outer catch
          throw err
        }

        // Nothing worked: fall back to straight line
        console.debug('LeafletMap: no ORS or Geoapify route available, using straight-line fallback')
        setDebugText('Route: none (no key)')
        drawStraight()
      } catch (err) {
        console.error('Routing error:', err)
        try { alert('Unable to fetch driving route (routing service failed). Showing straight-line fallback.') } catch {}
        setDebugText('Route: error')
        drawStraight()
      }
    }

    drawMarkers()
    fitBoundsIfNeeded()
    // Always draw something quickly, then try to replace with routed geometry
    addDebugControl()
    if (from && to) {
      drawStraight()
      fetchRoute()
    }

    return () => {
      map.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [from?.lat, from?.lon, to?.lat, to?.lon, animateKey])

  return (
    <div className="w-full overflow-hidden rounded-2xl border border-border bg-card/90 shadow-soft backdrop-blur">
      <div className="flex items-center justify-between border-b border-border px-4 py-3 text-sm text-foreground/70">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-2 rounded-full bg-primary" />
          <span>Route Preview</span>
        </div>
        {from && to ? (
          <span className="truncate">{from.name ?? 'Start'} → {to.name ?? 'End'}</span>
        ) : (
          <span className="truncate">Select locations to preview</span>
        )}
      </div>
      <div id="route-map" style={{ height: 360, width: '100%' }} />
    </div>
  )
}


