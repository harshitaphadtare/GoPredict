import React, { useEffect } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

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
        markers.push(L.marker([from.lat, from.lon]).addTo(map))
      }
      if (to) {
        markers.push(L.marker([to.lat, to.lon]).addTo(map))
      }
    }

    const drawStraight = () => {
      if (!from || !to) return
      if (routeLayer) routeLayer.remove()
      routeLayer = L.polyline(
        [
          [from.lat, from.lon],
          [to.lat, to.lon],
        ],
        { color: '#2563eb', weight: 3, opacity: 0.85 }
      ).addTo(map)
      // @ts-ignore
      if ((routeLayer as any).bringToFront) (routeLayer as any).bringToFront()
    }

    const fetchRoute = async () => {
      if (!from || !to) return
      try {
        const apiKey = import.meta.env.VITE_GEOAPIFY_API_KEY
        if (apiKey) {
          // Geoapify routing
          // IMPORTANT: Geoapify expects lon,lat order in waypoints
          const url = `https://api.geoapify.com/v1/routing?waypoints=${from.lon},${from.lat}|${to.lon},${to.lat}&mode=drive&format=geojson&apiKey=${apiKey}`
          const res = await fetch(url)
          if (!res.ok) throw new Error(`Routing HTTP ${res.status}`)
          const data = await res.json()
          // Validate geometry exists
          const hasFeatures = data && data.features && data.features.length > 0
          if (!hasFeatures) throw new Error('No route geometry')
          const feature = data.features[0]
          if (routeLayer) routeLayer.remove()
          routeLayer = L.geoJSON(feature, {
            style: { color: '#2563eb', weight: 4, opacity: 0.95 },
          }).addTo(map)
          // ensure route is above tiles
          // @ts-ignore
          if ((routeLayer as any).bringToFront) (routeLayer as any).bringToFront()
        } else {
          // No key → draw straight line
          drawStraight()
        }
      } catch (e) {
        // On any error → draw straight line
        drawStraight()
      }
    }

    drawMarkers()
    fitBoundsIfNeeded()
    // Always draw something quickly, then try to replace with routed geometry
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


