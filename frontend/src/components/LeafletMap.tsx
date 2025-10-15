import { useEffect } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'


// Fix for default markers not showing in bundled environments
delete (L.Icon.Default.prototype as any)._getIconUrl

// Create a more beautiful, custom pin-style icon using SVG
const createCustomIcon = (color: string) => {
  // SVG for a classic map pin with enhanced styling
  const markerHtml = `
    <svg viewBox="0 0 32 48" width="32" height="48" style="filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));">
      {/* The main pin shape with a white border for better contrast */}
      <path
        fill="${color}"
        stroke="#FFFFFF"
        stroke-width="2"
        d="M16 2 C9.925 2 5 6.925 5 13 c0 7.875 11 23 11 23 s11 -15.125 11 -23 C27 6.925 22.075 2 16 2z"
      />
      {/* A white inner circle for a polished look */}
      <circle cx="16" cy="13" r="5" fill="#FFFFFF" />
    </svg>`

  return L.divIcon({
    className: 'leaflet-custom-icon',
    html: markerHtml,
    iconSize: [32, 48], // Size of the icon
    iconAnchor: [16, 48], // Point of the icon which will correspond to marker's location
    popupAnchor: [0, -48], // Point from which the popup should open relative to the iconAnchor
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
          icon: createCustomIcon('#2563eb'), // Blue pin for start
        }).addTo(map)
        startMarker.bindPopup(`<b>Start:</b> ${from.name || 'Start Location'}`)
        markers.push(startMarker)
      }
      if (to) {
        const endMarker = L.marker([to.lat, to.lon], {
          icon: createCustomIcon('#ef4444'), // Red pin for end
        }).addTo(map)
        endMarker.bindPopup(`<b>End:</b> ${to.name || 'End Location'}`)
        markers.push(endMarker)
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
    }

    const fetchRoute = async () => {
      if (!from || !to) return
      const apiKey = import.meta.env.VITE_GEOAPIFY_API_KEY
      if (!apiKey) {
        drawStraight()
        return
      }
      try {
         //EXAMPLE:https://api.geoapify.com/v1/routing?waypoints=40.7757145,-73.87336398511545|40.6604335,-73.8302749&mode=drive&apiKey=YOUR_API_KEY
        const url = `https://api.geoapify.com/v1/routing?waypoints=${from.lat},${from.lon}|${to.lat},${to.lon}&mode=drive&format=geojson&apiKey=${apiKey}`
        console.log(url);
        const res = await fetch(url)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        console.log('Geoapify route data:', data);
        if (!data?.features?.[0]) throw new Error('No route')
        if (routeLayer) routeLayer.remove()
        routeLayer = L.geoJSON(data.features[0], {
          style: { color: '#2563eb', weight: 4, opacity: 0.95 },
        }).addTo(map)
      } catch {
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