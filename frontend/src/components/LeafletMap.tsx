import React, { useEffect } from 'react'
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
        const url = `https://api.geoapify.com/v1/routing?waypoints=${from.lon},${from.lat}|${to.lon},${to.lat}&mode=drive&format=geojson&apiKey=${apiKey}`
        const res = await fetch(url)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        
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


