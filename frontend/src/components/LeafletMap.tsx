import { useEffect, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import {Loader2 } from 'lucide-react'


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

// Create a small white dot icon for turn-by-turn markers
const createTurnIcon = () => {
  const markerHtml = `
    <svg viewBox="0 0 12 12" width="12" height="12" style="filter: drop-shadow(0 1px 2px rgba(0,0,0,0.4));">
      <circle cx="6" cy="6" r="4" fill="#FFFFFF" stroke="#333333" stroke-width="1.5" />
    </svg>`

  return L.divIcon({
    className: 'leaflet-turn-icon',
    html: markerHtml,
    iconSize: [12, 12],
    iconAnchor: [6, 6],
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
  isPredicting?: boolean
}

interface RouteStep {
  instruction: {
    text: string
  }
}

export default function LeafletMap({ from, to, animateKey, isPredicting }: LeafletMapProps) {
  const [isRouteLoading, setIsRouteLoading] = useState(false)
  const [routeSteps, setRouteSteps] = useState<RouteStep[]>([])

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
    let turnMarkers: L.Marker[] = []

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

    const clearTurnMarkers = () => {
      turnMarkers.forEach(m => m.remove())
      turnMarkers = []
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

    const animateRoute = (geoJsonData: any) => {
      if (routeLayer) routeLayer.remove();
      clearTurnMarkers();

      const allCoords = geoJsonData.geometry.coordinates.flat(1).map((c: number[]) => L.latLng(c[1], c[0]));
      const animatedPolyline = L.polyline([], { color: '#2563eb', weight: 4, opacity: 0.95 }).addTo(map);
      routeLayer = animatedPolyline;

      const animationDuration = 750; // Animate over 750ms
      let startTime: number | null = null;

      const step = (timestamp: number) => {
        if (!startTime) {
          startTime = timestamp;
        }

        const progress = Math.min((timestamp - startTime) / animationDuration, 1);
        const pointsToShow = Math.floor(progress * allCoords.length);

        // Only update if there are new points to show to avoid unnecessary re-renders
        if (pointsToShow > animatedPolyline.getLatLngs().length) {
          animatedPolyline.setLatLngs(allCoords.slice(0, pointsToShow));
        }

        if (progress < 1) {
          requestAnimationFrame(step);
        } else {
          animatedPolyline.setLatLngs(allCoords); // Ensure the full route is drawn
          const properties = geoJsonData.properties;
          if (properties) {
            const distanceKm = (properties.distance / 1000).toFixed(1);
            const timeMinutes = Math.round(properties.time / 60);
            animatedPolyline.bindPopup(`<b>Route Details</b><br>Distance: ${distanceKm} km<br>Est. Time: ${timeMinutes} minutes`);
          }

          // Draw turn markers after animation is complete
          const steps = geoJsonData.properties?.legs?.[0]?.steps;
          if (steps && steps.length > 1) {
            const turnIcon = createTurnIcon();
            // Start from the second step to get the first turn coordinate
            for (let i = 1; i < steps.length; i++) {
              const turnIndex = steps[i].from_index;
              if (turnIndex < allCoords.length) {
                const turnMarker = L.marker(allCoords[turnIndex], { icon: turnIcon }).addTo(map);
                turnMarkers.push(turnMarker);
              }
            }
          }
        }
      };
      requestAnimationFrame(step);
    };

    const fetchRoute = async () => {
      if (!from || !to) return
      const apiKey = import.meta.env.VITE_GEOAPIFY_API_KEY
      if (!apiKey) {
        drawStraight()
        return
      }
      setIsRouteLoading(true)
      clearTurnMarkers()
      setRouteSteps([]) // Clear previous steps
      try {
        //EXAMPLE:https://api.geoapify.com/v1/routing?waypoints=40.7757145,-73.87336398511545|40.6604335,-73.8302749&mode=drive&apiKey=YOUR_API_KEY
        const url = `https://api.geoapify.com/v1/routing?waypoints=${from.lat},${from.lon}|${to.lat},${to.lon}&mode=drive&format=geojson&apiKey=${apiKey}`
        console.log(url);
        const res = await fetch(url)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        console.log('Geoapify route data:', data);
        if (!data?.features?.[0]) throw new Error('No route')

        animateRoute(data.features[0]);

        // Extract and set turn-by-turn instructions
        if (data.features[0]?.properties?.legs?.[0]?.steps) {
          setRouteSteps(data.features[0].properties.legs[0].steps)
        }
      } catch {
        drawStraight()
      } finally {
        setIsRouteLoading(false)
      }
    }

    drawMarkers()
    fitBoundsIfNeeded()
    // Always draw something quickly, then try to replace with routed geometry
    if (from && to) {
      fetchRoute()
    }

    return () => {
      clearTurnMarkers()
      map.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [from?.lat, from?.lon, to?.lat, to?.lon, animateKey])

  return (
    <div className="flex flex-col w-full overflow-hidden rounded-2xl border border-border bg-card/90 shadow-soft backdrop-blur">
      <div className="flex items-center justify-between border-b border-border px-4 py-3 text-sm text-foreground/70">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-2 rounded-full bg-primary" />
          {isPredicting ? (
            <span>Calculating route...</span>
          ) : (
            <span>Route Preview</span>
          )}
        </div>
        {from && to ? (
          <span className="truncate">{from.name ?? 'Start'} → {to.name ?? 'End'}</span>
        ) : (
          <span className="truncate">Select locations to preview</span>
        )}
      </div>
      <div className="relative">
        <div id="route-map" style={{ height: 360, width: '100%' }} />
        {isRouteLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/20 backdrop-blur-sm">
            <Loader2 className="h-10 w-10 animate-spin text-white" />
          </div>
        )}
      </div>

    </div>
  )
}