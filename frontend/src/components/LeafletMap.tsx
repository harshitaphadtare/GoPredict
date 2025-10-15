import { useEffect, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { List, Loader2 } from 'lucide-react'


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

interface RouteStep {
  instruction: {
    text: string
  }
}

export default function LeafletMap({ from, to, animateKey }: LeafletMapProps) {
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

    const animateRoute = (geoJsonData: any) => {
      if (routeLayer) routeLayer.remove();

      const allCoords = geoJsonData.geometry.coordinates.flat(1).map((c: number[]) => L.latLng(c[1], c[0]));
      const animatedPolyline = L.polyline([], { color: '#2563eb', weight: 4, opacity: 0.95 }).addTo(map);
      routeLayer = animatedPolyline;

      let i = 0;
      const step = () => {
        if (i < allCoords.length) {
          animatedPolyline.addLatLng(allCoords[i]);
          i++;
          requestAnimationFrame(step);
        } else {
          // Animation finished, bind the popup
          const properties = geoJsonData.properties;
          if (properties) {
            const distanceKm = (properties.distance / 1000).toFixed(1);
            const timeMinutes = Math.round(properties.time / 60);
            animatedPolyline.bindPopup(
              `<b>Route Details</b><br>Distance: ${distanceKm} km<br>Est. Time: ${timeMinutes} minutes`
            );
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
      drawStraight()
      fetchRoute()
    }

    return () => {
      map.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [from?.lat, from?.lon, to?.lat, to?.lon, animateKey])

  return (
    <div className="flex flex-col w-full overflow-hidden rounded-2xl border border-border bg-card/90 shadow-soft backdrop-blur">
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
      <div className="relative">
        <div id="route-map" style={{ height: 360, width: '100%' }} />
        {isRouteLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/20 backdrop-blur-sm">
            <Loader2 className="h-10 w-10 animate-spin text-white" />
          </div>
        )}
      </div>
      {routeSteps.length > 0 && (
        <div className="border-t border-border">
          <div className="flex items-center gap-2 px-4 py-2 text-sm font-medium">
            <List className="h-4 w-4" />
            <span>Turn-by-Turn Directions</span>
          </div>
          <ol className="max-h-48 overflow-y-auto list-decimal list-inside bg-background/50 px-4 pb-3 text-sm">
            {routeSteps.map((step, index) => (
              <li key={index} className="py-1.5 border-b border-border/50 last:border-b-0">
                {step.instruction.text}
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}