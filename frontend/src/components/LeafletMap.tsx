import { useEffect, useState, useRef, useCallback } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import {Loader2 } from 'lucide-react'
import { RouteStep,LeafletMapProps} from '@/types/LeafletMaps';


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

export default function LeafletMap({ from, to, animateKey, isPredicting }: LeafletMapProps) {
  const [isRouteLoading, setIsRouteLoading] = useState(false)
  const [routeSteps, setRouteSteps] = useState<RouteStep[]>([])
  const [routeError, setRouteError] = useState<string | null>(null);

  // Refs to hold Leaflet instances, preventing re-initialization on re-renders
  const mapRef = useRef<L.Map | null>(null);
  const routeLayerRef = useRef<L.Polyline | L.GeoJSON | null>(null);
  const markersRef = useRef<L.Marker[]>([]);
  const turnMarkersRef = useRef<L.Marker[]>([]);

  // --- Map Drawing and Animation Functions (SRP) ---

  const clearTurnMarkers = useCallback(() => {
    turnMarkersRef.current.forEach(m => m.remove());
    turnMarkersRef.current = [];
  }, []);

  const animateRoute = useCallback((geoJsonData: any) => {
    const map = mapRef.current;
    if (!map) return;

    if (routeLayerRef.current) routeLayerRef.current.remove();
    clearTurnMarkers();

    const allCoords = geoJsonData.geometry.coordinates.flat(1).map((c: number[]) => L.latLng(c[1], c[0]));
    const animatedPolyline = L.polyline([], { color: '#2563eb', weight: 4, opacity: 0.95 }).addTo(map);
    routeLayerRef.current = animatedPolyline;

    const steps = geoJsonData.properties?.legs?.[0]?.steps;
    const turnPoints = (steps && steps.length > 1)
      ? steps.slice(1).map((step: any) => ({
          index: step.from_index,
          latlng: allCoords[step.from_index],
        })).filter((turn: any) => turn.latlng)
      : [];

    let nextTurnIndex = 0;
    const turnIcon = createTurnIcon();
    const animationDuration = 750;
    let startTime: number | null = null;

    const step = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / animationDuration, 1);
      const pointsToShow = Math.floor(progress * allCoords.length);

      if (pointsToShow > animatedPolyline.getLatLngs().length) {
        animatedPolyline.setLatLngs(allCoords.slice(0, pointsToShow));

        while (nextTurnIndex < turnPoints.length && turnPoints[nextTurnIndex].index <= pointsToShow) {
          const turn = turnPoints[nextTurnIndex];
          turnMarkersRef.current.push(L.marker(turn.latlng, { icon: turnIcon }).addTo(map));
          nextTurnIndex++;
        }
      }

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        animatedPolyline.setLatLngs(allCoords);
        const properties = geoJsonData.properties;
        if (properties) {
          const distanceKm = (properties.distance / 1000).toFixed(1);
          const timeMinutes = Math.round(properties.time / 60);
          animatedPolyline.bindPopup(`<b>Route Details</b><br>Distance: ${distanceKm} km<br>Est. Time: ${timeMinutes} minutes`);
        }
      }
    };
    requestAnimationFrame(step);
  }, [clearTurnMarkers]);

  const fetchRoute = useCallback(async () => {
    if (!from || !to) return;

    const apiKey = import.meta.env.VITE_GEOAPIFY_API_KEY;
    if (!apiKey) {
      // Fallback to straight line if no API key
      if (mapRef.current) { // Guard against mapRef.current being null
        if (routeLayerRef.current) routeLayerRef.current.remove();
        routeLayerRef.current = L.polyline([[from.lat, from.lon], [to.lat, to.lon]], { color: '#2563eb', weight: 3, opacity: 0.85 }).addTo(mapRef.current);
      }
      return;
    }

    setIsRouteLoading(true);
    clearTurnMarkers();
    setRouteError(null);
    setRouteSteps([]);
    if (routeLayerRef.current) routeLayerRef.current.remove();

    try {
      const primaryRouteResponse = await fetch(
        `https://api.geoapify.com/v1/routing?waypoints=${from.lat},${from.lon}|${to.lat},${to.lon}&mode=drive&format=geojson&waypoints.snapped=true&apiKey=${apiKey}`
      );
      let routeData = await primaryRouteResponse.json();

      if (routeData.statusCode === 400) {
        console.warn("Initial routing failed. Attempting fallback with reverse geocoding.");
        const fromPromise = fetch(`https://api.geoapify.com/v1/geocode/reverse?lat=${from.lat}&lon=${from.lon}&apiKey=${apiKey}`).then(res => res.json());
        const toPromise = fetch(`https://api.geoapify.com/v1/geocode/reverse?lat=${to.lat}&lon=${to.lon}&apiKey=${apiKey}`).then(res => res.json());
        const [fromRev, toRev] = await Promise.all([fromPromise, toPromise]);

        const correctedFrom = fromRev?.features?.[0]?.properties;
        const correctedTo = toRev?.features?.[0]?.properties;

        if (!correctedFrom || !correctedTo) throw new Error("Reverse geocoding failed.");

        const fallbackRouteResponse = await fetch(
          `https://api.geoapify.com/v1/routing?waypoints=${correctedFrom.lat},${correctedFrom.lon}|${correctedTo.lat},${correctedTo.lon}&mode=drive&format=geojson&apiKey=${apiKey}`
        );
        if (!fallbackRouteResponse.ok) throw new Error(`Fallback routing failed.`);
        routeData = await fallbackRouteResponse.json();
      }

      if (!routeData?.features?.[0]) throw new Error("No route feature found.");

      animateRoute(routeData.features[0]);
      setRouteSteps(routeData.features[0]?.properties?.legs?.[0]?.steps || []);
    } catch (error) {
      if (error instanceof Error) {
        console.error("Final routing error:", error.message);
        setRouteError(error.message);
      }
    } finally {
      setIsRouteLoading(false);
    }
  }, [from, to, animateRoute, clearTurnMarkers]);

  // --- useEffect Hooks for Lifecycle Management ---

  // Effect for map initialization (runs once)
  useEffect(() => {
    if (mapRef.current) return; // Initialize map only once

    mapRef.current = L.map('route-map', { zoomControl: true });
    const apiKey = import.meta.env.VITE_GEOAPIFY_API_KEY;
    const tileUrl = apiKey
      ? `https://maps.geoapify.com/v1/tile/osm-bright/{z}/{x}/{y}.png?apiKey=${apiKey}`
      : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

    L.tileLayer(tileUrl, {
      maxZoom: 19,
      attribution: apiKey ? '© OpenMapTiles © OpenStreetMap contributors | © Geoapify' : '© OpenStreetMap contributors',
    }).addTo(mapRef.current);

    // Cleanup function to remove map on component unmount
    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

  // Effect for updating markers and view when 'from' or 'to' change
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Draw start/end markers
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];
    if (from) {
      const startMarker = L.marker([from.lat, from.lon], { icon: createCustomIcon('#2563eb') }).addTo(map);
      startMarker.bindPopup(`<b>Start:</b> ${from.name || 'Start Location'}`);
      markersRef.current.push(startMarker);
    }
    if (to) {
      const endMarker = L.marker([to.lat, to.lon], { icon: createCustomIcon('#ef4444') }).addTo(map);
      endMarker.bindPopup(`<b>End:</b> ${to.name || 'End Location'}`);
      markersRef.current.push(endMarker);
    }

    // Adjust map view
    const points: L.LatLngExpression[] = [];
    if (from) points.push([from.lat, from.lon]);
    if (to) points.push([to.lat, to.lon]);

    if (points.length > 0) {
      map.fitBounds(L.latLngBounds(points).pad(0.25));
    } else {
      map.setView([40.7128, -74.006], 11); // Default view
    }
  }, [from, to]);

  // Effect for fetching and drawing the route
  useEffect(() => {
    if (from && to) {
      fetchRoute();
    } else {
      // Clear route if 'from' or 'to' is missing
      if (routeLayerRef.current) routeLayerRef.current.remove();
      clearTurnMarkers();
    }
  }, [from, to, animateKey, fetchRoute, clearTurnMarkers]);

  return (
    <div className="flex flex-col w-full overflow-hidden rounded-2xl border border-border bg-card/90 shadow-soft backdrop-blur">
      <div className="flex items-center justify-between border-b border-border px-4 py-3 text-sm text-foreground/70">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-2 rounded-full bg-primary" />
          {isPredicting ? (
            <span>Calculating route...</span>
          ) : (
            routeError ? 
            <span className="text-red-500">Route not found</span> :
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