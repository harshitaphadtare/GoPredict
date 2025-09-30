import React, { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export type GeoPoint = {
  name?: string;
  lat: number;
  lon: number;
};

interface GoogleMapProps {
  from?: GeoPoint | null;
  to?: GeoPoint | null;
  /** Changing this value restarts the animation */
  animateKey?: string | number;
  className?: string;
}

declare global {
  interface Window {
    google: any;
    initMap: () => void;
  }
}

export function GoogleMap({ from, to, animateKey, className }: GoogleMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const directionsServiceRef = useRef<any>(null);
  const directionsRendererRef = useRef<any>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load Google Maps API
  useEffect(() => {
    if (window.google) {
      setIsLoaded(true);
      return;
    }

    const script = document.createElement('script');
    script.src = `https://maps.googleapis.com/maps/api/js?key=${import.meta.env.VITE_GOOGLE_MAPS_API_KEY}&libraries=geometry&callback=initMap`;
    script.async = true;
    script.defer = true;
    
    window.initMap = () => {
      setIsLoaded(true);
    };

    script.onerror = () => {
      setError('Failed to load Google Maps');
    };

    document.head.appendChild(script);

    return () => {
      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }
    };
  }, []);

  // Initialize map
  useEffect(() => {
    if (!isLoaded || !mapRef.current || !window.google) return;

    try {
      const map = new window.google.maps.Map(mapRef.current, {
        zoom: 12,
        center: from ? { lat: from.lat, lng: from.lon } : { lat: 40.7128, lng: -74.0060 },
        mapTypeId: window.google.maps.MapTypeId.ROADMAP,
        styles: [
          {
            featureType: "poi",
            elementType: "labels",
            stylers: [{ visibility: "off" }]
          }
        ]
      });

      mapInstanceRef.current = map;
      directionsServiceRef.current = new window.google.maps.DirectionsService();
      directionsRendererRef.current = new window.google.maps.DirectionsRenderer({
        suppressMarkers: false,
        polylineOptions: {
          strokeColor: '#3b82f6',
          strokeWeight: 4,
          strokeOpacity: 0.8
        }
      });
      directionsRendererRef.current.setMap(map);
    } catch (err) {
      console.error('Error initializing map:', err);
      setError('Failed to initialize map');
    }
  }, [isLoaded, from]);

  // Update route when locations change
  useEffect(() => {
    if (!isLoaded || !directionsServiceRef.current || !directionsRendererRef.current || !from || !to) {
      return;
    }

    const request = {
      origin: { lat: from.lat, lng: from.lon },
      destination: { lat: to.lat, lng: to.lon },
      travelMode: window.google.maps.TravelMode.DRIVING,
    };

    directionsServiceRef.current.route(request, (result: any, status: any) => {
      if (status === window.google.maps.DirectionsStatus.OK) {
        directionsRendererRef.current.setDirections(result);
        
        // Add custom markers
        const startMarker = new window.google.maps.Marker({
          position: { lat: from.lat, lng: from.lon },
          map: mapInstanceRef.current,
          title: from.name || 'Start',
          icon: {
            path: window.google.maps.SymbolPath.CIRCLE,
            scale: 8,
            fillColor: '#10b981',
            fillOpacity: 1,
            strokeColor: '#ffffff',
            strokeWeight: 2
          }
        });

        const endMarker = new window.google.maps.Marker({
          position: { lat: to.lat, lng: to.lon },
          map: mapInstanceRef.current,
          title: to.name || 'End',
          icon: {
            path: window.google.maps.SymbolPath.CIRCLE,
            scale: 8,
            fillColor: '#ef4444',
            fillOpacity: 1,
            strokeColor: '#ffffff',
            strokeWeight: 2
          }
        });

        // Add car marker at start position
        const carMarker = new window.google.maps.Marker({
          position: { lat: from.lat, lng: from.lon },
          map: mapInstanceRef.current,
          title: 'Your car',
          icon: {
            path: 'M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L15 1H5C3.89 1 3 1.89 3 3V21C3 22.11 3.89 23 5 23H19C20.11 23 21 22.11 21 21V9M19 9H14V4H5V21H19V9Z',
            scale: 1.5,
            fillColor: '#3b82f6',
            fillOpacity: 1,
            strokeColor: '#ffffff',
            strokeWeight: 2,
            anchor: new window.google.maps.Point(12, 12)
          }
        });

        // Animate car along route
        if (animateKey && result.routes && result.routes[0]) {
          const route = result.routes[0];
          const path = route.overview_path;
          let currentStep = 0;
          const totalSteps = path.length;
          const animationDuration = 4000; // 4 seconds
          const stepDuration = animationDuration / totalSteps;

          const animateCar = () => {
            if (currentStep < totalSteps) {
              const point = path[currentStep];
              carMarker.setPosition(point);
              currentStep++;
              setTimeout(animateCar, stepDuration);
            }
          };

          setTimeout(animateCar, 100);
        }
      } else {
        console.error('Directions request failed:', status);
        setError('Failed to get directions');
      }
    });
  }, [from, to, animateKey, isLoaded]);

  if (error) {
    return (
      <div className={cn("w-full overflow-hidden rounded-2xl border border-border bg-card/90 shadow-soft backdrop-blur", className)}>
        <div className="flex items-center justify-between border-b border-border px-4 py-3 text-sm text-foreground/70">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-2 w-2 rounded-full bg-red-500" />
            <span>Map Error</span>
          </div>
        </div>
        <div className="flex h-64 w-full items-center justify-center bg-muted/50">
          <div className="text-center">
            <p className="text-sm text-muted-foreground">{error}</p>
            <p className="text-xs text-muted-foreground mt-1">Please check your Google Maps API key</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("w-full overflow-hidden rounded-2xl border border-border bg-card/90 shadow-soft backdrop-blur", className)}>
      <div className="flex items-center justify-between border-b border-border px-4 py-3 text-sm text-foreground/70">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-2 rounded-full bg-primary" />
          <span>Route Preview</span>
        </div>
        {from && to ? (
          <span className="truncate">{from.name ?? "Start"} → {to.name ?? "End"}</span>
        ) : (
          <span className="truncate">Select locations to preview</span>
        )}
      </div>

      <div className="h-64 w-full">
        <div 
          ref={mapRef} 
          className="h-full w-full"
          style={{ minHeight: '300px' }}
        />
      </div>
    </div>
  );
}
