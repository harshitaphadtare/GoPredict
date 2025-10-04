import { useState } from "react";
import { LocationSearch } from "@/components/LocationSearch";
import LeafletMap from "@/components/LeafletMap";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { predictTravelTime } from "@/lib/api";
import { Clock, MapPin, Car } from "lucide-react";
import Footer from "@/components/Footer";

type Location = {
  id: string;
  name: string;
  lat: number;
  lon: number;
};

function toRad(d: number) {
  return (d * Math.PI) / 180;
}

function haversineKm(a: Location, b: Location) {
  const R = 6371;
  const dLat = toRad(b.lat - a.lat);
  const dLon = toRad(b.lon - a.lon);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

function estimateMinutes(distanceKm: number, date: Date) {
  const hour = date.getHours();
  const isRush = (hour >= 7 && hour <= 10) || (hour >= 16 && hour <= 19);
  const baseSpeedKmh = isRush ? 18 : 28;
  const safetyFactor = 1.15;
  const minutes = (distanceKm / baseSpeedKmh) * 60 * safetyFactor;
  return Math.max(4, minutes);
}

export default function Home() {
  const [fromId, setFromId] = useState("");
  const [toId, setToId] = useState("");
  const [fromLocation, setFromLocation] = useState<Location | null>(null);
  const [toLocation, setToLocation] = useState<Location | null>(null);
  const [dateStr, setDateStr] = useState("");
  const [predicted, setPredicted] = useState<number | null>(null);
  const [animKey, setAnimKey] = useState(0);
  const [currentCity, setCurrentCity] = useState<'new_york' | 'san_francisco'>('new_york');
  const [isLoading, setIsLoading] = useState(false);

  // Update city when location changes
  const handleFromLocationSelect = (location: Location | null) => {
    setFromLocation(location);
    if (location) {
      const city = location.id.startsWith('ny_') ? 'new_york' : 'san_francisco';
      setCurrentCity(city);
      // Clear destination if it's from a different city
      if (toLocation && toLocation.id.startsWith(city === 'new_york' ? 'sf_' : 'ny_')) {
        setToLocation(null);
        setToId('');
      }
    }
  };

  const handleToLocationSelect = (location: Location | null) => {
    setToLocation(location);
    if (location) {
      const city = location.id.startsWith('ny_') ? 'new_york' : 'san_francisco';
      setCurrentCity(city);
    }
  };

  const canPredict = fromLocation && toLocation && dateStr && fromLocation.id !== toLocation.id;

  const onPredict = async () => {
    if (!fromLocation || !toLocation) return;
    
    const date = new Date(dateStr);
    setIsLoading(true);

    // Validate that both locations are within the same city
    const isFromNY = fromLocation.id.startsWith('ny_');
    const isFromSF = fromLocation.id.startsWith('sf_');
    const isToNY = toLocation.id.startsWith('ny_');
    const isToSF = toLocation.id.startsWith('sf_');

    if ((isFromNY && isToSF) || (isFromSF && isToNY)) {
      alert('Cross-city travel is not supported. Please select locations within the same city (New York or San Francisco)');
      setIsLoading(false);
      return;
    }

    try {
      const response = await predictTravelTime({
        from: fromLocation,
        to: toLocation,
        startTime: date.toISOString(),
        city: currentCity
      });
      
      if (typeof response.minutes === "number" && isFinite(response.minutes)) {
        setPredicted(response.minutes);
        setAnimKey((k) => k + 1);
        setIsLoading(false);
        return;
      }
    } catch (error) {
      console.error('Prediction API error:', error);
    }

    // Fallback calculation
    const km = haversineKm(fromLocation, toLocation);
    const minutes = estimateMinutes(km, date);
    setPredicted(minutes);
    setAnimKey((k) => k + 1);
    setIsLoading(false);
  };

  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background">
      {/* Background Pattern */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 opacity-60"
        style={{ 
          backgroundImage: "url('data:image/svg+xml,%3Csvg width=\"60\" height=\"60\" viewBox=\"0 0 60 60\" xmlns=\"http://www.w3.org/2000/svg\"%3E%3Cg fill=\"none\" fill-rule=\"evenodd\"%3E%3Cg fill=\"%239C92AC\" fill-opacity=\"0.1\"%3E%3Ccircle cx=\"30\" cy=\"30\" r=\"2\"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')",
          backgroundPosition: "center",
          backgroundSize: "60px 60px"
        }}
      />
      <div className="absolute inset-0 -z-10 bg-gradient-to-br from-white/80 via-white/60 to-white/30 dark:from-black/40 dark:via-black/25 dark:to-black/10" />

      {/* Header */}
      <header className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15 text-primary shadow-soft">
            <Car className="h-5 w-5" />
          </div>
          <span className="text-base font-semibold tracking-tight">GoPredict</span>
        </div>
        <div>
          <ThemeToggle />
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto flex-1 px-4 pb-4 pt-2">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {/* Left Column - Results and Map */}
          <div className="flex flex-col gap-4">
            {/* Prediction Result - always visible */}
            <div className="rounded-2xl border border-border bg-card/90 p-6 shadow-soft backdrop-blur">
              <div className="flex items-center gap-3">
                <Clock className="h-6 w-6 text-primary" />
                <div>
                  <h3 className="text-lg font-semibold">Estimated Travel Time</h3>
                  <p className="text-3xl font-bold text-primary">
                    {typeof predicted === "number" && isFinite(predicted) ? `${Math.round(predicted)} minutes` : "-"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {typeof predicted === "number" && isFinite(predicted)
                      ? (() => {
                          const total = Math.round(predicted);
                          const hours = Math.floor(total / 60);
                          const mins = total % 60;
                          // If 60 minutes exactly, roll up to 1h 0m
                          const adjHours = mins === 60 ? hours + 1 : hours;
                          const adjMins = mins === 60 ? 0 : mins;
                          return `${adjHours}h ${adjMins}m`;
                        })()
                      : "--"}
                  </p>
                </div>
              </div>
            </div>

            {/* Map */}
            <LeafletMap
              from={fromLocation}
              to={toLocation}
              animateKey={`${animKey}-${fromLocation?.id}-${toLocation?.id}`}
            />
          </div>

          {/* Right Column - Input Form */}
          <div className="flex flex-col gap-3 rounded-2xl border border-border bg-card/90 p-4 shadow-soft backdrop-blur">
            <h2 className="text-lg font-semibold mb-2">Plan Your Trip</h2>
            
            <LocationSearch
              id="from"
              label="Start Location"
              value={fromId}
              onChange={setFromId}
              onLocationSelect={handleFromLocationSelect}
              selectedLocation={fromLocation}
              city={currentCity}
              placeholder="Search for start location..."
            />
            
            <LocationSearch
              id="to"
              label="End Location"
              value={toId}
              onChange={setToId}
              onLocationSelect={handleToLocationSelect}
              selectedLocation={toLocation}
              city={currentCity}
              placeholder="Search for end location..."
            />
            
            {/* City Switcher */}
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant={currentCity === 'new_york' ? 'default' : 'outline'}
                onClick={() => {
                  setCurrentCity('new_york');
                  setFromLocation(null);
                  setToLocation(null);
                  setFromId('');
                  setToId('');
                }}
              >
                New York City
              </Button>
              <Button
                variant={currentCity === 'san_francisco' ? 'default' : 'outline'}
                onClick={() => {
                  setCurrentCity('san_francisco');
                  setFromLocation(null);
                  setToLocation(null);
                  setFromId('');
                  setToId('');
                }}
              >
                San Francisco
              </Button>
            </div>
            
            <div className="w-full">
              <label htmlFor="start_time" className="mb-2 block text-sm font-medium text-foreground/80">
                Date & Time of Travel
              </label>
              <input
                id="start_time"
                type="datetime-local"
                value={dateStr}
                onChange={(e) => setDateStr(e.target.value)}
                className="w-full rounded-lg border border-border bg-background px-4 py-3 text-foreground shadow-soft outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/30"
              />
            </div>
            
            <Button
              onClick={onPredict}
              disabled={!canPredict || isLoading}
              className="h-12 w-full rounded-lg bg-primary text-primary-foreground shadow-soft transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? "Predicting..." : "Predict Travel Time"}
            </Button>

            {/* City Info */}
            <div className="mt-4 rounded-lg bg-muted/50 p-3">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <MapPin className="h-4 w-4" />
                <span>
                  Currently showing locations for: <strong>{currentCity === 'new_york' ? 'New York City' : 'San Francisco'}</strong>
                </span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}
