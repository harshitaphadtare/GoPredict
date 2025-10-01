import React, { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { MapPin, Search } from "lucide-react";

type Location = {
  id: string;
  name: string;
  lat: number;
  lon: number;
};

interface LocationSearchProps {
  id: string;
  label: string;
  value: string;
  onChange: (val: string) => void;
  onLocationSelect: (location: Location) => void;
  selectedLocation?: Location | null;
  placeholder?: string;
  className?: string;
  city: 'new_york' | 'san_francisco';
}

// Predefined locations for each city
const NY_LOCATIONS: Location[] = [
  { id: "ny_times_square", name: "Times Square, NYC", lat: 40.7580, lon: -73.9855 },
  { id: "ny_central_park", name: "Central Park, NYC", lat: 40.7829, lon: -73.9654 },
  { id: "ny_wall_street", name: "Wall Street, NYC", lat: 40.7074, lon: -74.0113 },
  { id: "ny_brooklyn_bridge", name: "Brooklyn Bridge, NYC", lat: 40.7061, lon: -73.9969 },
  { id: "ny_jfk_airport", name: "JFK Airport, NYC", lat: 40.6413, lon: -73.7781 },
  { id: "ny_laguardia", name: "LaGuardia Airport, NYC", lat: 40.7769, lon: -73.8740 },
  { id: "ny_manhattan", name: "Manhattan, NYC", lat: 40.7831, lon: -73.9712 },
  { id: "ny_brooklyn", name: "Brooklyn, NYC", lat: 40.6782, lon: -73.9442 },
  { id: "ny_queens", name: "Queens, NYC", lat: 40.7282, lon: -73.7949 },
  { id: "ny_bronx", name: "Bronx, NYC", lat: 40.8448, lon: -73.8648 },
];

const SF_LOCATIONS: Location[] = [
  { id: "sf_fishermans_wharf", name: "Fisherman's Wharf, SF", lat: 37.8087, lon: -122.4098 },
  { id: "sf_golden_gate", name: "Golden Gate Bridge, SF", lat: 37.8199, lon: -122.4783 },
  { id: "sf_union_square", name: "Union Square, SF", lat: 37.7880, lon: -122.4074 },
  { id: "sf_chinatown", name: "Chinatown, SF", lat: 37.7941, lon: -122.4078 },
  { id: "sf_castro", name: "Castro District, SF", lat: 37.7609, lon: -122.4350 },
  { id: "sf_mission", name: "Mission District, SF", lat: 37.7599, lon: -122.4148 },
  { id: "sf_soma", name: "SoMa, SF", lat: 37.7749, lon: -122.4194 },
  { id: "sf_marina", name: "Marina District, SF", lat: 37.8024, lon: -122.4368 },
  { id: "sf_sfo_airport", name: "SFO Airport, SF", lat: 37.6213, lon: -122.3790 },
  { id: "sf_oakland_airport", name: "Oakland Airport, SF", lat: 37.7126, lon: -122.2196 },
];

export function LocationSearch({ 
  id, 
  label, 
  value, 
  onChange, 
  onLocationSelect, 
  selectedLocation,
  placeholder = "Search for a location...", 
  className,
  city 
}: LocationSearchProps) {
  const [searchResults, setSearchResults] = useState<Location[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const locations = city === 'new_york' ? NY_LOCATIONS : SF_LOCATIONS;

  // Filter locations based on search input
  useEffect(() => {
    if (value.trim()) {
      const filtered = locations.filter(location =>
        location.name.toLowerCase().includes(value.toLowerCase())
      );
      setSearchResults(filtered);
      setIsOpen(filtered.length > 0);
    } else {
      setSearchResults([]);
      setIsOpen(false);
    }
  }, [value, locations]);

  // Handle location selection
  const handleLocationSelect = (location: Location) => {
    onLocationSelect(location);
    onChange(location.name);
    setIsOpen(false);
    // Blur input to ensure dropdown closes and user doesn't need to click twice
    inputRef.current?.blur();
  };

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    onChange(newValue);
    
    // If user clears the input, clear the selected location
    if (!newValue.trim()) {
      onLocationSelect(null as any);
    }
  };

  // Handle input focus
  const handleInputFocus = () => {
    if (searchResults.length > 0) {
      setIsOpen(true);
    }
  };

  // Handle input blur
  const handleInputBlur = (e: React.FocusEvent) => {
    // Delay closing to allow for click events on dropdown items
    setTimeout(() => {
      if (!dropdownRef.current?.contains(e.relatedTarget as Node)) {
        setIsOpen(false);
      }
    }, 150);
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setIsOpen(false);
      inputRef.current?.blur();
    }
    if (e.key === 'Enter') {
      // Select first result on Enter for faster UX
      if (searchResults.length > 0) {
        handleLocationSelect(searchResults[0]);
        e.preventDefault();
      }
    }
  };

  return (
    <div className={cn("w-full relative", className)}>
      <label htmlFor={id} className="mb-2 block text-sm font-medium text-foreground/80">
        {label}
      </label>
      <div className="relative">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            ref={inputRef}
            id={id}
            type="text"
            value={value}
            onChange={handleInputChange}
            onFocus={handleInputFocus}
            onBlur={handleInputBlur}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="pl-10 pr-10"
            disabled={isSearching}
          />
          {selectedLocation && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-1 top-1/2 h-6 w-6 -translate-y-1/2 p-0"
              onClick={() => {
                onChange('');
                onLocationSelect(null as any);
              }}
            >
              ×
            </Button>
          )}
        </div>

        {/* Dropdown */}
        {isOpen && searchResults.length > 0 && (
          <div
            ref={dropdownRef}
            className="absolute z-50 mt-1 w-full rounded-lg border border-border bg-background shadow-lg"
          >
            <div className="max-h-60 overflow-y-auto">
              {searchResults.map((location) => (
                <button
                  key={location.id}
                  type="button"
                  className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-muted/50 focus:bg-muted/50 focus:outline-none"
                  onClick={() => handleLocationSelect(location)}
                >
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <div className="text-sm font-medium">{location.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* No results message */}
        {isOpen && searchResults.length === 0 && value.trim() && (
          <div className="absolute z-50 mt-1 w-full rounded-lg border border-border bg-background p-4 text-center text-sm text-muted-foreground shadow-lg">
            No locations found. Please try a different search term.
          </div>
        )}
      </div>

      {/* Selected location info */}
      {selectedLocation && (
        <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
          <MapPin className="h-3 w-3" />
          <span>Selected: {selectedLocation.name}</span>
        </div>
      )}
    </div>
  );
}
