import { useState, useEffect, useRef } from "react";
import { LocationSearch } from "../components/LocationSearch"; // Relative path
import LeafletMap from "../components/LeafletMap"; // Relative path
import { DateTimePicker } from "../components/DateTimePicker"; // Relative path
import { Button } from "../components/ui/button"; // Relative path
import { ThemeToggle } from "../components/ui/theme-toggle"; // Relative path
import { predictTravelTime } from "../lib/api"; // Relative path
import { Clock, MapPin, Car } from "lucide-react";
import Footer from "../components/Footer"; // Relative path
import { motion, AnimatePresence } from "framer-motion";
// --- 1. Import Link, auth hook, and NEW UserNav ---
import { Link } from "react-router-dom";
import { useAuth } from "../AuthContext"; // Relative path
import { UserNav } from "../components/UserNav"; // Relative path
// --- Removed auth, signOut, useNavigate ---

type Location = {
  id: string;
  name: string;
  lat: number;
  lon: number;
};

export default function Home() {
  // --- 2. Get auth state ---
  const { user } = useAuth(); // Get the current user
  // --- Removed navigate ---

  // --- (All your existing state) ---
  const [fromId, setFromId] = useState("");
  const [toId, setToId] = useState("");
  const [fromLocation, setFromLocation] = useState<Location | null>(null);
  const [toLocation, setToLocation] = useState<Location | null>(null);
  const [travelDate, setTravelDate] = useState<Date | null>(new Date());
  const [predicted, setPredicted] = useState<number | null>(null);
  const [animKey, setAnimKey] = useState(0);
  const [currentCity, setCurrentCity] = useState<"new_york" | "san_francisco">(
    "new_york"
  );
  const [isLoading, setIsLoading] = useState(false);
  const [warning, setWarning] = useState("");

  // --- (handleSignOut removed, it's in UserNav now) ---

  // --- (All your existing functions: handleFromLocationSelect, handleToLocationSelect, canPredict, onPredict) ---
  const handleFromLocationSelect = (location: Location | null) => {
    setFromLocation(location);
    if (location) {
      const city = location.id.startsWith("ny_") ? "new_york" : "san_francisco";
      setCurrentCity(city);
      if (
        toLocation &&
        toLocation.id.startsWith(city === "new_york" ? "sf_" : "ny_")
      ) {
        setToLocation(null);
        setToId("");
      }
    }
  };

  const handleToLocationSelect = (location: Location | null) => {
    setToLocation(location);
    if (location) {
      const city = location.id.startsWith("ny_") ? "new_york" : "san_francisco";
      setCurrentCity(city);
    }
    if (fromLocation && location?.id === fromLocation.id) {
      setWarning(
        "Start and End locations cannot be the same. Please select different locations."
      );
    } else {
      setWarning("");
    }
  };

  const canPredict = Boolean(
    fromLocation && toLocation && travelDate && fromLocation?.id !== toLocation?.id
  );

  const onPredict = async () => {
    if (!fromLocation || !toLocation || !travelDate) return;

    if (fromLocation.id === toLocation.id) {
      setWarning(
        "Start and End locations cannot be the same. Please select different locations."
      );
      return;
    } else {
      setWarning("");
    }

    setIsLoading(true);
    const isMobile = window.innerWidth <= 768;

    const isFromNY = fromLocation.id.startsWith("ny_");
    const isFromSF = fromLocation.id.startsWith("sf_");
    const isToNY = toLocation.id.startsWith("ny_");
    const isToSF = toLocation.id.startsWith("sf_");

    if ((isFromNY && isToSF) || (isFromSF && isToNY)) {
      alert(
        "Cross-city travel is not supported. Please select locations within the same city (New York or San Francisco)"
      );
      setIsLoading(false);
      return;
    }

    try {
      const response = await predictTravelTime({
        from: fromLocation,
        to: toLocation,
        startTime: travelDate.toISOString(),
        city: currentCity,
      });

      if (typeof response.minutes === "number" && isFinite(response.minutes)) {
        setPredicted(response.minutes);
        if (isMobile) {
          setTimeout(() => setAnimKey((k) => k + 1), 900);
        } else {
          setAnimKey((k) => k + 1);
        }
        setIsLoading(false);
        return;
      } else {
        // Handle cases where response.minutes is not a valid number
        console.error("Invalid prediction value received:", response.minutes);
        alert("Received an invalid prediction value. Please try again.");
      }
    } catch (error) {
      console.error("Prediction API error:", error);
      alert("Unable to predict travel time. Please try again later.");
    }
    
    setIsLoading(false); // Ensure loading is set to false even if prediction fails
  };


  const resultRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (animKey > 0 && window.innerWidth <= 768) {
      setTimeout(() => {
        resultRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 150);
    }
  }, [animKey]);

  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background">
      {/* Background Pattern */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.6 }}
        transition={{ duration: 0.8 }}
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 opacity-60"
        style={{
          backgroundImage:
            'url(\'data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%239C92AC" fill-opacity="0.1"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\')',
          backgroundPosition: "center",
          backgroundSize: "60px 60px",
        }}
      />
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="absolute inset-0 -z-10 bg-gradient-to-br from-white/80 via-white/60 to-white/30 dark:from-black/40 dark:via-black/25 dark:to-black/10"
      />

      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        ref={resultRef}
        id="prediction-result"
        className="container mx-auto flex h-16 items-center justify-between px-4"
      >
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="flex items-center gap-3"
        >
          {/* Make logo link to home */}
          <Link to="/" className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15 text-primary shadow-soft">
              <Car className="h-5 w-5" />
            </div>
            <span className="text-base font-semibold tracking-tight">
              GoPredict
            </span>
          </Link>
        </motion.div>
        
        {/* --- Section for conditional auth button --- */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="flex items-center gap-2" // Kept flex and gap
        >
          {/* Conditionally show UserNav or Sign In */}
          {user ? (
            <UserNav />
          ) : (
            <Link to="/sign-in">
              <Button variant="outline">Sign In</Button>
            </Link>
          )}
          {/* End of change */}
          
          <ThemeToggle />
        </motion.div>
        {/* End of modification */}
        
      </motion.header>

      {/* Main Content */}
      <main className="container mx-auto flex-1 px-4 pb-4 pt-2">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="grid grid-cols-1 gap-6 md:grid-cols-2"
        >
          {/* Left Column - Results and Map */}
          <div className="flex flex-col gap-4">
            {/* Prediction Result - always visible */}
            <AnimatePresence mode="wait">
              <motion.div
                key={animKey}
                initial={{
                  opacity: 0,
                  scale: 0.95,
                  y: 20,
                }}
                animate={{
                  opacity: 1,
                  scale: 1,
                  y: 0,
                }}
                transition={{
                  duration: 0.5,
                  ease: [0.34, 1.56, 0.64, 1],
                  scale: {
                    type: "spring",
                    damping: 15,
                    stiffness: 200,
                  },
                }}
                className="rounded-2xl border border-border bg-card/90 p-6 shadow-soft backdrop-blur"
              >
                <motion.div
                  className="absolute inset-0 rounded-2xl"
                  initial={{ opacity: 0 }}
                  animate={{
                    opacity: [0, 0.3, 0],
                    boxShadow: [
                      "0 0 0px 0px rgba(59, 130, 246, 0)",
                      "0 0 30px 5px rgba(59, 130, 246, 0.2)",
                      "0 0 0px 0px rgba(59, 130, 246, 0)",
                    ],
                  }}
                  transition={{
                    duration: 0.8,
                    ease: "easeOut",
                  }}
                />
                <div className="relative flex items-center gap-3">
                  <motion.div
                    initial={{ rotate: -180, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    transition={{
                      duration: 0.6,
                      ease: "easeOut",
                      delay: 0.1,
                    }}
                  >
                    <Clock className="h-6 w-6 text-primary" />
                  </motion.div>
                  <div>
                    <motion.h3
                      className="text-lg font-semibold"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.2, duration: 0.4 }}
                    >
                      Estimated Travel Time
                    </motion.h3>
                    <motion.p
                      className="text-3xl font-bold text-primary"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{
                        delay: 0.3,
                        duration: 0.5,
                        type: "spring",
                        stiffness: 150,
                      }}
                    >
                      {typeof predicted === "number" && isFinite(predicted)
                        ? `${Math.round(predicted)} minutes`
                        : "-"}
                    </motion.p>
                    <motion.p
                      className="text-sm text-muted-foreground"
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{
                        delay: 0.4,
                        duration: 0.4,
                      }}
                    >
                      {typeof predicted === "number" && isFinite(predicted)
                        ? (() => {
                            const total = Math.round(predicted);
                            const hours = Math.floor(total / 60);
                            const mins = total % 60;
                            const adjHours = mins === 60 ? hours + 1 : hours;
                            const adjMins = mins === 60 ? 0 : mins;
                            return `${adjHours}h ${adjMins}m`;
                          })()
                        : "--"}
                    </motion.p>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>

            {/* Map */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <LeafletMap
                from={fromLocation}
                to={toLocation}
                animateKey={`${animKey}-${fromLocation?.id}-${toLocation?.id}`}
              />
            </motion.div>
          </div>

          {/* Right Column - Input Form */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-col gap-3 rounded-2xl border border-border bg-card/90 p-4 shadow-soft backdrop-blur"
          >
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
            {/* Warning Message */}
            {warning && (
              <div
                className="mb-2 flex items-center gap-2 rounded border px-3 py-2 text-sm font-medium
                bg-red-100 text-red-700 border-red-300
                dark:bg-red-900/30 dark:text-red-300 dark:border-red-800"
              >
                <span className="h-4 w-4" /> {/* Consider adding an icon here */}
                {warning}
              </div>
            )}
            
            {/* City Switcher */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.5 }}
              className="grid grid-cols-2 gap-2"
            >
              <Button
                variant={currentCity === 'new_york' ? 'default' : 'outline'}
                onClick={() => {
                  setCurrentCity("new_york");
                  setFromLocation(null);
                  setToLocation(null);
                  setFromId('');
                  setToId('');
                  setPredicted(null); // Clear prediction when city changes
                  setAnimKey(0);
                }}
              >
                New York City
              </Button>
              <Button
                variant={currentCity === 'san_francisco' ? 'default' : 'outline'}
                onClick={() => {
                  setCurrentCity("san_francisco");
                  setFromLocation(null);
                  setToLocation(null);
                  setFromId('');
                  setToId('');
                  setPredicted(null); // Clear prediction when city changes
                  setAnimKey(0);
                }}
              >
                San Francisco
              </Button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.6 }}
              className="w-full overflow-hidden"
            >
              <label className="mb-2 block text-sm font-medium text-foreground/80">
                Date & Time of Travel
              </label>
              <DateTimePicker
                selected={travelDate}
                onChange={(date) => setTravelDate(date)}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.7 }}
            >
              <Button
                onClick={onPredict}
                disabled={!canPredict || isLoading}
                className="h-12 w-full rounded-lg bg-primary text-primary-foreground shadow-soft transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isLoading ? "Predicting..." : "Predict Travel Time"}
              </Button>
            </motion.div>

            {/* City Info */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.8 }}
              className="mt-4 rounded-lg bg-muted/50 p-3"
            >
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <MapPin className="h-4 w-4" />
                <span>
                  Currently showing locations for:{" "}
                  <strong>
                    {currentCity === "new_york"
                      ? "New York City"
                      : "San Francisco"}
                  </strong>
                </span>
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      </main>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <Footer />
      </motion.div>
    </div>
  );
}
