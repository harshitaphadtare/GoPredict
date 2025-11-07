import React, { useState, useEffect, useCallback, useMemo } from 'react';
// --- Corrected Relative Paths ---
// From src/pages/Profile.tsx to src/components/ui/* needs ../../
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { ThemeToggle } from '../components/ui/theme-toggle';
// From src/pages/Profile.tsx to src/components/* needs ../../
import { UserNav } from '../components/UserNav';
// From src/pages/Profile.tsx to src/* needs ../
import { auth } from '../firebase';
import { useAuth } from '../AuthContext';
// --- End Corrected Paths ---
import { signOut, updateProfile } from 'firebase/auth';
import { motion } from 'framer-motion';
import { Car, Loader2, Edit, X, ArrowUp, ArrowDown,Trash2, AlertTriangle, ArrowUpDown } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { format, isValid } from 'date-fns'; // Make sure date-fns is installed: pnpm add date-fns

// --- Interfaces ---
interface UserProfile {
  firebaseUid: string;
  email: string;
  displayName: string;
  phone: string;
  location: string;
}
interface Trip {
  _id: string; // MongoDB ObjectId
  firebaseUid: string;
  timestamp: Date;
  travelDateTime: Date; // Ensure this is a Date object after fetching
  startLocation: { name: string; lat: number; lon: number };
  endLocation: { name: string; lat: number; lon: number };
  city: 'new_york' | 'san_francisco';
  predictedDuration: number; // in minutes
}

// --- Placeholder API Functions ---
const fetchProfile = async (userId: string): Promise<UserProfile | null> => {
  console.log("Placeholder: Fetching profile for user:", userId);
  await new Promise(resolve => setTimeout(resolve, 600));
  return { firebaseUid: userId, email: auth.currentUser?.email || "mock@example.com", displayName: auth.currentUser?.displayName || "Mock User", phone: "123-456-7890", location: "San Francisco, CA", };
};
const saveProfile = async (userId: string, data: Partial<Pick<UserProfile, 'displayName' | 'phone' | 'location'>>): Promise<void> => {
  console.log("Placeholder: Saving profile for user:", userId, data);
  await new Promise(resolve => setTimeout(resolve, 800));
  console.log("Placeholder: Profile saved successfully");
};
const fetchTrips = async (userId: string): Promise<any[]> => { // Return 'any' initially
    console.log("Placeholder: Fetching trips for user:", userId);
    await new Promise(resolve => setTimeout(resolve, 1200));
    const now = new Date();
    // Simulate API returning date strings
    return [
        { _id: 'trip1', firebaseUid: userId, timestamp: new Date(now.getTime() - 86400000).toISOString(), travelDateTime: new Date(now.getTime() - 86400000 + 3600000).toISOString(), startLocation: { name: 'Ferry Building', lat: 37.795, lon: -122.393 }, endLocation: { name: 'Golden Gate Park', lat: 37.769, lon: -122.486 }, city: 'san_francisco', predictedDuration: 15 },
        { _id: 'trip2', firebaseUid: userId, timestamp: new Date(now.getTime() - 172800000).toISOString(), travelDateTime: new Date(now.getTime() - 172800000 + 7200000).toISOString(), startLocation: { name: 'Times Square', lat: 40.758, lon: -73.985 }, endLocation: { name: 'Central Park', lat: 40.782, lon: -73.965 }, city: 'new_york', predictedDuration: 25 },
        { _id: 'trip3', firebaseUid: userId, timestamp: new Date(now.getTime() - 3600000).toISOString(), travelDateTime: now.toISOString(), startLocation: { name: 'Coit Tower', lat: 37.802, lon: -122.405 }, endLocation: { name: 'Fisherman\'s Wharf', lat: 37.808, lon: -122.417 }, city: 'san_francisco', predictedDuration: 12 },
        { _id: 'trip4', firebaseUid: userId, timestamp: "Invalid Date String", travelDateTime: null, startLocation: { name: 'Bad Data', lat: 0, lon: 0 }, endLocation: { name: 'Bad Data', lat: 0, lon: 0 }, city: 'new_york', predictedDuration: 5 },
    ];
};
const deleteTripAPI = async (userId: string, tripId: string): Promise<void> => {
    console.log(`Placeholder: Deleting trip ${tripId} for user ${userId}`);
    await new Promise(resolve => setTimeout(resolve, 500));
    console.log(`Placeholder: Trip ${tripId} deleted successfully`);
};
// --- End Placeholder API ---

// --- Types ---
type SortKey = keyof Trip | 'startLocation.name' | 'endLocation.name' | null;
type SortDirection = 'asc' | 'desc';

// --- Helper to safely create Date ---
const safeNewDate = (dateInput: string | number | Date | undefined | null): Date => {
    if (dateInput instanceof Date && isValid(dateInput)) { return dateInput; }
    const date = new Date(dateInput as any);
    return isValid(date) ? date : new Date(NaN); // Return invalid date object if creation fails
};

// --- ProfilePage Component ---
const ProfilePage: React.FC = () => {
  // --- State ---
  const { user } = useAuth();
  const navigate = useNavigate();
  const [profileData, setProfileData] = useState<UserProfile | null>(null);
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [isLoadingProfile, setIsLoadingProfile] = useState(true);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [tempDisplayName, setTempDisplayName] = useState('');
  const [tempPhone, setTempPhone] = useState('');
  const [tempLocation, setTempLocation] = useState('');
  const [trips, setTrips] = useState<Trip[]>([]); // State now holds Trip type with Date objects
  const [isLoadingTrips, setIsLoadingTrips] = useState(true);
  const [tripsError, setTripsError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('travelDateTime');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [filterStartDate, setFilterStartDate] = useState<string>('');
  const [filterEndDate, setFilterEndDate] = useState<string>('');
  const [filterCity, setFilterCity] = useState<'all' | 'new_york' | 'san_francisco'>('all');
  const [deletingTripId, setDeletingTripId] = useState<string | null>(null);
  const [confirmingDeleteTripId, setConfirmingDeleteTripId] = useState<string | null>(null);

  // --- Data Loading ---
  const loadProfile = useCallback(async () => {
      if (user) { setIsLoadingProfile(true); setProfileError(null); try { const data = await fetchProfile(user.uid); if (data) { setProfileData(data); setTempDisplayName(data.displayName || ''); setTempPhone(data.phone || ''); setTempLocation(data.location || ''); } else { setProfileError("Could not load profile data."); } } catch (err) { setProfileError("Failed to load profile data."); console.error("Error fetching profile:", err); } finally { setIsLoadingProfile(false); } } else { setIsLoadingProfile(false); setProfileData(null); }
  }, [user]);
  const loadTrips = useCallback(async () => {
      if (user) {
        setIsLoadingTrips(true); setTripsError(null);
        try {
          const rawData = await fetchTrips(user.uid);
          const processedData = rawData.map((t: any) => ({
              ...t,
              travelDateTime: safeNewDate(t.travelDateTime),
              timestamp: safeNewDate(t.timestamp),
          }));
          setTrips(processedData as Trip[]);
        }
        catch (err) { console.error("Error fetching trips:", err); setTripsError("Failed to load trip history."); }
        finally { setIsLoadingTrips(false); }
      } else { setIsLoadingTrips(false); setTrips([]); }
  }, [user]);
  useEffect(() => { loadProfile(); loadTrips(); }, [loadProfile, loadTrips]);

  // --- Event Handlers ---
  const handleSignOut = async () => { try { await signOut(auth); navigate('/'); } catch (error) { setProfileError("Failed to sign out."); console.error("Sign out error:", error); } };
  const handleSaveProfile = async () => { if (!user || !profileData) return; setIsSavingProfile(true); setProfileError(null); const updatedData: Partial<Pick<UserProfile, 'displayName' | 'phone' | 'location'>> = {}; if (tempDisplayName !== profileData.displayName) updatedData.displayName = tempDisplayName; if (tempPhone !== profileData.phone) updatedData.phone = tempPhone; if (tempLocation !== profileData.location) updatedData.location = tempLocation; if (Object.keys(updatedData).length === 0) { setIsEditingProfile(false); setIsSavingProfile(false); return; } try { await saveProfile(user.uid, updatedData); setProfileData(prev => ({ ...prev!, ...updatedData })); if (updatedData.displayName && auth.currentUser && auth.currentUser.displayName !== updatedData.displayName) { await updateProfile(auth.currentUser, { displayName: updatedData.displayName }); } setIsEditingProfile(false); } catch (err) { setProfileError("Failed to save profile."); console.error("Save profile error:", err); } finally { setIsSavingProfile(false); } };
  const handleCancelEditProfile = () => { if (profileData) { setTempDisplayName(profileData.displayName || ''); setTempPhone(profileData.phone || ''); setTempLocation(profileData.location || ''); } setIsEditingProfile(false); setProfileError(null); };
  const handleSort = (key: SortKey) => { if (!key) return; if (key === sortKey) { setSortDirection(prev => (prev === 'asc' ? 'desc' : 'asc')); } else { setSortKey(key); setSortDirection('asc'); } };
  const handleDeleteTrip = async (tripId: string) => { if (!user) return; setDeletingTripId(tripId); setTripsError(null); try { await deleteTripAPI(user.uid, tripId); setTrips(currentTrips => currentTrips.filter(trip => trip._id !== tripId)); } catch (err: any) { console.error("Delete trip error:", err); setTripsError(`Failed to delete trip: ${err.message || 'Please try again.'}`); } finally { setDeletingTripId(null); setConfirmingDeleteTripId(null); } };

  // --- Filtering & Sorting Memo ---
  const sortedAndFilteredTrips = useMemo(() => {
    let tempTrips = [...trips];
    // Ensure dates are valid Date objects and filter invalid ones
    tempTrips = tempTrips.filter(trip => trip.travelDateTime instanceof Date && isValid(trip.travelDateTime));
    // Filtering
    if (filterCity !== 'all') { tempTrips = tempTrips.filter(trip => trip.city === filterCity); }
    try { if (filterStartDate) { const start = safeNewDate(filterStartDate); start.setHours(0, 0, 0, 0); if (isValid(start)) { tempTrips = tempTrips.filter(trip => trip.travelDateTime >= start); } } if (filterEndDate) { const end = safeNewDate(filterEndDate); end.setHours(23, 59, 59, 999); if (isValid(end)) { tempTrips = tempTrips.filter(trip => trip.travelDateTime <= end); } } } catch (e) { console.error("Date filter error", e); }
    // Sorting
    if (sortKey) {
      tempTrips.sort((a, b) => {
        const getSortValue = (obj: Trip, key: SortKey): string | number | null => { if (!key) return null; if (key === 'startLocation.name') return obj.startLocation.name?.toLowerCase() ?? ''; if (key === 'endLocation.name') return obj.endLocation.name?.toLowerCase() ?? ''; const value = obj[key as keyof Trip]; if (value instanceof Date && isValid(value)) return value.getTime(); if (typeof value === 'string') return value.toLowerCase(); if (typeof value === 'number') return value; return null; };
        const valA = getSortValue(a, sortKey); const valB = getSortValue(b, sortKey);
        let comparison = 0; if (valA === null && valB === null) comparison = 0; else if (valA === null) comparison = 1; else if (valB === null) comparison = -1; else if (typeof valA === 'number' && typeof valB === 'number') comparison = valA - valB; else if (typeof valA === 'string' && typeof valB === 'string') comparison = valA.localeCompare(valB);
        return sortDirection === 'asc' ? comparison : -comparison;
      });
    }
    return tempTrips;
  }, [trips, sortKey, sortDirection, filterStartDate, filterEndDate, filterCity]);

  // --- Render Loading/Error/No User ---
  if (isLoadingProfile && !profileData) { /* ... Loading ... */ return (<div className="flex justify-center items-center min-h-screen"><Loader2 className="h-8 w-8 animate-spin text-primary" /></div>); }
  if (!user) { /* ... Redirecting ... */ navigate('/sign-in'); return null; }
  if (!profileData && !isLoadingProfile) { /* ... Error ... */ return (<div className="flex flex-col justify-center items-center min-h-screen space-y-4 p-4 text-center"><p className="text-destructive">{profileError || "Could not load profile data."}</p><Button onClick={handleSignOut} variant="outline">Sign Out</Button></div>); }
  if (!profileData) { /* ... Fallback Loading ... */ return (<div className="flex justify-center items-center min-h-screen"><Loader2 className="h-8 w-8 animate-spin text-primary" /></div>); }


  // --- Sortable Header Renderer ---
  const renderSortableHeader = (key: SortKey, label: string) => (
    <th className="p-2 text-left cursor-pointer hover:bg-muted/50 whitespace-nowrap group" onClick={() => handleSort(key)}>
      <div className="flex items-center gap-1 select-none">
        {label}
        <span className="opacity-0 group-hover:opacity-100 transition-opacity">
           {sortKey === key ? ( sortDirection === 'asc' ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" /> ) : ( <ArrowUpDown className="h-3 w-3 text-muted-foreground/50" /> )}
        </span>
      </div>
    </th>
  );


  // --- Main Return JSX ---
  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background">
      {/* Background & Header */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 0.6 }} transition={{ duration: 0.8 }} aria-hidden className="pointer-events-none absolute inset-0 -z-10 opacity-60" style={{ backgroundImage: 'url(\'data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%239C92AC" fill-opacity="0.1"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\')', backgroundPosition: "center", backgroundSize: "60px 60px" }} />
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8 }} className="absolute inset-0 -z-10 bg-gradient-to-br from-white/80 via-white/60 to-white/30 dark:from-black/40 dark:via-black/25 dark:to-black/10" />
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: "easeOut" }} className="container mx-auto flex h-16 items-center justify-between px-4">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6, delay: 0.1 }} className="flex items-center gap-3">
          <Link to="/" className="flex items-center gap-3"> <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15 text-primary shadow-soft"><Car className="h-5 w-5" /></div> <span className="text-base font-semibold tracking-tight">GoPredict</span> </Link>
        </motion.div>
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6, delay: 0.1 }} className="flex items-center gap-2">
          <UserNav />
          <ThemeToggle />
        </motion.div>
      </motion.header>

      {/* Main Content */}
      <main className="container mx-auto flex-1 p-4 md:p-8 flex flex-col items-center space-y-8">
        {/* Profile Section */}
        <motion.section aria-labelledby="profile-heading" className="w-full max-w-2xl rounded-2xl border border-border bg-card/90 p-6 shadow-soft backdrop-blur space-y-6">
          <div className="flex items-center justify-between">
            <h1 id="profile-heading" className="text-2xl font-bold">Your Profile</h1>
            <div> {isEditingProfile ? (<Button variant="ghost" size="icon" onClick={handleCancelEditProfile} disabled={isSavingProfile} title="Cancel Edit"><X className="h-5 w-5 text-muted-foreground hover:text-foreground"/></Button>) : (<Button variant="ghost" size="icon" onClick={() => { setIsEditingProfile(true); setProfileError(null); }} title="Edit Profile"><Edit className="h-5 w-5 text-muted-foreground hover:text-foreground"/></Button>)}</div>
          </div>
          <div className="space-y-4">
               <div> <label htmlFor="displayName" className="block text-sm font-medium text-muted-foreground">Full Name</label> {isEditingProfile ? <Input id="displayName" value={tempDisplayName} onChange={(e) => setTempDisplayName(e.target.value)} disabled={isSavingProfile} className="mt-1" placeholder="Your full name"/> : <p className="mt-1 text-base font-medium">{profileData.displayName || <span className="italic text-muted-foreground">Not set</span>}</p>} </div>
               <div> <label htmlFor="email" className="block text-sm font-medium text-muted-foreground">Email</label> <p className="mt-1 text-base text-muted-foreground">{profileData.email}</p> </div>
               <div> <label htmlFor="phone" className="block text-sm font-medium text-muted-foreground">Phone Number</label> {isEditingProfile ? <Input id="phone" type="tel" value={tempPhone} onChange={(e) => setTempPhone(e.target.value)} placeholder="e.g., +1 123 456 7890" disabled={isSavingProfile} className="mt-1" /> : <p className="mt-1 text-base font-medium">{profileData.phone || <span className="italic text-muted-foreground">Not set</span>}</p>} </div>
               <div> <label htmlFor="location" className="block text-sm font-medium text-muted-foreground">Current Location</label> {isEditingProfile ? <Input id="location" value={tempLocation} onChange={(e) => setTempLocation(e.target.value)} placeholder="e.g., San Francisco, CA" disabled={isSavingProfile} className="mt-1" /> : <p className="mt-1 text-base font-medium">{profileData.location || <span className="italic text-muted-foreground">Not set</span>}</p>} </div>
               {profileError && (isEditingProfile || !isLoadingProfile) && (<p className="text-sm text-destructive mt-2">{profileError}</p>)}
               {isEditingProfile && (<div className="flex justify-end pt-4"><Button onClick={handleSaveProfile} disabled={isSavingProfile}>{isSavingProfile && <Loader2 className="mr-2 h-4 w-4 animate-spin" />} Save Changes</Button></div>)}
          </div>
           {!isEditingProfile && (<> <hr className="my-6 border-border/50"/> <Button variant="outline" onClick={handleSignOut} className="w-full">Sign Out</Button> </>)}
        </motion.section>

        {/* Trip History Section */}
        <motion.section aria-labelledby="trip-history-heading" className="w-full max-w-4xl rounded-2xl border border-border bg-card/90 p-6 shadow-soft backdrop-blur space-y-6">
            <div className="flex items-center justify-between">
                <h2 id="trip-history-heading" className="text-2xl font-bold">Trip History</h2>
            </div>
            {/* Filter Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 border rounded-lg bg-muted/30">
                 <div> <label htmlFor="filterCity" className="block text-sm font-medium text-muted-foreground mb-1">City</label> <select id="filterCity" value={filterCity} onChange={(e) => setFilterCity(e.target.value as any)} className="w-full p-2 border rounded bg-background text-sm h-10"> <option value="all">All Cities</option><option value="new_york">New York</option><option value="san_francisco">San Francisco</option> </select> </div>
                 <div> <label htmlFor="filterStartDate" className="block text-sm font-medium text-muted-foreground mb-1">Start Date</label> <Input id="filterStartDate" type="date" value={filterStartDate} onChange={e => setFilterStartDate(e.target.value)} className="text-sm"/> </div>
                 <div> <label htmlFor="filterEndDate" className="block text-sm font-medium text-muted-foreground mb-1">End Date</label> <Input id="filterEndDate" type="date" value={filterEndDate} onChange={e => setFilterEndDate(e.target.value)} className="text-sm"/> </div>
            </div>
            {/* Trip Table */}
            <div className="overflow-x-auto relative">
                {/* Error Message */}
                {tripsError && !isLoadingTrips && (<div className="my-2 p-2 text-center text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded"><AlertTriangle className="inline-block h-4 w-4 mr-1" /> {tripsError}</div>)}
                {/* Loading State */}
                {isLoadingTrips && (<div className="flex justify-center items-center py-8"><Loader2 className="h-6 w-6 animate-spin text-primary"/> <span className="ml-2 text-muted-foreground">Loading trips...</span></div>)}
                {/* No Trips Message */}
                {!isLoadingTrips && !tripsError && sortedAndFilteredTrips.length === 0 && (<p className="text-muted-foreground text-center py-4">No trip history found.</p>)}
                {/* Table */}
                {!isLoadingTrips && !tripsError && sortedAndFilteredTrips.length > 0 && (
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b">
                                {renderSortableHeader('travelDateTime', 'Trip Date/Time')}
                                {renderSortableHeader('startLocation.name', 'Start Location')}
                                {renderSortableHeader('endLocation.name', 'End Location')}
                                {renderSortableHeader('city', 'City')}
                                {renderSortableHeader('predictedDuration', 'Duration (min)')}
                                <th className="p-2 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedAndFilteredTrips.map((trip) => {
                                const isDateValid = isValid(trip.travelDateTime);
                                return (
                                    <tr key={trip._id} className={`border-b hover:bg-muted/50 ${confirmingDeleteTripId === trip._id ? 'bg-destructive/10' : ''}`}>
                                        <td className="p-2 whitespace-nowrap">
                                            {isDateValid ? format(trip.travelDateTime, 'Pp') : <span className="text-destructive text-xs italic">Invalid Date</span> }
                                        </td>
                                        <td className="p-2">{trip.startLocation.name}</td>
                                        <td className="p-2">{trip.endLocation.name}</td>
                                        <td className="p-2 capitalize">{trip.city.replace('_', ' ')}</td>
                                        <td className="p-2 text-right">{trip.predictedDuration}</td>
                                        <td className="p-2 text-right whitespace-nowrap">
                                            {confirmingDeleteTripId === trip._id ? (
                                                <div className="flex items-center justify-end gap-1">
                                                     <span className="text-xs text-destructive mr-1">Delete?</span>
                                                    <Button variant="destructive" size="sm" onClick={() => handleDeleteTrip(trip._id)} disabled={deletingTripId === trip._id} title="Confirm Delete" className="h-6 px-2 text-xs"> {deletingTripId === trip._id ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Yes'} </Button>
                                                    <Button variant="ghost" size="sm" onClick={() => setConfirmingDeleteTripId(null)} disabled={deletingTripId === trip._id} title="Cancel Delete" className="h-6 px-2 text-xs"> No </Button>
                                                </div>
                                            ) : (
                                                <Button variant="ghost" size="icon" onClick={() => setConfirmingDeleteTripId(trip._id)} disabled={!!deletingTripId || isSavingProfile} title="Delete Trip" className="text-muted-foreground hover:text-destructive h-6 w-6"> {deletingTripId === trip._id ? (<Loader2 className="h-4 w-4 animate-spin" />) : (<Trash2 className="h-4 w-4" />)} </Button>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </motion.section>
      </main>

    </div>
  );
};

export default ProfilePage;