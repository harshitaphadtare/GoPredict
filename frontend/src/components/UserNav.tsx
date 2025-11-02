import { useNavigate } from 'react-router-dom';
import { Button } from './ui/button'; // Standard button
import { auth } from '../firebase'; // Go UP one level from components to src
import { signOut } from 'firebase/auth';
import { useAuth } from '../AuthContext'; // Go UP one level from components to src
import { LogOut, User as UserIcon } from 'lucide-react'; // Import icons

export function UserNav() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      // AuthContext will handle the redirect via its listener
      navigate('/');
    } catch (error) {
      console.error('Failed to sign out:', error);
    }
  };

  if (!user) {
    return null; // Don't show anything if user is not logged in
  }

  return (
    // Container to hold the two buttons side-by-side
    <div className="flex items-center gap-2">
      {/* Button to navigate to the Profile page */}
      <Button
        variant="ghost" // Use ghost variant for a subtle look
        size="icon"       // Make it icon-sized
        onClick={() => navigate('/profile')}
        title="Profile" // Add tooltip text
      >
        <UserIcon className="h-5 w-5" /> {/* User icon */}
      </Button>

      {/* Button to handle Sign Out */}
      <Button
        variant="ghost" // Use ghost variant
        size="icon"       // Make it icon-sized
        onClick={handleSignOut}
        title="Sign Out" // Add tooltip text
      >
        <LogOut className="h-5 w-5" /> {/* Sign out icon */}
      </Button>
    </div>
  );
}

