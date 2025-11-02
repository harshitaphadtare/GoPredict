import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import {
  createUserWithEmailAndPassword,
  updateProfile,
  signInWithPopup,
  GoogleAuthProvider,
  GithubAuthProvider,
  // --- 1. Import the function to check email ---
  fetchSignInMethodsForEmail
} from 'firebase/auth';
import { auth } from '../firebase';
import { motion } from 'framer-motion';
import { Car } from 'lucide-react';
import { ThemeToggle } from '../components/ui/theme-toggle';

// --- SVG Icons (Google, GitHub) ---
const IconGoogle = () => (
  <svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2">
    <title>Google</title>
    <path d="M12.48 10.92v3.28h7.84c-.24 1.5-.9 2.73-1.89 3.6-1.08 1.08-2.56 1.62-4.4 1.62-3.8 0-6.88-3.1-6.88-6.9 0-3.8 3.1-6.9 6.88-6.9 2.2 0 3.96.86 4.9 1.74l2.4-2.32C18.1.9 15.48 0 12.48 0 5.8 0 0 5.6 0 12.3s5.8 12.3 12.48 12.3c3.2 0 5.76-1.1 7.68-3.08 1.96-2 3.04-4.8 3.04-8.08 0-.8-.1-1.48-.24-2.16z" fill="#4285F4" />
  </svg>
);

const IconGitHub = () => (
  <svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 fill-current">
    <title>GitHub</title>
    <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
  </svg>
);
// --- End Icons ---

const SignUp: React.FC = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    // ... (This function remains the same)
    e.preventDefault();
    setError(null);

    if (password.length < 6) {
      setError('Password must be at least 6 characters long.');
      return;
    }

    setIsLoading(true);

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      await updateProfile(user, {
        displayName: name
      });
      
      navigate('/'); 

    } catch (err: any) {
      if (err.code === 'auth/email-already-in-use') {
        setError('This email address is already in use.');
      } else if (err.code === 'auth/weak-password') {
        setError('Password is too weak. Please use at least 6 characters.');
      } else {
        setError('Failed to create account. Please try again.');
      }
      console.error(err); 

    } finally {
      setIsLoading(false);
    }
  };

  const handleSocialLogin = async (providerName: 'google' | 'github') => {
    setIsLoading(true);
    setError(null);
    
    const provider = providerName === 'google' 
      ? new GoogleAuthProvider() 
      : new GithubAuthProvider();
      
    try {
      // We still use signInWithPopup, as this handles both sign in AND sign up
      await signInWithPopup(auth, provider);
      navigate('/');
    } catch (err: any) {
      // --- 2. Add the same GitHub error handling logic ---
      if (err.code === 'auth/account-exists-with-different-credential') {
        const email = err.customData?.email;
        if (email) {
          // Find out which provider is linked to that email
          const methods = await fetchSignInMethodsForEmail(auth, email);
          // Capitalize the provider name for the error message
          const provider = methods[0].charAt(0).toUpperCase() + methods[0].slice(1).split('.')[0];
          setError(`This email is already linked to an account with ${provider}. Please sign in using ${provider}.`);
        } else {
          setError('Account exists with a different login method.');
        }
      } else {
        setError('Failed to sign up with ' + providerName + '. Please try again.');
      }
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background">
      {/* --- (Background and Header code) --- */}
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
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="container mx-auto flex h-16 items-center justify-between px-4"
      >
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="flex items-center gap-3"
        >
          <Link to="/" className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15 text-primary shadow-soft">
              <Car className="h-5 w-5" />
            </div>
            <span className="text-base font-semibold tracking-tight">
              GoPredict
            </span>
          </Link>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <ThemeToggle />
        </motion.div>
      </motion.header>

      {/* --- (Form Card code) --- */}
      <main className="flex flex-1 items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="w-full max-w-md rounded-2xl border border-border bg-card/90 p-6 shadow-soft backdrop-blur"
        >
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold">Sign Up</h1>
              <p className="text-muted-foreground">
                Create your account to get started.
              </p>
            </div>

            <form className="space-y-4" onSubmit={handleSubmit}>
              {/* --- (Form inputs) --- */}
              <div>
                <label htmlFor="name" className="text-sm font-medium">Full Name</label>
                <Input
                  id="name"
                  type="text"
                  placeholder="Your Name"
                  required
                  className="mt-1"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={isLoading}
                />
              </div>

              <div>
                <label htmlFor="email" className="text-sm font-medium">Email</label>
                <Input
                  id="email"
                  type="email"
                  placeholder="name@example.com"
                  required
                  className="mt-1"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={isLoading}
                />
              </div>

              <div>
                <label htmlFor="password" className="text-sm font-medium">Password</label>
                <Input
                  id="password"
                  type="password"
                  required
                  className="mt-1"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading}
                />
              </div>

              {/* --- 3. Error message will now show the helpful text --- */}
              {error && (
                <p className="text-sm text-destructive">{error}</p>
              )}

              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? 'Creating Account...' : 'Sign Up'}
              </Button>
            </form>

            {/* --- (Divider and Social buttons) --- */}
            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">
                  Or continue with
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <Button variant="outline" onClick={() => handleSocialLogin('google')} disabled={isLoading}>
                <IconGoogle />
                Google
              </Button>
              <Button variant="outline" onClick={() => handleSocialLogin('github')} disabled={isLoading}>
                <IconGitHub />
                GitHub
              </Button>
            </div>

            <p className="text-center text-sm text-muted-foreground">
              Already have an account?{' '}
              <Link to="/sign-in" className="font-medium text-primary hover:underline">
                Login
              </Link>
            </p>
          </div>
        </motion.div>
      </main>
    </div>
  );
};

export default SignUp;

