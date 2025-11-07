import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from './AuthContext'; // Use relative path

export const PublicOnlyRoute: React.FC = () => {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    // Show a loading state or nothing while auth is checking
    return <div>Loading...</div>; 
  }

  if (user) {
    // User is authenticated, redirect them away from sign-in/up
    return <Navigate to="/" replace />;
  }

  // User is not authenticated, show the child route (SignIn or SignUp)
  return <Outlet />;
};


