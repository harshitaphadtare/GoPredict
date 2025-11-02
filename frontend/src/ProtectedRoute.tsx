import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from './AuthContext'; // Relative path

export const ProtectedRoute: React.FC = () => {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    // Optional: Show a loading spinner
    return <div>Loading...</div>;
  }

  if (!user) {
    // User is not authenticated, redirect to sign-in page
    return <Navigate to="/sign-in" replace />;
  }

  // User is authenticated, render the child route
  return <Outlet />;
};

