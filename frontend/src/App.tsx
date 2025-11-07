import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { ThemeProvider } from 'next-themes';

// --- Using relative paths to be safe ---
import Home from './pages/Home';
import NotFound from './pages/NotFound';
import SignIn from './pages/SignIn';
import SignUp from './pages/SignUp';
import ProfilePage from './pages/Profile'; // Import Profile Page
import { PublicOnlyRoute } from './PublicOnlyRoute'; // Import Public Route
import { ProtectedRoute } from './ProtectedRoute'; // Import Protected Route

function App() {
  const router = createBrowserRouter([
    { path: '/', element: <Home /> },

    // Public-only routes (users who are logged in will be redirected)
    {
      element: <PublicOnlyRoute />,
      children: [
        { path: '/sign-in', element: <SignIn /> },
        { path: '/sign-up', element: <SignUp /> },
      ],
    },

    // Protected routes (users who are NOT logged in will be redirected)
    {
      element: <ProtectedRoute />,
      children: [
        { path: '/profile', element: <ProfilePage /> },
        // Add other protected routes here
      ],
    },

    // Not found route
    { path: '*', element: <NotFound /> },
  ]);

  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <RouterProvider router={router} />
    </ThemeProvider>
  );
}

export default App;

