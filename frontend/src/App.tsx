import React from 'react'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { ThemeProvider } from 'next-themes'
import Home from './pages/Home'
import NotFound from './pages/NotFound'

function App() {
  const router = createBrowserRouter(
    [
      { path: '/', element: <Home /> },
      { path: '*', element: <NotFound /> },
    ],
    {
      future: {
        v7_startTransition: true,
        v7_relativeSplatPath: true,
      },
    }
  )
  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <RouterProvider router={router} />
    </ThemeProvider>
  )
}

export default App
