@echo off
echo Setting up GoPredict Frontend...
cd frontend

echo Installing dependencies...
npm install

echo Creating environment file...
if not exist .env (
    copy env.example .env
    echo Please edit .env file and add your Google Maps API key
)

echo Frontend setup complete!
echo.
echo Next steps:
echo 1. Edit frontend\.env and add your Google Maps API key
echo 2. Run: npm run dev
echo 3. Open http://localhost:3000
pause
