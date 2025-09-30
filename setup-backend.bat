@echo off
echo Setting up GoPredict Backend...
cd backend

echo Installing dependencies...
npm install

echo Creating environment file...
if not exist .env (
    copy env.example .env
    echo Environment file created
)

echo Backend setup complete!
echo.
echo Next steps:
echo 1. Run: npm start
echo 2. API will be available at http://localhost:8000
echo 3. Test with: http://localhost:8000/api/health
pause
