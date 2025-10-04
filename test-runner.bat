@echo off
echo ========================================
echo Running GoPredict Test Suite
echo ========================================

echo.
echo [1/3] Installing Frontend Dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo Frontend dependency installation failed!
    exit /b 1
)

echo.
echo [2/3] Installing Backend Dependencies...
cd ..\backend
call npm install
if %errorlevel% neq 0 (
    echo Backend dependency installation failed!
    exit /b 1
)

echo.
echo [3/3] Running Tests...
echo.
echo === Frontend Tests ===
cd ..\frontend
call npm run test:run
set frontend_result=%errorlevel%

echo.
echo === Backend Tests ===
cd ..\backend
call npm test
set backend_result=%errorlevel%

echo.
echo ========================================
echo Test Results Summary
echo ========================================
if %frontend_result% equ 0 (
    echo Frontend Tests: PASSED ✓
) else (
    echo Frontend Tests: FAILED ✗
)

if %backend_result% equ 0 (
    echo Backend Tests: PASSED ✓
) else (
    echo Backend Tests: FAILED ✗
)

if %frontend_result% equ 0 if %backend_result% equ 0 (
    echo.
    echo All tests passed! 🎉
    exit /b 0
) else (
    echo.
    echo Some tests failed. Check the output above.
    exit /b 1
)
