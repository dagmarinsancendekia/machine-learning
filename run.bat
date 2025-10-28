@echo off
echo Starting Excel Analyzer Application...
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo Installing/updating requirements...
pip install -r requirements.txt

REM Start the backend server
echo Starting backend server on port 3000...
start "Excel Analyzer Backend" python backend/app.py

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Open the frontend in default browser
echo Opening frontend in browser...
start http://127.0.0.1:3000/frontend/index.html

echo.
echo Application started successfully!
echo - Backend running on: http://127.0.0.1:3000
echo - Frontend available at: http://127.0.0.1:3000/frontend/index.html
echo.
echo Press any key to stop the application...
pause > nul

REM Stop the backend (this will close when the batch file ends)
echo Stopping application...
taskkill /FI "WINDOWTITLE eq Excel Analyzer Backend*" /T /F > nul 2>&1

echo Application stopped.
pause
