@echo off
echo Starting GRIDX Development Environment...

echo.
echo Starting Backend (FastAPI)...
start "GRIDX Backend" cmd /k "cd backend && venv\Scripts\activate && uvicorn app.main:app --reload"

echo.
echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Frontend (React)...
start "GRIDX Frontend" cmd /k "cd frontend && npm start"

echo.
echo Development servers starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/api/docs

pause
