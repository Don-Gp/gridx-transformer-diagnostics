# GRIDX Setup Script
Write-Host "🚀 GRIDX Development Setup" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green

# Backend setup
Write-Host "`n📦 Setting up Backend..." -ForegroundColor Yellow
Set-Location backend
python -m venv venv
& "venv\Scripts\activate.ps1"
pip install -r requirements.txt
Copy-Item ".env.example" ".env"

# Frontend setup  
Write-Host "`n⚛️ Setting up Frontend..." -ForegroundColor Yellow
Set-Location ../frontend
npm install
Copy-Item ".env.example" ".env"

Set-Location ..

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "Run 'start-dev.bat' to start development servers" -ForegroundColor Cyan
