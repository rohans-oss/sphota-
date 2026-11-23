# Starts the Sphota model API (port 8000) and the website (port 3000). Run from D:\diffusion:  .\run_website.ps1
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$root\pyshims;$root\sphota"
Start-Process -FilePath "$root\venv\Scripts\python.exe" -ArgumentList "-m","truesight.app.api" -WorkingDirectory "$root\sphota"
Set-Location "$root\sphota-web"
npm run dev
