# Stop all Python processes first
taskkill /F /IM python.exe 2>nul

# Delete ALL cache files
del /s /q *.pyc
for /d /r . %i in (__pycache__) do @if exist "%i" rmdir /s /q "%i"

# Also delete the vector DB to start fresh
rmdir /s /q outputs\vector_db

# Verify cache is cleared
dir /s /b *.pyc
If that doesn't show "File Not Found", run this PowerShell version:
# In PowerShell
Get-ChildItem -Path . -Include *.pyc -Recurse | Remove-Item -Force
Get-ChildItem -Path . -Include __pycache__ -Recurse | Remove-Item -Recurse -Force
Remove-Item -Path outputs\vector_db -Recurse -Force -ErrorAction SilentlyContinue

scp -i ~/.ssh/azurevm_key.pem -r myproject azureuser@20.204.145.10:/home/azureuser/
