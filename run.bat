@echo off
echo Starting Django Server...
start cmd /k "cd plant_api && python manage.py runserver 0.0.0.0:8000"

timeout /t 3

@echo off
echo Starting Flutter Windows App...
cd plant_disease\build\windows\x64\runner\Release
start plant_disease.exe
exit

echo Both servers started.
pause