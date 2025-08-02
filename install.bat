@echo off
echo 📱 Instalador del Detector de Uso de Celular
echo ==========================================
echo.
echo Verificando Python...
python --version
if errorlevel 1 (
    echo ❌ Python no está instalado o no está en PATH
    echo.
    echo Por favor instala Python desde: https://python.org
    pause
    exit /b 1
)
echo.
echo ✅ Python encontrado
echo.
echo Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Error instalando dependencias
    pause
    exit /b 1
)
echo.
echo ✅ Instalación completada exitosamente!
echo.
echo Para ejecutar el detector, usa:
echo   python phone_detector_optimized.py
echo.
echo O ejecuta directamente:
echo   dist\Detector-Celular.exe
echo.
pause