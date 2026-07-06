@echo off
setlocal EnableExtensions

set "NUNN_ROOT=%~dp0.."
for %%I in ("%NUNN_ROOT%") do set "NUNN_ROOT=%%~fI"
set "NUNN_BIN=%NUNN_ROOT%\bin"
set "PATH=%NUNN_BIN%;%PATH%"

echo Nunn developer command prompt
echo Install root: %NUNN_ROOT%
echo.
echo Common commands:
echo   nunn_tests
echo   ocr_test
echo   winttt
echo   mnist_test
echo   xor_test
echo   net2json
echo.
echo Installed executables:
dir /b "%NUNN_BIN%\*.exe"
echo.

cmd /k
