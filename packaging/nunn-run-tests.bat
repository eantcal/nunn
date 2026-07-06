@echo off
setlocal EnableExtensions

set "NUNN_ROOT=%~dp0.."
for %%I in ("%NUNN_ROOT%") do set "NUNN_ROOT=%%~fI"
set "NUNN_BIN=%NUNN_ROOT%\bin"
set "PATH=%NUNN_BIN%;%PATH%"

if not exist "%NUNN_BIN%\nunn_tests.exe" (
    echo nunn_tests.exe was not installed in "%NUNN_BIN%".
    echo Reinstall Nunn with the runtime component enabled.
    echo.
    pause
    exit /b 1
)

pushd "%NUNN_BIN%"
nunn_tests.exe %*
set "TEST_EXIT=%ERRORLEVEL%"
popd

echo.
if "%TEST_EXIT%"=="0" (
    echo Nunn tests completed successfully.
) else (
    echo Nunn tests failed with exit code %TEST_EXIT%.
)
pause
exit /b %TEST_EXIT%
