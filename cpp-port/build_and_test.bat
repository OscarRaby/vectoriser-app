@echo off
echo Building Organic Vectoriser C++ Foundation Tests...
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring CMake...
cmake ..
if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo.
echo Building...
cmake --build . --config Release
if errorlevel 1 (
    echo Build failed!
    cd ..
    pause
    exit /b 1
)

echo.
echo Build successful!
echo.
echo Running tests...
echo.

REM Run the test executable
Release\vectoriser_test.exe

cd ..
echo.
echo Tests completed. Check the generated PNG files.
pause
