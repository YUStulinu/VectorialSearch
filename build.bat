@echo off
REM build.bat - direct MSVC build, for when CMake is not installed.
REM Run this from a "Developer Command Prompt for VS" so cl.exe is on PATH.
setlocal
if not exist out mkdir out

echo Compiling the AVX2 kernels (only this file gets /arch:AVX2)...
cl /nologo /std:c++17 /O2 /EHsc /arch:AVX2 /Iinclude /c src\distance_avx2.cpp /Foout\distance_avx2.obj
if errorlevel 1 goto fail

echo Compiling the core...
cl /nologo /std:c++17 /O2 /EHsc /Iinclude /c src\hnsw.cpp src\storage.cpp src\distance.cpp src\distance_scalar.cpp /Foout\
if errorlevel 1 goto fail

echo Building tests...
cl /nologo /std:c++17 /O2 /EHsc /Iinclude /Itests tests\test_hnsw.cpp out\*.obj /Feout\tests.exe
if errorlevel 1 goto fail

echo Building the benchmark...
cl /nologo /std:c++17 /O2 /EHsc /Iinclude bench\bench.cpp out\hnsw.obj out\storage.obj out\distance.obj out\distance_scalar.obj out\distance_avx2.obj /Feout\bench.exe
if errorlevel 1 goto fail

echo.
echo Done.  Run:  out\tests.exe   and   out\bench.exe
exit /b 0

:fail
echo.
echo Build failed. Make sure this is running inside a Developer Command Prompt.
exit /b 1
