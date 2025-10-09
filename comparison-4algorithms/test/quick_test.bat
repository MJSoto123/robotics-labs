@echo off
REM Script de prueba rápida para verificar la implementación
REM Ejecutar desde el directorio test/

echo ========================================
echo  PRUEBA RÁPIDA - VERIFICACIÓN
echo ========================================
echo.

echo [1/3] Compilando proyecto...
cd ..
g++ -std=c++17 -O2 ./src/graph_generator.cpp ./src/dijkstra.cpp ./src/data_structure_d.cpp ./src/bmssp.cpp ./src/astar.cpp ./src/dstar_lite.cpp ./test/main.cpp -o ./test/test_4algorithms.exe

if %errorlevel% neq 0 (
    echo ❌ Error en la compilación
    exit /b 1
)

echo ✅ Compilación exitosa
echo.

cd test

echo [2/3] Prueba con grafo pequeño (1000 nodos)...
test_4algorithms.exe --graph random-m -n 1000 -m 5000 -t 2 -o test_small.csv

if %errorlevel% neq 0 (
    echo ❌ Error en la ejecución
    exit /b 1
)

echo ✅ Prueba pequeña exitosa
echo.

echo [3/3] Prueba con grafo mediano (10000 nodos)...
test_4algorithms.exe --graph random-m -n 10000 -m 50000 -t 1 -o test_medium.csv

if %errorlevel% neq 0 (
    echo ❌ Error en la ejecución
    exit /b 1
)

echo ✅ Prueba mediana exitosa
echo.

echo ========================================
echo  VERIFICACIÓN COMPLETADA
echo ========================================
echo.
echo Archivos generados:
dir test_*.csv
echo.
echo Para ejecutar pruebas grandes, usar:
echo   run_large_tests.bat

