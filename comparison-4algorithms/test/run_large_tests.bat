@echo off
REM Scripts de ejemplo para pruebas con grafos grandes
REM Ejecutar desde el directorio test/

echo ========================================
echo  PRUEBAS CON GRAFOS GRANDES (2-5M nodos)
echo ========================================
echo.

echo [1/6] Grafo Aleatorio - 2M nodos, 8M aristas
test_4algorithms.exe --graph random-m -n 2000000 -m 8000000 -t 3 -o times_2M_random.csv
echo.

echo [2/6] Grafo Erdős–Rényi - 3M nodos
test_4algorithms.exe --graph er -n 3000000 --p 0.0003 -t 3 -o times_3M_er.csv
echo.

echo [3/6] Grafo Barabási–Albert - 4M nodos
test_4algorithms.exe --graph ba -n 4000000 --attach 3 -t 3 -o times_4M_ba.csv
echo.

echo [4/6] Malla 2D - 2000x2000 (4M nodos)
test_4algorithms.exe --graph grid2d --rows 2000 --cols 2000 -t 3 -o times_4M_grid.csv
echo.

echo [5/6] Grafo Watts–Strogatz - 5M nodos
test_4algorithms.exe --graph ws -n 5000000 --k 10 --beta 0.1 -t 3 -o times_5M_ws.csv
echo.

echo [6/6] DAG por Capas - 5M nodos
test_4algorithms.exe --graph layered-dag --layers 100 --width 50000 --dagp 0.01 -t 3 -o times_5M_dag.csv
echo.

echo ========================================
echo  TODAS LAS PRUEBAS COMPLETADAS
echo ========================================
echo.
echo Archivos CSV generados:
dir *.csv

