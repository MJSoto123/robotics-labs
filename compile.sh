#!/bin/bash

# Script de compilación para el proyecto con A* y D*-lite
# Uso: ./compile.sh

echo "Compilando proyecto con 4 algoritmos (BMSSP, Dijkstra, A*, D*-lite)..."

cd test/

# Compilar con optimizaciones para grafos grandes
g++ -std=c++17 -O3 -march=native -mtune=native \
  ./../src/graph_generator.cpp \
  ./../src/dijkstra.cpp \
  ./../src/data_structure_d.cpp \
  ./../src/bmssp.cpp \
  ./../src/astar.cpp \
  ./../src/dstar_lite.cpp \
  main.cpp -o test_4algorithms

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa!"
    echo "Ejecutable: test/test_4algorithms"
    echo ""
    echo "Ejemplos de uso:"
    echo "  # Grafo aleatorio 2M nodos, 8M aristas"
    echo "  ./test_4algorithms --graph random-m -n 2000000 -m 8000000 -t 5"
    echo ""
    echo "  # Grafo Erdős–Rényi 3M nodos"
    echo "  ./test_4algorithms --graph er -n 3000000 --p 0.0003 -t 5"
    echo ""
    echo "  # Grafo Barabási–Albert 4M nodos"
    echo "  ./test_4algorithms --graph ba -n 4000000 --attach 3 -t 5"
    echo ""
    echo "  # Malla 2D 2000x2000 (4M nodos)"
    echo "  ./test_4algorithms --graph grid2d --rows 2000 --cols 2000 -t 5"
else
    echo "❌ Error en la compilación"
    exit 1
fi

