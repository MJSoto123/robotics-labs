# BMSSP – Bounded Multi-Source Shortest Path (Robótica – Lab)

Implementación en **C++** del algoritmo **BMSSP (Bounded Multi-Source Shortest Path)** para el laboratorio de **Robótica**. 

---

## Objetivo
- Implementar el algoritmo BMSSP en C++.
- Comparar tiempos de ejecución frente a Dijkstra.
---


## Compilación

Desde la carpeta `tests/`, compilar:

```bash
g++ -std=c++17 -O3 ./../src/graph_generator.cpp ./../src/dijkstra.cpp ./../src/data_structure_d.cpp ./../src/bmssp.cpp main.cpp -o test
```

## Ejecución
Ejemplo 1: generar un grafo de **10,000 vértices** y **50,000 aristas**  
```bash
./test -n 10000 -m 50000
```