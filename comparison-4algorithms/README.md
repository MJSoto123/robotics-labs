# Comparación de 4 Algoritmos de Caminos Más Cortos

Implementación y comparación de 4 algoritmos para encontrar caminos más cortos en grafos dirigidos de gran escala:

1. **BMSSP** (Bounded Multi-Source Shortest Path)
2. **Dijkstra** (algoritmo clásico)
3. **A*** (algoritmo heurístico)
4. **D*-lite** (planificación dinámica)

---

## 📊 Resultados Principales

Benchmarks ejecutados en grafos de **2M a 5M nodos**:

| Tamaño | BMSSP | A* | D*-lite | Dijkstra |
|--------|-------|-------|---------|----------|
| 2M | 0.084s | 0.559s | 0.845s | 5.086s |
| 3M | 0.069s | 0.776s | 1.254s | 7.143s |
| 4M | 0.233s | 1.214s | 1.935s | 9.948s |
| 5M | 0.239s | 1.624s | 2.602s | 12.122s |

### Speedup vs Dijkstra:
- **BMSSP**: 50-103x más rápido
- **A***: 7-9x más rápido
- **D*-lite**: 5-6x más rápido

---

## 🛠️ Compilación

### Windows
```cmd
compile.bat
```

### Linux/macOS
```bash
chmod +x compile.sh
./compile.sh
```

### Manual
```bash
cd test/
g++ -std=c++17 -O3 -march=native -mtune=native \
  ./../src/graph_generator.cpp \
  ./../src/dijkstra.cpp \
  ./../src/data_structure_d.cpp \
  ./../src/bmssp.cpp \
  ./../src/astar.cpp \
  ./../src/dstar_lite.cpp \
  main.cpp -o test_4algorithms
```

---

## 🚀 Ejecución

### Sintaxis
```bash
./test_4algorithms --graph <tipo> [opciones]
```

### Parámetros Generales
- `-n, --nodes <int>`: número de nodos (default: 2,000,000)
- `-m, --edges <int>`: número de aristas (default: 8,000,000)
- `-t, --trials <int>`: número de pruebas (default: 10)
- `-s, --seed <int>`: semilla base (default: 0)
- `-o, --output <archivo.csv>`: archivo de salida
- `--source <id>`: nodo fuente (default: 0)
- `--target <id>`: nodo objetivo (default: 1000)
- `--wmax <real>`: peso máximo de aristas (default: 100.0)

---

## 📝 Ejemplos de Uso

### Grafo Aleatorio - 2M nodos
```bash
./test_4algorithms --graph random-m -n 2000000 -m 8000000 -t 5 -o results_2M.csv
```

### Grafo Erdős-Rényi - 3M nodos
```bash
./test_4algorithms --graph er -n 3000000 --p 0.0003 -t 5 -o results_3M.csv
```

### Grafo Barabási-Albert - 4M nodos
```bash
./test_4algorithms --graph ba -n 4000000 --attach 3 -t 5 -o results_4M.csv
```

### Malla 2D - 2000x2000 (4M nodos)
```bash
./test_4algorithms --graph grid2d --rows 2000 --cols 2000 -t 5 -o results_4M.csv
```

### Grafo Watts-Strogatz - 5M nodos
```bash
./test_4algorithms --graph ws -n 5000000 --k 10 --beta 0.1 -t 5 -o results_5M.csv
```

---

## 📈 Análisis de Resultados

### Generar Gráficos
```bash
python analyze_results.py results_2M.csv results_3M.csv results_4M.csv results_5M.csv
```

Esto generará:
- `algorithm_comparison.png` - Gráfico comparativo de rendimiento

### Formato CSV de Salida
```csv
trial,seed,time_dijkstra,time_bmssp,time_astar,time_dstar_lite
0,0,5.086,0.084,0.559,0.845
```

---

## 🧪 Scripts de Testing

### Prueba Rápida (Verificación)
```bash
cd test/
quick_test.bat         # Windows
```

### Pruebas con Grafos Grandes
```bash
cd test/
run_large_tests.bat    # Windows
./run_large_tests.sh   # Linux/macOS
```

---

## 📚 Estructura del Proyecto

```
comparison-4algorithms/
├── README.md              # Este archivo
├── compile.bat/sh         # Scripts de compilación
├── analyze_results.py     # Script de análisis
├── algorithm_comparison.png # Gráfico de resultados
│
├── include/               # Headers
│   ├── types.h           # Tipos base
│   ├── graph_generator.h # Generación de grafos
│   ├── dijkstra.h        # Dijkstra clásico
│   ├── bmssp.h           # BMSSP
│   ├── astar.h           # A*
│   ├── dstar_lite.h      # D*-lite
│   └── data_structure_d.h # Estructura auxiliar
│
├── src/                   # Implementaciones
│   ├── graph_generator.cpp
│   ├── dijkstra.cpp
│   ├── bmssp.cpp
│   ├── astar.cpp
│   ├── dstar_lite.cpp
│   └── data_structure_d.cpp
│
└── test/                  # Testing y benchmarking
    ├── main.cpp           # Programa principal
    ├── results_*.csv      # Resultados de benchmarks
    ├── quick_test.bat     # Prueba rápida
    └── run_large_tests.* # Suite de pruebas grandes
```

---

## 🎯 Características de los Algoritmos

### BMSSP (Bounded Multi-Source Shortest Path)
- **Ventaja**: Dramáticamente más rápido (50-103x)
- **Uso**: Grafos densos de gran escala
- **Consistencia**: Excelente en grafos grandes (CV < 0.07)

### A* (A-Star)
- **Ventaja**: Balance óptimo velocidad/consistencia (7-9x)
- **Uso**: Sistemas de tiempo real, navegación
- **Consistencia**: Muy predecible (CV < 0.16)

### D*-lite
- **Ventaja**: Optimizado para replanificación (5-6x)
- **Uso**: Entornos dinámicos que cambian
- **Consistencia**: Alta estabilidad (CV < 0.13)

### Dijkstra
- **Ventaja**: Referencia, máxima garantía teórica
- **Uso**: Cuando precisión > velocidad
- **Consistencia**: Máxima (CV < 0.05)

---

## ⚙️ Requisitos

- **Compilador**: g++ con soporte C++17
- **Memoria**: Mínimo 8GB RAM, recomendado 16GB+
- **Python**: 3.6+ (para análisis de resultados)
- **Librerías Python**: pandas, numpy, matplotlib

---

## 🔧 Optimizaciones Implementadas

- Compilación con `-O3 -march=native`
- Reserva de memoria con `reserve()`
- Uso eficiente de estructuras STL
- Heurísticas optimizadas para A* y D*-lite
- Instrumentación de métricas de rendimiento

---

## 📊 Resultados de Benchmarks Incluidos

- `test/results_2M.csv` - 2 millones de nodos
- `test/results_3M.csv` - 3 millones de nodos
- `test/results_4M.csv` - 4 millones de nodos
- `test/results_5M.csv` - 5 millones de nodos

---

## 🤝 Contribución

Este proyecto es parte de un laboratorio colaborativo de robótica. Cada colaborador implementa y compara diferentes enfoques.

---

## 📖 Referencias

- BMSSP: Bounded Multi-Source Shortest Path algorithm
- Dijkstra: Classic shortest path algorithm (1959)
- A*: Hart, Nilsson, and Raphael (1968)
- D*-lite: Koenig and Likhachev (2002)
