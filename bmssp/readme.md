# BMSSP – Bounded Multi-Source Shortest Path (Robótica – Lab)

Implementación en **C++** del algoritmo **BMSSP (Bounded Multi-Source Shortest Path)**.

---

## Objetivo
- Implementar y ejecutar **BMSSP** en C++.
- **Comparar tiempos** de ejecución frente a **Dijkstra** en distintos tipos de grafos.
---

## Compilación

Desde `test/`:

```bash
g++ -std=c++17 -O3 \
  ./../src/graph_generator.cpp \
  ./../src/dijkstra.cpp \
  ./../src/data_structure_d.cpp \
  ./../src/bmssp.cpp \
  main.cpp -o test
```

## Ejecución
```
./test --graph <tipo> [opciones del grafo] [opciones generales]
```
---

## Opciones generales

- `-t, --trials <int>`: número de corridas (default: 100)  
- `-s, --seed <int>`: semilla base (default: 0)  
- `-o, --output <ruta.csv>`: archivo CSV de salida (default: benchmark_times.csv)  
- `--source <id>`: nodo fuente (default: 0)  
- `--wmax <real>`: peso máximo de aristas (default: 100.0)  

---

## Tipos de grafo y parámetros

### 1) random-m (aleatorio con M aristas)

Flags: `-n <nodos>`, `-m <aristas>`  

```
./test --graph random-m -n 10000 -m 50000 -t 50 -o times_random.csv
```


---

### 2) er (Erdős–Rényi G(n, p))

Flags: `-n <nodos>`, `--p <prob>`  

```
./test --graph er -n 20000 --p 0.0005 -t 50 -o times_er.csv
```


---

### 3) ba (Barabási–Albert)

Flags: `-n <nodos>`, `--attach <m>` (aristas de nuevo nodo)  

```
./test --graph ba -n 20000 --attach 3 -t 50 -o times_ba.csv
```


---

### 4) ws (Watts–Strogatz)

Flags: `-n <nodos>`, `--k <grado par>`, `--beta <rewire>`  

```
./test --graph ws -n 10000 --k 10 --beta 0.1 -t 50 -o times_ws.csv
```


---

### 5) grid2d (malla 2D)

Flags: `--rows <r>`, `--cols <c>`, `--diag` (opcional, 8 vecinos)  

```
./test --graph grid2d --rows 200 --cols 200 --diag -t 50 -o times_grid.csv
```


---

### 6) layered-dag (DAG por capas)

Flags: `--layers <L>`, `--width <W>`, `--dagp <p>`  

```
./test --graph layered-dag --layers 80 --width 300 --dagp 0.03 -t 50 -o times_dag.csv
```

