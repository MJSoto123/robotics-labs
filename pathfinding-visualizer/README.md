# Pathfinding Visualizer 🗺️

Herramienta educativa para visualizar y comparar algoritmos de pathfinding en tiempo real con soporte para obstáculos dinámicos.

## 🎯 Propósito

Este visualizador NO es un juego - es una **herramienta educativa** que demuestra:

- **Por qué A\* no siempre es el mejor algoritmo**
- **Cuándo usar cada algoritmo específico**
- **El impacto de obstáculos dinámicos en el rendimiento**
- **Comparación cuantitativa** mediante métricas en tiempo real

## 🚀 Inicio Rápido

### Activar entorno virtual (Windows):
```cmd
venv\Scripts\activate
```

### Ejecutar:
```cmd
python main.py
```

## 🎮 Controles Completos

| Tecla/Acción | Función |
|--------------|---------|
| **LEFT CLICK** | Dibujar/quitar obstáculos |
| **S + CLICK** | Establecer nodo de inicio (verde) |
| **G + CLICK** | Establecer nodo objetivo (rojo) |
| **SPACE** | Ejecutar algoritmo |
| **D** | Activar/desactivar MODO DINÁMICO |
| **1** | Seleccionar Dijkstra |
| **2** | Seleccionar A\* (default) |
| **3** | Seleccionar D\*-lite |
| **4** | Seleccionar BMSSP |
| **↑ / ↓** | Ajustar velocidad de animación |
| **R** | Reiniciar camino (mantiene obstáculos) |
| **C** | Limpiar todos los obstáculos |
| **Q / ESC** | Salir |

## 🔥 Característica Principal: Obstáculos Dinámicos

### ¿Qué es el Modo Dinámico?

Presiona **D** para activar el modo dinámico. Una vez activado:

- **Durante la ejecución del algoritmo**: puedes agregar obstáculos haciendo click
- **Propósito**: Demostrar cómo cada algoritmo maneja cambios en el entorno

### Comportamiento por Algoritmo:

| Algoritmo | Comportamiento con Obstáculos Dinámicos |
|-----------|----------------------------------------|
| **Dijkstra** | ❌ No replantea - puede fallar o dar camino sub-óptimo |
| **A\*** | ❌ No replantea - sigue su camino original |
| **D\*-lite** | ✅ Diseñado para replanificación incremental |
| **BMSSP** | ⚡ Búsqueda bidireccional - puede adaptarse mejor |

## 📊 Algoritmos Implementados

### 1. Dijkstra
- **Tipo**: Búsqueda exhaustiva sin heurística
- **Ventajas**: Garantiza camino óptimo, funciona sin estimaciones
- **Desventajas**: Explora uniformemente (lento)
- **Cuándo usar**: No hay heurística válida disponible

### 2. A\* (A-Star)
- **Tipo**: Búsqueda informada con heurística Manhattan
- **Ventajas**: Más rápido que Dijkstra (~50% menos nodos)
- **Desventajas**: Requiere heurística admisible
- **Cuándo usar**: Entornos estáticos con buena heurística

### 3. D\*-lite
- **Tipo**: Búsqueda incremental con replanificación
- **Ventajas**: Eficiente en entornos dinámicos
- **Desventajas**: Más complejo de implementar
- **Cuándo usar**: Robots/agentes en entornos desconocidos

### 4. BMSSP (Bidirectional Multi-Source)
- **Tipo**: Búsqueda bidireccional
- **Ventajas**: Explora desde ambos extremos (más eficiente)
- **Desventajas**: Complejidad en la implementación
- **Cuándo usar**: Distancias muy largas

## 📈 Métricas en Tiempo Real

El visualizador muestra:

```
Metricas: Nodos=2196  Tiempo=3.62s  Longitud=76
```

- **Nodos explorados**: Cuánto trabajo realizó el algoritmo
- **Tiempo**: Duración de la ejecución
- **Longitud del camino**: Optimidad de la solución

## 🧪 Experimentos Sugeridos

### Experimento 1: A\* vs Dijkstra en Espacio Abierto

```
1. Coloca inicio y meta en esquinas opuestas
2. NO agregues obstáculos
3. Ejecuta A* (tecla 2) → Observa métricas
4. Presiona R para reiniciar
5. Cambia a Dijkstra (tecla 1) → Compara métricas
```

**Resultado esperado**: A\* explora ~50% menos nodos

### Experimento 2: Obstáculos Dinámicos

```
1. Presiona D para activar modo dinámico
2. Ejecuta A* (tecla 2) con velocidad lenta (↓)
3. Durante ejecución: dibuja obstáculos en frente del camino
4. Observa cómo A* NO puede replantear
5. Presiona R y prueba con D*-lite (tecla 3)
```

**Resultado esperado**: D\*-lite maneja cambios mejor

### Experimento 3: Comparación Completa

```
1. Crea un escenario con obstáculos complejos
2. Anota métricas de cada algoritmo:
   - Dijkstra (tecla 1)
   - A* (tecla 2)
   - D*-lite (tecla 3)
   - BMSSP (tecla 4)
3. Compara nodos explorados y tiempo
```

## 📂 Estructura del Proyecto

```
pathfinding-visualizer/
├── main.py                     # Punto de entrada
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias
│
├── src/
│   ├── environment/            # Grid y nodos
│   │   ├── node.py            # Clase Node con costos y estados
│   │   └── grid.py            # Grid 60x45 con gestión de vecinos
│   │
│   ├── visualization/          # Renderizado
│   │   ├── colors.py          # Esquema de colores
│   │   └── renderer.py        # Sistema de dibujo Pygame
│   │
│   └── algorithms/             # Algoritmos de pathfinding
│       ├── dijkstra.py        # Dijkstra con generador
│       ├── astar.py           # A* con heurística Manhattan
│       ├── dstar_lite.py      # D*-lite simplificado
│       └── bmssp.py           # Búsqueda bidireccional
│
└── venv/                       # Entorno virtual
```

## 🎨 Esquema de Colores

| Color | Significado |
|-------|-------------|
| 🟩 Verde | Nodo de inicio |
| 🟥 Rojo | Nodo objetivo |
| ⬛ Negro | Obstáculo |
| 🟦 Azul claro | Nodos en evaluación (open set) |
| 🟦 Azul medio | Nodos evaluados (closed set) |
| 🟨 Amarillo | Camino final encontrado |

## 💻 Requisitos

- Python 3.8+
- Pygame 2.x
- NumPy

## ⚙️ Configuración del Grid

Para cambiar el tamaño del grid, edita `main.py` (líneas 11-13):

```python
GRID_WIDTH = 60   # Ancho (número de columnas)
GRID_HEIGHT = 45  # Alto (número de filas)
CELL_SIZE = 15    # Tamaño de cada celda en píxeles
```

**Configuraciones recomendadas:**

- **Pequeño**: 40x30 con CELL_SIZE=20 (1,200 nodos)
- **Mediano**: 60x45 con CELL_SIZE=15 (2,700 nodos) - **Default**
- **Grande**: 80x60 con CELL_SIZE=12 (4,800 nodos)

## 📊 Estado del Proyecto

### ✅ Fase 1 - Setup Básico
- [x] Grid 2D interactivo configurable
- [x] Sistema de renderizado con Pygame
- [x] Colocación de obstáculos con mouse
- [x] Selección de nodo inicio/objetivo
- [x] Esquema de colores completo

### ✅ Fase 2 - Algoritmos Base
- [x] Dijkstra con visualización paso a paso
- [x] A\* con heurística Manhattan
- [x] Sistema de generadores para visualización
- [x] Control de velocidad de animación

### ✅ Fase 3 - Características Avanzadas
- [x] D\*-lite para replanificación
- [x] BMSSP bidireccional
- [x] **Obstáculos dinámicos** (MODO DINÁMICO)
- [x] **Métricas en tiempo real** (nodos, tiempo, longitud)
- [x] **Historial de resultados** para comparación
- [x] Grid de 60x45 (2,700 nodos)

## 🎓 Casos de Uso Educativos

### Para Estudiantes:
1. Visualizar cómo funcionan diferentes algoritmos
2. Entender el concepto de heurística (A\* vs Dijkstra)
3. Ver el impacto de obstáculos dinámicos
4. Comparar eficiencia mediante métricas

### Para Docentes:
1. Demostración visual en clases de algoritmos
2. Ejercicios comparativos con métricas cuantitativas
3. Explicación de casos de uso de cada algoritmo
4. Análisis de trade-offs (velocidad vs optimalidad)

### Para Desarrolladores:
1. Comparar rendimiento en escenarios reales
2. Probar heurísticas personalizadas
3. Entender cuándo usar cada algoritmo
4. Base para implementaciones propias

## 📝 Notas de Desarrollo

### Implementación Técnica:

- **Generadores Python**: Permiten visualización paso a paso sin bloquear
- **Priority Queue (heapq)**: Para selección eficiente del siguiente nodo
- **Estado de nodos**: Tracking de open/closed sets para visualización
- **Métricas en tiempo real**: Actualización continua durante ejecución

### Diferencias Clave:

| Aspecto | Dijkstra | A\* | D\*-lite | BMSSP |
|---------|----------|-----|----------|-------|
| Heurística | ❌ No | ✅ Sí | ✅ Sí | ✅ Sí |
| Replanificación | ❌ No | ❌ No | ✅ Sí | ⚡ Parcial |
| Dirección | → Unidireccional | → Unidireccional | ← Reversa | ↔ Bidireccional |
| Nodos explorados | Alto | Medio | Medio | Bajo |
| Complejidad | Baja | Media | Alta | Alta |

## 🤝 Contribuciones

Este es un proyecto educativo. Mejoras sugeridas:

- [ ] Más algoritmos (JPS, Theta\*, etc.)
- [ ] Escenarios predefinidos (laberintos)
- [ ] Modo comparación lado a lado
- [ ] Exportar métricas a CSV
- [ ] Heurísticas personalizables

## 📄 Licencia

Proyecto educativo para aprendizaje de algoritmos de pathfinding.

---

**Desarrollado como parte del laboratorio de robótica - Comparación de algoritmos de pathfinding**
