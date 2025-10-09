#!/usr/bin/env python3
"""
Script de análisis de resultados para comparar los 4 algoritmos
Uso: python analyze_results.py [archivo.csv]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def analyze_csv(filename):
    """Analiza un archivo CSV de resultados"""
    if not os.path.exists(filename):
        print(f"[ERROR] Archivo {filename} no encontrado")
        return
    
    print(f"[ANALISIS] Analizando: {filename}")
    print("=" * 50)
    
    # Cargar datos
    df = pd.read_csv(filename)
    
    # Estadísticas básicas
    algorithms = ['time_dijkstra', 'time_bmssp', 'time_astar', 'time_dstar_lite']
    algorithm_names = ['Dijkstra', 'BMSSP', 'A*', 'D*-lite']
    
    print("Tiempos promedio (segundos):")
    for alg, name in zip(algorithms, algorithm_names):
        avg_time = df[alg].mean()
        std_time = df[alg].std()
        print(f"  {name:8}: {avg_time:.4f} ± {std_time:.4f}")
    
    print()
    
    # Comparación relativa
    print("Comparación relativa (vs Dijkstra):")
    dijkstra_avg = df['time_dijkstra'].mean()
    for alg, name in zip(algorithms[1:], algorithm_names[1:]):
        ratio = df[alg].mean() / dijkstra_avg
        if ratio < 1:
            print(f"  {name:8}: {ratio:.2f}x más rápido")
        else:
            print(f"  {name:8}: {ratio:.2f}x más lento")
    
    print()
    
    # Ranking de velocidad
    print("Ranking de velocidad (más rápido primero):")
    avg_times = [(df[alg].mean(), name) for alg, name in zip(algorithms, algorithm_names)]
    avg_times.sort()
    
    for i, (time, name) in enumerate(avg_times, 1):
        print(f"  {i}. {name:8}: {time:.4f}s")
    
    print()
    
    # Análisis de consistencia
    print("Consistencia (coeficiente de variación):")
    for alg, name in zip(algorithms, algorithm_names):
        cv = df[alg].std() / df[alg].mean()
        print(f"  {name:8}: {cv:.3f} ({'muy consistente' if cv < 0.1 else 'consistente' if cv < 0.2 else 'variable'})")
    
    print()

def create_comparison_plot(filenames):
    """Crea un gráfico comparativo con colores distintivos"""
    plt.figure(figsize=(14, 10))

    algorithms = ['time_dijkstra', 'time_bmssp', 'time_astar', 'time_dstar_lite']
    algorithm_names = ['Dijkstra', 'BMSSP', 'A*', 'D*-lite']

    # Colores distintivos y más comprensibles
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Azul, Rosa, Naranja, Rojo

    x_pos = np.arange(len(algorithm_names))
    width = 0.15

    for i, filename in enumerate(filenames):
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            avg_times = [df[alg].mean() for alg in algorithms]

            # Crear etiqueta más descriptiva
            label = filename.replace('test_', '').replace('.csv', '').replace('_', ' ').title()

            plt.bar(x_pos + i*width, avg_times, width,
                   label=label,
                   color=colors[i % len(colors)],
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=0.5)

    plt.xlabel('Algoritmos', fontsize=12, fontweight='bold')
    plt.ylabel('Tiempo Promedio (segundos)', fontsize=12, fontweight='bold')
    plt.title('Comparación de Rendimiento de Algoritmos de Caminos Más Cortos',
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x_pos + width*1.5, algorithm_names, fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Uso: python analyze_results.py [archivo1.csv] [archivo2.csv] ...")
        print("Ejemplo: python analyze_results.py times_2M_random.csv times_3M_er.csv")
        return
    
    filenames = sys.argv[1:]
    
    # Analizar cada archivo
    for filename in filenames:
        analyze_csv(filename)
        print()
    
    # Crear gráfico comparativo si hay múltiples archivos
    if len(filenames) > 1:
        print("[GRAFICO] Creando grafico comparativo...")
        create_comparison_plot(filenames)

if __name__ == "__main__":
    main()

