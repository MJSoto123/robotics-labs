# Q-Learning Robot Navigation

## Descripción general
Este proyecto implementa un agente que aprende a navegar hacia una meta utilizando el algoritmo **Q-Learning** dentro del simulador **Webots**.  
El robot usa sensores de distancia, GPS y brújula para percibir su entorno y aprender políticas de movimiento óptimas mediante refuerzo.

## Funcionamiento básico
1. Se crea una **Q-table** (tabla de pesos) que asocia cada estado `(sensores, posición, orientación)` con el valor de cada acción posible.  
   Si ya existe un archivo `q_table.pkl`, se carga para continuar el entrenamiento anterior.

2. En cada paso de simulación, el robot:
   - Percibe su estado actual.
   - Elige una acción (avanzar, girar izquierda, girar derecha o retroceder) usando una política *epsilon-greedy*.
   - Ejecuta la acción y recibe una recompensa basada en la distancia a la meta y la presencia de obstáculos.
   - Actualiza la Q-table según la ecuación de Q-Learning.

3. Si el robot **alcanza la meta**, se detiene inmediatamente.  
   En ese momento se aplica un **backward reward propagation**, donde la recompensa final se propaga hacia atrás por todos los pasos del episodio.  
   Esto acelera la convergencia del algoritmo, permitiendo que el robot aprenda más rápido de cada éxito.

4. Si el robot **no alcanza la meta después de 1000 pasos**, el episodio termina y el robot se detiene.  
   La tabla Q se guarda para conservar el progreso.

5. Después de cada episodio, la Q-table se actualiza y se guarda en el archivo `q_table.pkl` para ser reutilizada en entrenamientos posteriores.

## Comportamiento aprendido
Inicialmente, el robot se mueve de forma errática, explorando el entorno sin dirección clara.  
Con el paso de los episodios y la propagación hacia atrás de recompensas, el robot empieza a reconocer trayectorias exitosas.  
Una vez encuentra el camino a la meta algunas veces, la convergencia se acelera y el robot logra llegar consistentemente de manera eficiente.

Este comportamiento puede observarse en el video de demostración incluido en el repositorio:  
**Meta.mp4**
