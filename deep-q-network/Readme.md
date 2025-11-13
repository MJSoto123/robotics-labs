# Deep Q-Network Controller
Este controlador implementa un agente **DQN (Deep Q-Network)** en **PyTorch** para entrenar un robot **E-Puck** en Webots a alcanzar una meta definida mediante aprendizaje por refuerzo.

## Descripción general

1. **Inicialización del entorno:**  
   Se configuran los motores, sensores de distancia y GPS del robot.  
   Si el sensor inercial está disponible, también se utiliza para obtener la orientación.

2. **Red neuronal:**  
   La red neuronal recibe como entrada un vector de estado continuo (distancias, posición, ángulo y orientación) y estima los valores Q para 4 acciones posibles:
   - Avanzar
   - Girar a la izquierda
   - Girar a la derecha
   - Girar en el lugar

3. **Entrenamiento DQN:**  
   - Se usa una política *epsilon-greedy* para equilibrar exploración y explotación.  
   - Las experiencias `(estado, acción, recompensa, siguiente estado)` se guardan en un buffer de repetición.  
   - En cada paso se entrena la red con un lote aleatorio del buffer.  
   - La red objetivo se actualiza periódicamente para estabilizar el aprendizaje.

4. **Recompensas:**  
   - Recompensa alta al llegar a la meta.  
   - Penalización por colisiones o salir de los límites.  
   - Recompensa intermedia por acercarse y orientarse hacia la meta.

5. **Guardado y carga de modelo:**  
   El modelo se guarda automáticamente al final de cada episodio en el archivo `dqn_model_v2.pth`, y se puede cargar para continuar el entrenamiento desde ese punto.

## Comportamiento observado

Inicialmente, el robot se mueve de forma errática mientras explora el entorno.  
Después de algunos episodios, al encontrar la meta varias veces, la política converge rápidamente y el robot aprende a dirigirse de manera eficiente hacia el objetivo.
