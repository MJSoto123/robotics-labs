"""
Pathfinding Visualizer - Main Entry Point
"""
import pygame
import sys
import time
from src.environment import Grid
from src.visualization import Renderer, colors
from src.algorithms import dijkstra_generator, astar_generator, dstar_lite_generator, bmssp_generator

# Configuration - AJUSTA ESTOS VALORES PARA CAMBIAR EL TAMAÑO
GRID_WIDTH = 60   # Aumentado de 40 a 60
GRID_HEIGHT = 45  # Aumentado de 30 a 45
CELL_SIZE = 15    # Reducido de 20 a 15 para que quepa en pantalla
FPS = 60
INFO_PANEL_HEIGHT = 140  # Panel de información

# Calculate window size
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + INFO_PANEL_HEIGHT


class PathfindingVisualizer:
    """Main application class"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Pathfinding Visualizer")
        self.clock = pygame.time.Clock()

        # Create grid and renderer
        self.grid = Grid(GRID_WIDTH, GRID_HEIGHT, CELL_SIZE)
        self.renderer = Renderer(self.grid, self.screen)

        # State
        self.running = True
        self.drawing_mode = None  # None, 'obstacle', 'start', 'goal'
        self.algorithm_running = False
        self.algorithm_generator = None
        self.current_algorithm = "astar"  # Default to A*
        self.animation_speed = 10  # Steps per frame (higher = faster)
        self.dynamic_mode = False  # Allow adding obstacles during execution

        # Metrics
        self.nodes_explored = 0
        self.path_length = 0
        self.start_time = 0
        self.elapsed_time = 0
        self.last_path_found = False

        # Comparison history
        self.results_history = []

    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Allow clicks during algorithm if dynamic mode is on
                    if self.algorithm_running and self.dynamic_mode:
                        self.handle_dynamic_obstacle(event.pos)
                    elif not self.algorithm_running:
                        self.handle_mouse_click(event.pos)

            # Keyboard shortcuts
            elif event.type == pygame.KEYDOWN:
                self.handle_key_press(event.key)

        # Handle mouse drag for obstacles
        if pygame.mouse.get_pressed()[0]:  # Left button held
            if self.algorithm_running and self.dynamic_mode:
                self.handle_dynamic_obstacle(pygame.mouse.get_pos())
            elif not self.algorithm_running and not self.drawing_mode:
                self.handle_mouse_drag(pygame.mouse.get_pos())

    def handle_mouse_click(self, pos):
        """Handle mouse click on grid"""
        x, y = pos
        grid_x = x // CELL_SIZE
        grid_y = y // CELL_SIZE

        if grid_x < self.grid.width and grid_y < self.grid.height:
            # Handle start/goal placement
            if self.drawing_mode == 'start':
                if self.grid.set_start(grid_x, grid_y):
                    print(f">> INICIO colocado en ({grid_x}, {grid_y}) - Ahora presiona G para la meta")
                    self.drawing_mode = None
            elif self.drawing_mode == 'goal':
                if self.grid.set_goal(grid_x, grid_y):
                    print(f">> META colocada en ({grid_x}, {grid_y}) - LISTO! Presiona ESPACIO")
                    self.drawing_mode = None
            # Handle obstacle toggle when not in special mode
            else:
                self.grid.toggle_obstacle(grid_x, grid_y)

    def handle_mouse_drag(self, pos):
        """Handle mouse drag for drawing obstacles"""
        # Only draw obstacles when not in start/goal mode
        if self.drawing_mode:
            return

        x, y = pos
        grid_x = x // CELL_SIZE
        grid_y = y // CELL_SIZE

        if grid_x < self.grid.width and grid_y < self.grid.height:
            node = self.grid.get_node(grid_x, grid_y)
            if node and not node.is_start and not node.is_goal and not node.is_obstacle:
                node.is_obstacle = True
                node.state = "obstacle"

    def handle_dynamic_obstacle(self, pos):
        """Handle adding obstacles during algorithm execution"""
        x, y = pos
        grid_x = x // CELL_SIZE
        grid_y = y // CELL_SIZE

        if grid_x < self.grid.width and grid_y < self.grid.height:
            node = self.grid.get_node(grid_x, grid_y)
            if node and not node.is_start and not node.is_goal and not node.is_obstacle:
                node.is_obstacle = True
                node.state = "obstacle"
                print(f">> OBSTACULO DINAMICO agregado en ({grid_x}, {grid_y})")

    def handle_key_press(self, key):
        """Handle keyboard input"""
        # Set start node
        if key == pygame.K_s:
            self.drawing_mode = 'start'
            print(">> Haz CLICK en una celda para colocar el INICIO (verde)")

        # Set goal node
        elif key == pygame.K_g:
            self.drawing_mode = 'goal'
            print(">> Haz CLICK en una celda para colocar la META (rojo)")

        # Reset path (keep obstacles, start, goal)
        elif key == pygame.K_r:
            self.grid.reset_path()
            self.algorithm_running = False
            self.algorithm_generator = None
            print(">> Camino reiniciado (obstaculos mantenidos)")

        # Clear all obstacles
        elif key == pygame.K_c:
            self.grid.clear_obstacles()
            self.algorithm_running = False
            self.algorithm_generator = None
            print(">> Todo limpiado")

        # Quit
        elif key == pygame.K_q or key == pygame.K_ESCAPE:
            self.running = False

        # Toggle dynamic mode
        elif key == pygame.K_d:
            self.dynamic_mode = not self.dynamic_mode
            status = "ACTIVADO" if self.dynamic_mode else "DESACTIVADO"
            print(f">> Modo Dinamico: {status} - {'Puedes agregar obstaculos durante ejecucion' if self.dynamic_mode else 'Obstaculos solo antes de ejecutar'}")

        # Switch algorithm
        elif key == pygame.K_1:
            self.current_algorithm = "dijkstra"
            print(">> Algoritmo: DIJKSTRA (explora uniformemente)")

        elif key == pygame.K_2:
            self.current_algorithm = "astar"
            print(">> Algoritmo: A* (usa heuristica - mas rapido)")

        elif key == pygame.K_3:
            self.current_algorithm = "dstar_lite"
            print(">> Algoritmo: D*-LITE (replanificacion incremental)")

        elif key == pygame.K_4:
            self.current_algorithm = "bmssp"
            print(">> Algoritmo: BMSSP (busqueda bidireccional)")

        # Adjust animation speed
        elif key == pygame.K_UP:
            self.animation_speed = min(100, self.animation_speed + 5)
            print(f">> Velocidad: {self.animation_speed} pasos/frame (MAS RAPIDO)")

        elif key == pygame.K_DOWN:
            self.animation_speed = max(1, self.animation_speed - 5)
            print(f">> Velocidad: {self.animation_speed} pasos/frame (MAS LENTO)")

        # Run algorithm
        elif key == pygame.K_SPACE:
            if not self.algorithm_running:
                if self.grid.start and self.grid.goal:
                    self.start_algorithm()
                else:
                    print(">> ATENCION: Primero coloca INICIO (S) y META (G)!")

    def start_algorithm(self):
        """Initialize and start the selected algorithm"""
        self.grid.reset_path()  # Clear previous path
        self.algorithm_running = True

        # Reset metrics
        self.nodes_explored = 0
        self.path_length = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.last_path_found = False

        # Create generator based on selected algorithm
        if self.current_algorithm == "dijkstra":
            self.algorithm_generator = dijkstra_generator(
                self.grid, self.grid.start, self.grid.goal
            )
            print(">> EJECUTANDO DIJKSTRA... Explora uniformemente")
        elif self.current_algorithm == "astar":
            self.algorithm_generator = astar_generator(
                self.grid, self.grid.start, self.grid.goal
            )
            print(">> EJECUTANDO A*... Va directo a la meta")
        elif self.current_algorithm == "dstar_lite":
            self.algorithm_generator = dstar_lite_generator(
                self.grid, self.grid.start, self.grid.goal
            )
            print(">> EJECUTANDO D*-LITE... Busqueda incremental desde la meta")
        elif self.current_algorithm == "bmssp":
            self.algorithm_generator = bmssp_generator(
                self.grid, self.grid.start, self.grid.goal
            )
            print(">> EJECUTANDO BMSSP... Busqueda bidireccional (desde ambos lados)")

    def step_algorithm(self):
        """Execute one or more steps of the algorithm"""
        if not self.algorithm_running or not self.algorithm_generator:
            return

        try:
            # Run multiple steps per frame based on animation speed
            for _ in range(self.animation_speed):
                current, open_set, closed_set, path_found = next(self.algorithm_generator)

                # Update metrics
                self.nodes_explored = len(closed_set)
                self.elapsed_time = time.time() - self.start_time

                if path_found:
                    # Calculate path length
                    self.path_length = self.calculate_path_length()
                    self.last_path_found = True

                    # Save result to history
                    self.save_result()

                    print(f">> EXITO: Camino encontrado! | Nodos: {self.nodes_explored} | Tiempo: {self.elapsed_time:.2f}s | Longitud: {self.path_length}")
                    self.algorithm_running = False
                    break

        except StopIteration:
            self.elapsed_time = time.time() - self.start_time
            self.last_path_found = False
            print(f">> ERROR: No existe camino | Nodos explorados: {self.nodes_explored} | Tiempo: {self.elapsed_time:.2f}s")
            self.algorithm_running = False

    def calculate_path_length(self):
        """Calculate the length of the found path"""
        if not self.grid.goal:
            return 0

        length = 0
        current = self.grid.goal
        while current and current.parent:
            length += 1
            current = current.parent
        return length

    def save_result(self):
        """Save algorithm result to history for comparison"""
        result = {
            'algorithm': self.current_algorithm,
            'nodes_explored': self.nodes_explored,
            'time': self.elapsed_time,
            'path_length': self.path_length,
            'success': self.last_path_found
        }
        self.results_history.append(result)

        # Keep only last 10 results
        if len(self.results_history) > 10:
            self.results_history.pop(0)

    def update(self):
        """Update game state"""
        # Run algorithm steps if active
        if self.algorithm_running:
            self.step_algorithm()

    def draw(self):
        """Render everything"""
        self.screen.fill(colors.BACKGROUND)
        self.renderer.draw_grid()
        self.renderer.draw_instructions(
            self.current_algorithm,
            self.algorithm_running,
            self.dynamic_mode,
            self.nodes_explored,
            self.elapsed_time,
            self.path_length
        )
        pygame.display.flip()

    def run(self):
        """Main game loop"""
        print("=" * 75)
        print("=== VISUALIZADOR DE PATHFINDING (Grid: {}x{}) ===".format(GRID_WIDTH, GRID_HEIGHT))
        print("=" * 75)
        print("\nINSTRUCCIONES PASO A PASO:\n")
        print("1. Presiona 'S' y haz CLICK -> Coloca INICIO (verde)")
        print("2. Presiona 'G' y haz CLICK -> Coloca META (rojo)")
        print("3. CLICK y ARRASTRA el mouse -> Dibuja OBSTACULOS (negro)")
        print("4. Presiona 'D' -> Activa MODO DINAMICO (agregar obstaculos durante ejecucion)")
        print("5. Presiona ESPACIO -> Ejecuta el algoritmo\n")
        print("ALGORITMOS DISPONIBLES:")
        print("   [1] Dijkstra   - Explora uniformemente (garantiza optimo)")
        print("   [2] A*         - Usa heuristica (mas rapido, default)")
        print("   [3] D*-lite    - Replanificacion incremental (mejor con dinamicos)")
        print("   [4] BMSSP      - Busqueda bidireccional\n")
        print("MODO DINAMICO:")
        print("   Activa con [D] y agrega obstaculos DURANTE la ejecucion")
        print("   D*-lite y BMSSP pueden replantear, A*/Dijkstra se afectan!\n")
        print("CONTROLES:")
        print("   [FLECHA ARRIBA/ABAJO] = Ajustar velocidad")
        print("   [R] = Reiniciar camino  [C] = Limpiar todo  [Q] = Salir\n")
        print(f"Grid actual: {GRID_WIDTH}x{GRID_HEIGHT} ({GRID_WIDTH*GRID_HEIGHT} nodos)")
        print(f"Algoritmo: {self.current_algorithm.upper()}")
        print("=" * 75)
        print("\nTIP: Prueba modo dinamico con D*-lite vs A*!\n")

        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def main():
    """Entry point"""
    app = PathfindingVisualizer()
    app.run()


if __name__ == "__main__":
    main()
