"""
Renderer for grid visualization
"""
import pygame
from . import colors


class Renderer:
    """Handles drawing the grid and nodes"""

    def __init__(self, grid, screen):
        self.grid = grid
        self.screen = screen
        self.font = pygame.font.Font(None, 24)
        self.info_font = pygame.font.Font(None, 18)  # Reducido de 20 a 18

    def draw_grid(self):
        """Draw the entire grid with all nodes"""
        # Draw all cells
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                node = self.grid.nodes[x][y]
                self.draw_cell(node)

        # Draw grid lines
        self.draw_grid_lines()

    def draw_cell(self, node):
        """Draw a single cell"""
        x_pos = node.x * self.grid.cell_size
        y_pos = node.y * self.grid.cell_size

        # Get color based on node state
        color = colors.get_node_color(node)

        # Draw filled rectangle
        rect = pygame.Rect(x_pos, y_pos, self.grid.cell_size, self.grid.cell_size)
        pygame.draw.rect(self.screen, color, rect)

    def draw_grid_lines(self):
        """Draw grid lines"""
        width_px = self.grid.width * self.grid.cell_size
        height_px = self.grid.height * self.grid.cell_size

        # Vertical lines
        for x in range(self.grid.width + 1):
            x_pos = x * self.grid.cell_size
            pygame.draw.line(
                self.screen,
                colors.GRID_LINES,
                (x_pos, 0),
                (x_pos, height_px),
                1
            )

        # Horizontal lines
        for y in range(self.grid.height + 1):
            y_pos = y * self.grid.cell_size
            pygame.draw.line(
                self.screen,
                colors.GRID_LINES,
                (0, y_pos),
                (width_px, y_pos),
                1
            )

    def draw_path(self, path):
        """Draw the final path"""
        for node in path:
            if not node.is_start and not node.is_goal:
                node.state = "path"
            self.draw_cell(node)

    def draw_info_panel(self, info_text, y_offset):
        """Draw information panel at the bottom"""
        panel_y = self.grid.height * self.grid.cell_size + 10
        text_surface = self.info_font.render(info_text, True, colors.TEXT_COLOR)
        self.screen.blit(text_surface, (10, panel_y + y_offset))

    def draw_instructions(self, current_algorithm="astar", is_running=False, dynamic_mode=False,
                          nodes_explored=0, elapsed_time=0, path_length=0):
        """Draw instructions and metrics on screen"""
        algo_names = {
            "dijkstra": "DIJKSTRA",
            "astar": "A*",
            "dstar_lite": "D*-LITE",
            "bmssp": "BMSSP"
        }
        algo_name = algo_names.get(current_algorithm, current_algorithm.upper())

        # Dynamic mode indicator
        dynamic_status = "[D] DINAMICO ON" if dynamic_mode else "[D] Dinamico off"

        instructions = [
            f"Algoritmo: {algo_name}   [1] Dijkstra   [2] A*   [3] D*-lite   [4] BMSSP   {dynamic_status}",
            f"Metricas: Nodos={nodes_explored}  Tiempo={elapsed_time:.2f}s  Longitud={path_length}",
            "CLICK: Obstaculos | S: Inicio | G: Meta | ESPACIO: Ejecutar | R: Reiniciar | C: Limpiar",
            f"Estado: {'EJECUTANDO... (Click para agregar obstaculos)' if is_running and dynamic_mode else 'EJECUTANDO...' if is_running else 'Listo'}"
        ]

        panel_y = self.grid.height * self.grid.cell_size + 10
        line_height = 30  # Espaciado entre líneas
        for i, instruction in enumerate(instructions):
            self.draw_info_panel(instruction, i * line_height)

    def highlight_current(self, node):
        """Highlight the current node being processed"""
        if not node.is_start and not node.is_goal:
            node.state = "current"
        self.draw_cell(node)
