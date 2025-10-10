"""
Grid class for the environment
"""
from .node import Node


class Grid:
    """2D Grid for pathfinding visualization"""

    def __init__(self, width, height, cell_size=20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.start = None
        self.goal = None

        # Create 2D array of nodes
        self.nodes = [[Node(x, y) for y in range(height)] for x in range(width)]

    def get_node(self, x, y):
        """Get node at position (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.nodes[x][y]
        return None

    def set_start(self, x, y):
        """Set start position"""
        if self.start:
            self.start.is_start = False
        node = self.get_node(x, y)
        if node and not node.is_obstacle and not node.is_goal:
            self.start = node
            node.is_start = True
            node.state = "start"
            return True
        return False

    def set_goal(self, x, y):
        """Set goal position"""
        if self.goal:
            self.goal.is_goal = False
        node = self.get_node(x, y)
        if node and not node.is_obstacle and not node.is_start:
            self.goal = node
            node.is_goal = True
            node.state = "goal"
            return True
        return False

    def toggle_obstacle(self, x, y):
        """Toggle obstacle at position"""
        node = self.get_node(x, y)
        if node and not node.is_start and not node.is_goal:
            node.is_obstacle = not node.is_obstacle
            if node.is_obstacle:
                node.state = "obstacle"
            else:
                node.state = "unvisited"
            return True
        return False

    def clear_obstacles(self):
        """Remove all obstacles"""
        for x in range(self.width):
            for y in range(self.height):
                node = self.nodes[x][y]
                if node.is_obstacle:
                    node.is_obstacle = False
                    node.state = "unvisited"

    def reset_path(self):
        """Reset algorithm state but keep start, goal, and obstacles"""
        for x in range(self.width):
            for y in range(self.height):
                self.nodes[x][y].reset()

    def get_neighbors(self, node, diagonal=False):
        """Get neighboring nodes (4 or 8 directions)"""
        neighbors = []
        directions = [
            (0, -1),   # Up
            (1, 0),    # Right
            (0, 1),    # Down
            (-1, 0),   # Left
        ]

        if diagonal:
            directions += [
                (-1, -1),  # Up-left
                (1, -1),   # Up-right
                (1, 1),    # Down-right
                (-1, 1),   # Down-left
            ]

        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            neighbor = self.get_node(nx, ny)
            if neighbor and not neighbor.is_obstacle:
                neighbors.append(neighbor)

        return neighbors

    def is_valid_position(self, x, y):
        """Check if position is valid and walkable"""
        node = self.get_node(x, y)
        return node is not None and not node.is_obstacle

    def get_path(self, end_node):
        """Reconstruct path from end node to start"""
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]  # Reverse to get start -> goal
