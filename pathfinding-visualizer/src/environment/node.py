"""
Node/Cell class for the grid
"""

class Node:
    """Represents a cell in the grid"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_obstacle = False
        self.is_start = False
        self.is_goal = False

        # Algorithm costs
        self.g_cost = float('inf')  # Cost from start
        self.h_cost = 0              # Heuristic cost to goal
        self.f_cost = float('inf')  # Total cost (g + h)
        self.parent = None

        # D*-lite specific
        self.rhs = float('inf')     # One-step lookahead value

        # BMSSP specific (bidirectional search)
        self.g_cost_reverse = float('inf')  # Cost from goal
        self.parent_reverse = None
        self.from_start = False
        self.from_goal = False

        # Visualization state
        self.state = "unvisited"  # unvisited, open, closed, path

    def reset(self):
        """Reset node for new algorithm run"""
        if not self.is_obstacle and not self.is_start and not self.is_goal:
            self.state = "unvisited"
        self.g_cost = float('inf')
        self.h_cost = 0
        self.f_cost = float('inf')
        self.parent = None
        self.rhs = float('inf')
        self.g_cost_reverse = float('inf')
        self.parent_reverse = None
        self.from_start = False
        self.from_goal = False

    def __lt__(self, other):
        """For priority queue comparison"""
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        """Check if two nodes are equal"""
        if other is None:
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        """For use in sets and dictionaries"""
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Node({self.x}, {self.y})"
