"""
Color scheme for visualization
"""

# Grid colors
BACKGROUND = (255, 255, 255)      # White
GRID_LINES = (200, 200, 200)      # Light gray

# Cell states
UNVISITED = (255, 255, 255)       # White
OBSTACLE = (40, 40, 40)           # Dark gray
START = (0, 200, 0)               # Green
GOAL = (200, 0, 0)                # Red

# Algorithm visualization
OPEN_SET = (173, 216, 230)        # Light blue
CLOSED_SET = (100, 149, 237)      # Medium blue
CURRENT = (255, 215, 0)           # Gold
PATH = (255, 255, 0)              # Yellow

# Special states (for BMSSP)
PIVOT = (255, 0, 255)             # Magenta
PARTITION = (255, 165, 0)         # Orange

# UI colors
TEXT_COLOR = (0, 0, 0)            # Black
BUTTON_COLOR = (70, 130, 180)     # Steel blue
BUTTON_HOVER = (100, 149, 237)    # Cornflower blue
BUTTON_TEXT = (255, 255, 255)     # White


def get_node_color(node):
    """Get color for a node based on its state"""
    if node.is_start:
        return START
    if node.is_goal:
        return GOAL
    if node.is_obstacle:
        return OBSTACLE
    if node.state == "path":
        return PATH
    if node.state == "current":
        return CURRENT
    if node.state == "closed":
        return CLOSED_SET
    if node.state == "open":
        return OPEN_SET
    if node.state == "pivot":
        return PIVOT
    if node.state == "partition":
        return PARTITION

    return UNVISITED
