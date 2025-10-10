"""
D*-lite Algorithm with step-by-step visualization
Incremental pathfinding with replanning capability
"""
import heapq


def heuristic(node, goal):
    """Manhattan distance heuristic"""
    return abs(node.x - goal.x) + abs(node.y - goal.y)


def calculate_key(node, start, k_m):
    """Calculate priority key for node"""
    min_val = min(node.g_cost, node.rhs)
    return (min_val + heuristic(node, start) + k_m, min_val)


def dstar_lite_generator(grid, start, goal):
    """
    Generator that yields each step of D*-lite algorithm

    Simplified version that searches backward from goal to start.

    Yields:
        tuple: (current_node, open_set, closed_set, path_found)
    """
    # Initialize all nodes
    for x in range(grid.width):
        for y in range(grid.height):
            node = grid.nodes[x][y]
            node.g_cost = float('inf')
            node.h_cost = heuristic(node, start)
            node.f_cost = float('inf')
            node.parent = None
            if not node.is_start and not node.is_goal and not node.is_obstacle:
                node.state = "unvisited"

    # Initialize goal (searching backward from goal to start)
    goal.g_cost = 0
    goal.f_cost = goal.g_cost + heuristic(goal, start)

    counter = 0
    open_set = []
    open_set_hash = set()

    # Add goal to open set
    heapq.heappush(open_set, (goal.f_cost, counter, goal))
    open_set_hash.add(goal)
    counter += 1

    closed_set = set()

    # Compute shortest path (backward from goal to start)
    while open_set:
        # Get node with lowest f_cost
        current_f, _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)

        if current != start and current != goal:
            current.state = "current"

        # Yield current state
        yield (current, open_set_hash.copy(), closed_set.copy(), False)

        # Stop if we've reached the start
        if current == start:
            # Reconstruct path from start to goal
            path = []
            node = start
            while node:
                path.append(node)
                node = node.parent

            # Mark path
            for node in path:
                if node != start and node != goal:
                    node.state = "path"

            yield (start, set(), closed_set, True)
            return

        # Add to closed set
        closed_set.add(current)
        if current != start and current != goal:
            current.state = "closed"

        # Expand neighbors
        neighbors = grid.get_neighbors(current)
        for neighbor in neighbors:
            if neighbor in closed_set or neighbor.is_obstacle:
                continue

            # Calculate cost (backward search)
            tentative_g = current.g_cost + 1

            if tentative_g < neighbor.g_cost:
                # Found better path
                neighbor.parent = current
                neighbor.g_cost = tentative_g
                neighbor.h_cost = heuristic(neighbor, start)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (neighbor.f_cost, counter, neighbor))
                    open_set_hash.add(neighbor)
                    counter += 1

                    if neighbor != start and neighbor != goal:
                        neighbor.state = "open"

    # No path found
    yield (None, set(), closed_set, False)
