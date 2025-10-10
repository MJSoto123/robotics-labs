"""
A* Algorithm with step-by-step visualization
"""
import heapq


def heuristic(node, goal):
    """
    Calculate Manhattan distance heuristic

    Args:
        node: Current node
        goal: Goal node

    Returns:
        int: Manhattan distance
    """
    return abs(node.x - goal.x) + abs(node.y - goal.y)


def astar_generator(grid, start, goal):
    """
    Generator that yields each step of A* algorithm

    Yields:
        tuple: (current_node, open_set, closed_set, path_found)
    """
    # Reset all nodes
    for x in range(grid.width):
        for y in range(grid.height):
            node = grid.nodes[x][y]
            node.g_cost = float('inf')
            node.h_cost = heuristic(node, goal)
            node.f_cost = float('inf')
            node.parent = None
            if not node.is_start and not node.is_goal and not node.is_obstacle:
                node.state = "unvisited"

    # Initialize start node
    start.g_cost = 0
    start.h_cost = heuristic(start, goal)
    start.f_cost = start.g_cost + start.h_cost

    # Priority queue: (f_cost, counter, node)
    counter = 0
    open_set = []
    heapq.heappush(open_set, (start.f_cost, counter, start))

    # Track visited nodes
    open_set_hash = {start}
    closed_set = set()

    while open_set:
        # Get node with lowest f_cost
        current_f, _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)

        # Mark as current
        if current != start and current != goal:
            current.state = "current"

        # Yield current state for visualization
        yield (current, open_set_hash.copy(), closed_set.copy(), False)

        # Check if we reached the goal
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node:
                path.append(node)
                node = node.parent
            path.reverse()

            # Mark path
            for node in path:
                if node != start and node != goal:
                    node.state = "path"

            yield (current, open_set_hash.copy(), closed_set.copy(), True)
            return

        # Add to closed set
        closed_set.add(current)
        if current != start and current != goal:
            current.state = "closed"

        # Check all neighbors
        neighbors = grid.get_neighbors(current)

        for neighbor in neighbors:
            if neighbor in closed_set or neighbor.is_obstacle:
                continue

            # Calculate tentative g_cost
            tentative_g = current.g_cost + 1  # Assuming uniform cost of 1

            if tentative_g < neighbor.g_cost:
                # Found a better path
                neighbor.parent = current
                neighbor.g_cost = tentative_g
                neighbor.h_cost = heuristic(neighbor, goal)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                if neighbor not in open_set_hash:
                    counter += 1
                    heapq.heappush(open_set, (neighbor.f_cost, counter, neighbor))
                    open_set_hash.add(neighbor)

                    if neighbor != start and neighbor != goal:
                        neighbor.state = "open"

    # No path found
    yield (None, set(), closed_set, False)


def reconstruct_path(goal):
    """Reconstruct path from goal to start using parent pointers"""
    path = []
    current = goal
    while current:
        path.append(current)
        current = current.parent
    return path[::-1]
