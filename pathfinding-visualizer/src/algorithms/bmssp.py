"""
BMSSP (Bounded Multi-Source Shortest Path) adapted for 2D
Bidirectional search from start and goal simultaneously
"""
import heapq


def heuristic(node, goal):
    """Manhattan distance heuristic"""
    return abs(node.x - goal.x) + abs(node.y - goal.y)


def bmssp_generator(grid, start, goal):
    """
    Generator that yields each step of bidirectional BMSSP

    Searches from both start and goal simultaneously until they meet.
    This is more efficient than unidirectional search.

    Yields:
        tuple: (current_node, open_set, closed_set, path_found)
    """
    # Reset all nodes
    for x in range(grid.width):
        for y in range(grid.height):
            node = grid.nodes[x][y]
            node.g_cost = float('inf')
            node.g_cost_reverse = float('inf')  # Cost from goal
            node.parent = None
            node.parent_reverse = None
            node.from_start = False
            node.from_goal = False
            if not node.is_start and not node.is_goal and not node.is_obstacle:
                node.state = "unvisited"

    # Initialize start and goal
    start.g_cost = 0
    goal.g_cost_reverse = 0

    # Two priority queues (forward and backward)
    counter = 0
    open_forward = []
    open_backward = []
    heapq.heappush(open_forward, (0, counter, start))
    counter += 1
    heapq.heappush(open_backward, (0, counter, goal))
    counter += 1

    open_forward_hash = {start}
    open_backward_hash = {goal}

    closed_forward = set()
    closed_backward = set()

    # Best path found so far
    best_path_cost = float('inf')
    meeting_node = None

    # Alternate between forward and backward search
    forward_turn = True

    while open_forward or open_backward:
        # Alternate search direction
        if forward_turn and open_forward:
            # Forward search (from start)
            current_cost, _, current = heapq.heappop(open_forward)
            open_forward_hash.discard(current)
            current.from_start = True

            if current != start and current != goal:
                current.state = "open"

            # Yield current state
            all_open = open_forward_hash.union(open_backward_hash)
            all_closed = closed_forward.union(closed_backward)
            yield (current, all_open.copy(), all_closed.copy(), False)

            # Check if we've met the backward search
            if current in closed_backward or current in open_backward_hash:
                # Potential meeting point
                total_cost = current.g_cost + current.g_cost_reverse
                if total_cost < best_path_cost:
                    best_path_cost = total_cost
                    meeting_node = current

            # Terminate if current cost exceeds best path
            if current.g_cost >= best_path_cost:
                break

            # Add to closed set
            closed_forward.add(current)
            if current != start and current != goal:
                current.state = "closed"

            # Expand neighbors
            neighbors = grid.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in closed_forward or neighbor.is_obstacle:
                    continue

                tentative_g = current.g_cost + 1

                if tentative_g < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g

                    if neighbor not in open_forward_hash:
                        heapq.heappush(open_forward, (neighbor.g_cost, counter, neighbor))
                        open_forward_hash.add(neighbor)
                        counter += 1

                        if neighbor != start and neighbor != goal:
                            neighbor.state = "open"

        elif not forward_turn and open_backward:
            # Backward search (from goal)
            current_cost, _, current = heapq.heappop(open_backward)
            open_backward_hash.discard(current)
            current.from_goal = True

            if current != start and current != goal:
                current.state = "open"

            # Yield current state
            all_open = open_forward_hash.union(open_backward_hash)
            all_closed = closed_forward.union(closed_backward)
            yield (current, all_open.copy(), all_closed.copy(), False)

            # Check if we've met the forward search
            if current in closed_forward or current in open_forward_hash:
                total_cost = current.g_cost + current.g_cost_reverse
                if total_cost < best_path_cost:
                    best_path_cost = total_cost
                    meeting_node = current

            # Terminate if current cost exceeds best path
            if current.g_cost_reverse >= best_path_cost:
                break

            # Add to closed set
            closed_backward.add(current)
            if current != start and current != goal:
                current.state = "closed"

            # Expand neighbors
            neighbors = grid.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in closed_backward or neighbor.is_obstacle:
                    continue

                tentative_g = current.g_cost_reverse + 1

                if tentative_g < neighbor.g_cost_reverse:
                    neighbor.parent_reverse = current
                    neighbor.g_cost_reverse = tentative_g

                    if neighbor not in open_backward_hash:
                        heapq.heappush(open_backward, (neighbor.g_cost_reverse, counter, neighbor))
                        open_backward_hash.add(neighbor)
                        counter += 1

                        if neighbor != start and neighbor != goal:
                            neighbor.state = "open"

        # Switch direction
        forward_turn = not forward_turn

        # Check if we found a path
        if meeting_node and best_path_cost < float('inf'):
            # Reconstruct path
            # Forward: start -> meeting
            forward_path = []
            current = meeting_node
            while current:
                forward_path.append(current)
                current = current.parent
            forward_path.reverse()

            # Backward: meeting -> goal
            backward_path = []
            current = meeting_node.parent_reverse
            while current:
                backward_path.append(current)
                current = current.parent_reverse

            # Combine paths
            path = forward_path + backward_path

            # Mark path
            for node in path:
                if node != start and node != goal:
                    node.state = "path"

            all_open = open_forward_hash.union(open_backward_hash)
            all_closed = closed_forward.union(closed_backward)
            yield (meeting_node, all_open, all_closed, True)
            return

    # No path found
    all_open = open_forward_hash.union(open_backward_hash)
    all_closed = closed_forward.union(closed_backward)
    yield (None, all_open, all_closed, False)
