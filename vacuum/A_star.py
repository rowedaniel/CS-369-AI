from agent_common import dir_values


def manhattan_distance(start, goal):
    return sum(abs(s - g) for s, g in zip(start, goal))


def A_star(world, start, goal, dirt_bias=0.1):
    def heuristic(pos):
        return manhattan_distance(pos, goal)

    width = len(world)
    height = len(world[0])

    def get_neighbors(pos):
        x, y = pos
        return (
            (x + dx, y + dy)
            for dx, dy in dir_values
            if 0 <= x + dx < width
            and 0 <= y + dy < height
            and world[x + dx][y + dy] != "wall"
        )

    frontier = [(start, 0, heuristic(start), [start])]
    explored = set()
    while frontier:
        current, cost_so_far, _, path_to_current = frontier.pop(0)

        if current == goal:
            return path_to_current

        if current in explored:
            continue

        explored.add(current)
        for neighbor in get_neighbors(current):
            if neighbor in explored:
                continue

            # update costs
            # g(start to n-1) + g(n-1 to n)
            new_cost = (
                cost_so_far
                + 1
                - dirt_bias * (world[neighbor[0]][neighbor[1]] == "dirt")
            )
            heuristic_value = heuristic(neighbor)  # h(n)

            # update frontier
            path_to_neighbor = [*path_to_current, neighbor]
            frontier.append((neighbor, new_cost, heuristic_value, path_to_neighbor))

            # sort by total cost
            frontier.sort(key=lambda node: node[1] + node[2])  # f = g + h
    return None
