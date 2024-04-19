from agent_common import dir_values, directions
from itertools import combinations
import random
from mapping_memory import MappingMemory
from heapq import heappop, heappush


def sight_5_random_agent(percept):
    if percept[0] == "dirt":
        return "clean"
    elif percept[1] == "dirt":
        return "north"
    elif percept[2] == "dirt":
        return "south"
    elif percept[3] == "dirt":
        return "east"
    elif percept[4] == "dirt":
        return "west"
    else:
        return random.choice(directions)


def mapping_agent_sight_5(percept, mappingMemory):
    expected_tile = mappingMemory.get_next_expected_tile_type()
    mappingMemory.move()

    if expected_tile == "dirt":
        # at the target node!
        mappingMemory.expand_target()

    # update surroundings
    x, y = mappingMemory.get_root_pos()
    for direction, node_type in enumerate(percept[1:]):
        dx, dy = dir_values[direction]
        xp, yp = (x + dx, y + dy)
        if node_type is None:
            mappingMemory.set_pos_type((xp, yp), "wall")
        else:
            mappingMemory.set_pos_type((xp, yp), node_type)

    if expected_tile == "dirt":
        # at the target node!
        mappingMemory.retarget()
        mappingMemory.set_root_type("clean")
        return "clean"
    else:
        # update last timestep's move
        mappingMemory.move()

    new_dir = mappingMemory.get_next_direction()
    if new_dir == -1:
        # TODO: handle this better
        return sight_5_random_agent(percept)
    return directions[new_dir]


class TSP:
    def __init__(self, size, walls, cost_func="dirt"):
        self.vertices = tuple(
            (x, y) for x in range(size) for y in range(size) if (x, y) not in walls
        )
        self.walls = walls
        self.size = size
        self.explored_cost = size * 2
        if cost_func == "dirt":
            self.cost_func = self.dirt_cost_func
        else:
            self.cost_func = self.actions_cost_func

        self.inf = 10 * self.size**4
        self.metric = [
            [0 if i == j else None for i in range(len(self.vertices))]
            for j in range(len(self.vertices))
        ]

    def num_dirty_tiles(self):
        return len(self.vertices)

    def dirt_cost_func(self, path):
        dirt = self.num_dirty_tiles()
        get_distance = self.get_distance

        cost = sum(
            map(
                lambda i: (get_distance(path[i - 1][0], path[i][0]) + 1) * (dirt - i)
                + 1,
                range(1, len(path)),
            )
        )
        return cost

    def actions_cost_func(self, path):
        cost = sum(
            map(
                lambda i: self.get_distance(path[i - 1][0], path[i][0]) + 1,
                range(1, len(path)),
            )
        )
        return cost

    def manhattan_distance(self, i, j):
        return sum((abs(a - b) for a, b in zip(self.vertices[i], self.vertices[j])))

    def get_distance(self, i, j):
        metric = self.metric
        vertices = self.vertices
        get_neighbors = self.get_neighbors

        if metric[i][j] is not None:
            return metric[i][j]

        frontier = [(0, i, vertices[i])]
        explored = set()
        while frontier:
            dist, current_i, node = heappop(frontier)
            if current_i == j:
                return dist
            for neigh in get_neighbors(node):
                neigh_i = vertices.index(neigh)
                if neigh_i in explored:
                    continue
                explored.add(neigh_i)

                old_distance = metric[i][neigh_i]
                current_distance = dist + 1

                if old_distance is None or current_distance < old_distance:
                    metric[i][neigh_i] = current_distance
                    metric[neigh_i][i] = current_distance

                heappush(frontier, (current_distance, neigh_i, neigh))
        return self.inf

    def get_neighbors(self, pos):
        x, y = pos
        return (
            (x + dx, y + dy)
            for dx, dy in dir_values
            if 0 <= x + dx < self.size
            and 0 <= y + dy < self.size
            and (x + dx, y + dy) not in self.walls
        )

    def greedy(self, start):
        path = []
        current = (self.vertices.index(start), start)
        while len(path) < len(self.vertices):
            path.append(current)
            current = min(
                enumerate(self.vertices),
                key=lambda x: self.inf * (x in path)
                + self.get_distance(start[0], x[0]),
            )
        return path

    def search_mapper(self, start):
        best_path = []
        best_cost = self.inf
        for likely_border_weight in (0,):
            for possible_border_weight in (0,):
                for singleton_weight in (0,):
                    candidate_path = self.use_mapper(
                        start,
                        mapper_params={
                            "path_weight": 1,
                            "likely_border_weight": likely_border_weight,
                            "possible_border_weight": possible_border_weight,
                            "singleton_weight": singleton_weight,
                        },
                    )
                    candidate_cost = self.cost_func(candidate_path)
                    if candidate_cost < best_cost:
                        best_path = candidate_path
                        best_cost = candidate_cost
        return best_path

    def use_mapper(
        self,
        start,
        mapper_params={
            "path_weight": 1,
            "likely_border_weight": 100,
            "possible_border_weight": 4,
            "singleton_weight": 1,
        },
    ):
        path = []
        mapper = MappingMemory(**mapper_params)
        vertex_len = len(self.vertices)

        agent_pos = start
        explored = set()
        max_steps = 2000
        for _ in range(max_steps):
            x, y = agent_pos
            if agent_pos in self.vertices:
                agent_pos_i = self.vertices.index(agent_pos)
                next_path_node = (agent_pos_i, agent_pos)
                if next_path_node not in path:
                    path.append(next_path_node)
                    if len(path) >= vertex_len:
                        return path

            percept = [
                "wall"
                if pos not in self.vertices
                else ("clean" if pos in explored else "dirt")
                for pos in map(
                    lambda dir_val: (x + dir_val[0], y + dir_val[1]),
                    ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)),
                )
            ]

            action = mapping_agent_sight_5(percept, mapper)
            if action == "clean":
                explored.add(agent_pos)
            else:
                x, y = agent_pos
                new_dir = directions.index(action)
                dx, dy = dir_values[new_dir]
                xp, yp = (x + dx, y + dy)
                if (xp, yp) in self.vertices:
                    agent_pos = (x + dx, y + dy)

        # for missed in enumerate(self.vertices):
        #     if missed in path:
        #         continue
        #     path.append(missed)
        return path

    def OPT2_search(self, path):
        """
        Pick n sets of two vertices, and reverse the path between them
        """
        best_path = None
        best_path_cost = self.inf

        combos = combinations(range(1, len(path)), 2)
        rev = reversed
        cost_func = self.cost_func

        for combo in combos:
            if random.random() < 0.5:
                continue
            # generate new possible paths by cutting vertices, and stiching back together
            a, b = combo
            if b - a == 1:
                continue

            path[a:b] = rev(path[a:b])
            new_cost = cost_func(path)
            if new_cost < best_path_cost:
                best_path = path.copy()
                best_path_cost = new_cost

            # undo swap
            path[a:b] = rev(path[a:b])

        return best_path, best_path_cost

    def OPT3_search(self, path):
        best_path = None
        best_path_cost = self.inf

        directions = [lambda x: x, reversed]
        path_recombinations = [
            ((i, section), (j, second_dir), (k, third_dir))
            for i, section in enumerate(directions)
            for j, second_dir in enumerate(directions)
            for k, third_dir in enumerate(directions)
            if not (i == 0 and j == 0 and k == 0)
        ]

        combos = filter(
            lambda combo: (combo[1] - combo[0] == 1 or combo[2] - combo[1] == 1)
            and combo[2] - combo[0] > 2,
            combinations(range(1, len(path)), 3),
        )

        n = 0
        for combo in combos:
            n += 1
            # generate new possible paths by cutting vertices, and stiching back together
            for (i, section), (j, second_dir), (k, third_dir) in path_recombinations:
                a, b, c = combo
                if i == 0 and ((b - a == 1 and k == 0) or (c - b == 1 and j == 0)):
                    continue

                new_path = path.copy()
                section_1 = new_path[a:b]
                section_2 = new_path[b:c]
                section_2, section_1 = section([section_1, section_2])
                new_path[b:c] = third_dir(section_2)
                new_path[a:b] = second_dir(section_1)

                new_cost = self.cost_func(new_path)
                if new_cost < best_path_cost:
                    best_path = new_path
                    best_path_cost = new_cost

        return best_path, best_path_cost

    def search_path_OPT3(self, start, n=8):
        """
        simulated annealing with 3opt
        """
        initial_temp = 1
        temp = initial_temp
        cooling_rate = 0.95

        initial_path = self.use_mapper(start)

        path, cost = self.OPT3_search(initial_path)
        for i in range(n):
            candidate_path, candidate_cost = self.OPT3_search(path)

            if candidate_cost < cost or random.random() < temp:
                path, cost = candidate_path, candidate_cost
            else:
                break
            temp *= cooling_rate

        return path

    def search_path_OPT2(self, start, n=8, score_cutoff=-1):
        """
        simulated annealing with 2opt
        """
        initial_temp = 1
        temp = initial_temp
        cooling_rate = 0.95

        initial_path = self.search_mapper(start)

        path, cost = initial_path, self.cost_func(initial_path)
        for _ in range(n):
            if cost < score_cutoff:
                break
            candidate_path, candidate_cost = self.OPT2_search(path)

            if candidate_cost < cost or random.random() < temp:
                path, cost = candidate_path, candidate_cost
            else:
                break
            temp *= cooling_rate

        return path
