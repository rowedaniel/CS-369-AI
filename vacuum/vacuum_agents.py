"""
Use this script file to define your robot vacuum agents.

The run function will generate a map showing the animation of the robot, and return the output of the loss function at the end of the run. The many_runs function will run the simulation multiple times without the visualization and return the average loss.

You will need to implement a run_all function, which executes the many_runs function for all 12 of your agents (with the correct parameters) and sums up their returned losses as a single value. Your run_all function should use the following parameters for each agent: map_width=20, max_steps=50000 runs=100.
"""

from vacuum import run, many_runs
import random
from itertools import combinations
from heapq import heappop, heappush

directions = ["north", "south", "east", "west"]
dir_values = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def opposite_direction(direction):
    return direction ^ 1


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


class MappingNode:
    def __init__(
        self,
        parent=None,
        pos=(0, 0),
        cost=-1,
    ):
        """
        node on the map.
        Arguments:
        - parent (MappingNode): route to reach this node
        - pos ((x, y)): position of this node
        - cost: path cost to this node
        """
        self.parent = parent
        self.children = [None for i in range(4)]
        self.pos = pos
        self.cost = cost
        self.direction = None
        self.node_type = "dirt"

    def set_parent(self, parent, direction):
        if parent is None:
            print("attempting to set a null parent")
            return
        if parent in self.children:
            print("loop detected!")
        if parent.get_parent() == self:
            print("attempting to create a circular parent chain")
            raise ValueError(height)
        if self.children[opposite_direction(direction)] is not None:
            if self.children[opposite_direction(direction)] is not parent:
                raise ValueError("confusing parent/child situation happening")

        if self.parent is not None:
            self.parent.remove_child(self.direction)
        self.parent = parent
        parent.add_child(self, direction)
        self.set_direction(direction)

    def remove_parent(self):
        if self.parent is None:
            return
        self.parent.remove_child(self.direction)
        self.parent = None
        self.direction = -1

    def get_parent(self):
        return self.parent

    def add_child(self, child, direction):
        if self.children[direction] is not None:
            if self.children[direction] != child:
                print("attempting to set different child when one already exists")
        if child is self.parent:
            print("attempting to add child which is the parent")
            raise ValueError("AAAA")
        self.children[direction] = child

    def remove_child(self, direction):
        self.children[direction] = None

    def remove_all_children(self):
        for i in range(len(self.children)):
            self.children[i] = None

    def get_children(self):
        return self.children

    def has_child(self, direction):
        return self.children[direction] is not None

    def get_pos(self):
        return self.pos

    def get_child_positions(self):
        if self.node_type == "wall":
            return []
        x, y = self.pos
        return [(i, (x + ox, y + oy)) for i, (ox, oy) in enumerate(dir_values)]

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        if self.node_type == "wall":
            # TODO: handle this more elegantly?
            cost = 99999
        for child in self.children:
            if child is None:
                continue
            child.set_cost(cost + 1)
        self.cost = cost

    def get_type(self):
        return self.node_type

    def set_type(self, node_type):
        self.node_type = node_type

    def get_direction(self):
        return self.direction

    def set_direction(self, direction):
        self.direction = direction


class MappingMemory:
    def __init__(
        self,
        path_weight=1,
        likely_border_weight=100,
        possible_border_weight=2,
        singleton_weight=1,
    ):
        self.reset(
            path_weight, likely_border_weight, possible_border_weight, singleton_weight
        )

        # caches to save time
        self.border = {}

    def reset(
        self,
        path_weight,
        likely_border_weight,
        possible_border_weight,
        singleton_weight,
    ):

        # hyperparams
        self.path_weight = path_weight
        self.likely_border_weight = likely_border_weight
        self.possible_border_weight = possible_border_weight
        self.singleton_weight = singleton_weight

        # O(1) stuff:
        self.next_dir = -1
        self.current_target = MappingNode(cost=0)
        self.root = self.current_target

        # O(n) stuff:
        self.frontier = []  # max size: map size + 2 in both dimensions (height)
        self.explored = {
            (0, 0): self.current_target
        }  # some overlap with frontier, but always smaller than map size.

    def cost_function(self, node):
        path_cost = node.get_cost()

        # check if the node could be part of the border
        pos = node.get_pos()
        if pos in self.border:
            possible_border_cost = 1
            known_border_cost = self.border[pos]
        else:
            possible_border_cost = 0
            known_border_cost = 0

        singleton_cost = (
            len(
                tuple(
                    filter(
                        lambda pos: pos[1] not in self.explored
                        or self.explored[pos[1]].get_type() == "dirt",
                        node.get_child_positions(),
                    )
                )
            )
            / 4
        )

        return (
            path_cost * self.path_weight
            + known_border_cost * self.likely_border_weight
            + possible_border_cost * self.possible_border_weight
            + singleton_cost * self.singleton_weight
        )

    def update_borders(self):
        min_x = min_y = max_x = max_y = 0
        for x, y in self.explored:
            min_x = min(x, min_x)
            max_x = max(x, max_x)
            min_y = min(y, min_y)
            max_y = max(y, max_y)
        borders = [[], [], [], []]
        chances = [0, 0, 0, 0]
        for (x, y), n in self.explored.items():
            i = -1
            if x == min_x:
                i = 0
            if x == max_x:
                i = 1
            if y == min_y:
                i = 2
            if y == max_y:
                i = 3
            if i == -1:
                continue
            borders[i].append((x, y))
            node_type = n.get_type()
            if node_type == "wall":
                if chances[i] != -1:
                    chances[i] += 1
            elif node_type != "dirt":
                chances[i] = -1

        chances = [c / len(borders[i]) if c > 0 else 0 for i, c in enumerate(chances)]
        self.border = {
            pos: chances[i] for i in range(len(borders)) for pos in borders[i]
        }

    def get_next_direction(self):
        if self.current_target is self.root:
            return -1
        node = self.current_target
        direction = -1

        while node is not self.root:
            parent = node.get_parent()
            if parent is None:
                raise ValueError(
                    "Reached a parent of 'None' when trying to find next direction"
                )
            direction = node.get_direction()
            node = parent
        self.next_dir = direction
        return direction

    def get_next_expected_tile_type(self):
        if self.next_dir == -1:
            return self.root.get_type()
        if self.root.get_children()[self.next_dir] is None:
            raise ValueError("dunno what next tile should be")
        return self.root.get_children()[self.next_dir].get_type()

    def get_root_pos(self):
        return self.root.get_pos()

    def set_pos_type(self, pos, node_type):
        if pos not in self.explored:
            return
        self.explored[pos].set_type(node_type)

    def set_target_type(self, node_type):
        if self.current_target is None:
            raise ValueError("trying to set target when no target exists")
        self.current_target.set_type(node_type)

    def set_root_type(self, node_type):
        self.root.set_type(node_type)

    def move(self):
        direction = self.next_dir
        self.next_dir = -1

        if direction == -1:
            return
        x, y = self.root.get_pos()
        self.prev_dir = direction
        dx, dy = dir_values[direction]
        self.swap_root((x + dx, y + dy))

    def swap_root(self, pos):
        if not self.explored[pos]:
            raise ValueError("attempting to swap root with an unexplored location")

        new_root = self.explored[pos]
        op_dir = opposite_direction(new_root.get_direction())

        # remove old pointers
        new_root.remove_parent()

        # add new pointers
        if self.root.parent is not None:
            print("root with a parent???")
            raise ValueError("AAAAA")

        self.root.set_parent(new_root, op_dir)

        if new_root.parent is not None:
            raise ValueError("root with a parent?")
        if new_root in self.root.get_children():
            raise ValueError("root is a child?")

        self.root = new_root

    def update_costs(self, node=None):
        if node is None:
            self.root.set_cost(0)
            return self.update_costs(self.root)

        new_cost = node.get_cost() + 1
        for direction, pos in node.get_child_positions():
            if pos not in self.explored:
                continue
            child = self.explored[pos]
            if child.get_cost() > new_cost:
                child.set_parent(node, direction)
                child.set_cost(new_cost)

                if child not in self.frontier:
                    self.update_costs(child)

    def retarget(self):
        # update tree costs
        self.update_costs()
        self.update_borders()
        self.frontier.sort(key=self.cost_function)
        while len(self.frontier) > 0:
            self.current_target = self.frontier.pop(0)
            if self.current_target.get_type() != "wall":
                return

    def expand_target(self):
        node = self.current_target
        if node is None:
            return
        self.current_target = None
        self.expand_node(node)

    def expand_node(self, node):
        new_cost = node.get_cost() + 1

        for i, new_pos in node.get_child_positions():
            if new_pos in self.explored:
                continue
            new_child = MappingNode(pos=new_pos)
            new_child.set_parent(node, i)
            new_child.set_cost(new_cost)
            self.frontier.append(new_child)
            self.explored[new_pos] = new_child


# all memory objects
mappingMemory = MappingMemory()


def sight_1_random_agent(percept):
    if percept:
        return "clean"

    return random.choice(directions)


def mapping_agent_sight_1(percept):
    expected_tile = mappingMemory.get_next_expected_tile_type()
    if expected_tile == "dirt":
        # at the target node!
        if not percept:
            # encountered a wall
            mappingMemory.set_target_type("wall")
            mappingMemory.retarget()
        else:
            # update last timestep's move
            mappingMemory.move()
            mappingMemory.expand_target()
            mappingMemory.retarget()
            mappingMemory.set_root_type("clean")
            return "clean"
    else:
        # update last timestep's move
        mappingMemory.move()

    new_dir = mappingMemory.get_next_direction()
    if new_dir == -1:
        # This only runs if there's an unreachable dirty tile
        return sight_1_random_agent(percept)
    return directions[new_dir]


def sight_5_random_agent(percept):
    if percept[0] == "dirt":
        return "clean"
    elif percept[1] == "dirt":
        return "north"
    elif percept[3] == "dirt":
        return "east"
    elif percept[2] == "dirt":
        return "south"
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
        self.max_steps = 2000

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
        for i in range(self.max_steps):
            path.append(current)
            if len(path) >= len(self.vertices):
                break
            current = min(
                filter(lambda x: x not in path, enumerate(self.vertices)),
                key=lambda x: self.manhattan_distance(current[0], x[0]),
            )
        return path

    def search_mapper(self, start):
        best_path = self.greedy(start)
        best_cost = self.cost_func(best_path)
        for likely_border_weight in (0, 4, 40):
            for possible_border_weight in (0, 1.5, 2.5):
                for singleton_weight in (0, 0.5, 1):
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
        for _ in range(self.max_steps):
            x, y = agent_pos
            if agent_pos in self.vertices:
                agent_pos_i = self.vertices.index(agent_pos)
                next_path_node = (agent_pos_i, agent_pos)
                if next_path_node not in path:
                    path.append(next_path_node)
                    if len(path) >= vertex_len:
                        return path

            percept = [
                (
                    "wall"
                    if pos not in self.vertices
                    else ("clean" if pos in explored else "dirt")
                )
                for pos in map(
                    lambda dir_val: (x + dir_val[0], y + dir_val[1]),
                    ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)),
                )
            ]

            action = mapping_agent_sight_5(percept, mapper)
            if action == "clean":
                explored.add(agent_pos)
            elif action == "unsolvable":
                return path
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

    def search_path_OPT2(self, start, n=8, cutoff=-1):
        """
        simulated annealing with 2opt
        """
        initial_temp = 1
        temp = initial_temp
        cooling_rate = 0.95

        initial_path = self.search_mapper(start)

        path, cost = initial_path, self.cost_func(initial_path)
        for _ in range(n):
            if cost < cutoff:
                break
            candidate_path, candidate_cost = self.OPT2_search(path)

            if candidate_cost < cost or random.random() < temp:
                path, cost = candidate_path, candidate_cost
            else:
                break
            temp *= cooling_rate

        return path


# size will be number of non-wall spaces in world
routeplanMemory = [-1]


def clear_routeplanMemory():
    routeplanMemory.clear()
    routeplanMemory.append(-1)


def tsp_agent_no_memory(percept, loss="dirt", weights=None, depth=0, cutoff=-1):
    """
    TSP is a poor name here--I really just use a greedy algorithm.
    I wrote a bunch of code to solve TSP better than greedy, but it's simply too slow.
    Note that if there is no solution, this is exceedingly slow, so I had to add the ability to quit early, lest runs take hours.
    """
    world, pos = percept
    x, y = pos
    if world[x][y] == "dirt":
        return "clean"

    # pick the dirty tile most likely to be closest
    width = len(world)
    height = len(world[0])
    target_pos = (0, 0)
    target_distance = width**2 * height**2
    for xp in range(width):
        for yp in range(height):
            if world[xp][yp] != "dirt":
                continue
            dist = manhattan_distance(pos, (xp, yp))
            if dist < target_distance:
                target_pos = (xp, yp)
                target_distance = dist

    # find shortest path to tile
    path = A_star(world, pos, target_pos)

    # make sure path axtually exists to the node, otherwise repick
    if path is None or len(path) < 2:
        for xp in range(width):
            for yp in range(height):
                if world[xp][yp] != "dirt":
                    continue
                target_pos = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1),
                )
                path = A_star(world, pos, target_pos)
                if not (path is None or len(path) < 2):
                    break
            if not (path is None or len(path) < 2):
                break
        if path is None or len(path) < 2:
            return "unsolvable"

    xp, yp = path[1]
    next_dir = (xp - x, yp - y)
    if next_dir not in dir_values:
        return sight_1_random_agent(False)
    return directions[dir_values.index(next_dir)]


def tsp_agent(percept, loss="dirt", weights=None, depth=4, cutoff=150000):
    world, pos = percept
    x, y = pos
    if world[x][y] == "dirt":
        return "clean"

    width = len(world)
    height = len(world[0])

    if len(routeplanMemory) == 0:
        return sight_1_random_agent(world[x][y])

    if len(routeplanMemory) == 1 and routeplanMemory[0] == -1:
        routeplanMemory.pop()
        walls = [
            (x, y) for x in range(width) for y in range(height) if world[x][y] == "wall"
        ]
        tsp = TSP(len(world), walls, loss)
        route = tsp.search_path_OPT2(pos, n=depth, cutoff=cutoff)
        routeplanMemory.extend([pos for _, pos in reversed(route)])

    if routeplanMemory and pos == routeplanMemory[-1]:
        routeplanMemory.pop()
        if not routeplanMemory:
            return "unsolvable"
        goal_x, goal_y = routeplanMemory[-1]
        while world[goal_x][goal_y] == "clean":
            routeplanMemory.pop()
            if not routeplanMemory:
                return "unsolvable"
            goal_x, goal_y = routeplanMemory[-1]
    path = A_star(world, pos, routeplanMemory[-1])

    # make sure path axtually exists to the node, otherwise repick
    if path is None or len(path) < 2:
        for xp in range(width):
            for yp in range(height):
                if world[xp][yp] != "dirt":
                    continue
                target_pos = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1),
                )
                path = A_star(world, pos, target_pos)
                if not (path is None or len(path) < 2):
                    break
            if not (path is None or len(path) < 2):
                break
        if path is None or len(path) < 2:
            return "unsolvable"

    xp, yp = path[1]
    next_dir = (xp - x, yp - y)
    if next_dir not in dir_values:
        return sight_1_random_agent(False)
    return directions[dir_values.index(next_dir)]


single_no_actions = sight_1_random_agent
single_no_dirt = sight_1_random_agent
single_yes_actions = mapping_agent_sight_1
single_yes_actions_reset = lambda: mappingMemory.reset(1, 100, 4, 2)
single_yes_dirt = mapping_agent_sight_1
single_yes_dirt_reset = lambda: mappingMemory.reset(1, 40, 1.5, 1)

surrounding_no_actions = sight_5_random_agent
surrounding_no_dirt = sight_5_random_agent
surrounding_yes_actions = lambda percept: mapping_agent_sight_5(percept, mappingMemory)
surrounding_yes_actions_reset = lambda: mappingMemory.reset(1, 1, 3, 2)
surrounding_yes_dirt = lambda percept: mapping_agent_sight_5(percept, mappingMemory)
surrounding_yes_dirt_reset = lambda: mappingMemory.reset(1, 1, 0, 1)

world_no_actions = lambda percept: tsp_agent_no_memory(percept, loss="actions", depth=0)
world_no_dirt = lambda percept: tsp_agent_no_memory(percept, loss="dirt", depth=0)
world_yes_actions = lambda percept: tsp_agent(percept, loss="actions", depth=0)
world_yes_actions_reset = clear_routeplanMemory
world_yes_dirt = lambda percept: tsp_agent(percept, loss="dirt", depth=3, cutoff=152000)
world_yes_dirt_reset = clear_routeplanMemory


def run_all(seed):
    random.seed(seed)
    kwargs = {
        "map_width": 20,
        "max_steps": 50000,
        "runs": 100,
        "knowledge": "single",
    }

    scores = [0 for i in range(12)]

    kwargs["knowledge"] = "single"
    # fmt: off
    scores[0] = many_runs(**kwargs, agent_function=single_no_actions, loss_function="actions")
    scores[1] = many_runs(**kwargs, agent_function=single_no_dirt, loss_function="dirt")
    scores[2] = many_runs(**kwargs, agent_function=single_yes_actions, loss_function="actions", agent_reset_function=single_yes_actions_reset)
    scores[3] = many_runs(**kwargs, agent_function=single_yes_dirt, loss_function="dirt", agent_reset_function=single_yes_dirt_reset)
    # fmt:on

    print("single_no_actions:", scores[0])
    print("single_no_dirt:", scores[1])
    print("single_yes_actions:", scores[2])
    print("single_yes_dirt:", scores[3])

    kwargs["knowledge"] = "surrounding"
    # fmt: off
    scores[4] = many_runs(**kwargs, agent_function=surrounding_no_actions, loss_function="actions")
    scores[5] = many_runs(**kwargs, agent_function=surrounding_no_dirt, loss_function="dirt")
    scores[6] = many_runs(**kwargs, agent_function=surrounding_yes_actions, loss_function="actions", agent_reset_function=surrounding_yes_actions_reset)
    scores[7] = many_runs(**kwargs, agent_function=surrounding_yes_dirt, loss_function="dirt", agent_reset_function=surrounding_yes_dirt_reset)
    # fmt:on

    print("surrounding_no_actions:", scores[4])
    print("surrounding_no_dirt:", scores[5])
    print("surrounding_yes_actions:", scores[6])
    print("surrounding_yes_dirt:", scores[7])

    kwargs["knowledge"] = "world"
    # fmt: off
    scores[8] = many_runs(**kwargs, agent_function=world_no_actions, loss_function="actions")
    scores[9] = many_runs(**kwargs, agent_function=world_no_dirt, loss_function="dirt")
    scores[10] = many_runs(**kwargs, agent_function=world_yes_actions, loss_function="actions", agent_reset_function=world_yes_actions_reset)
    scores[11] = many_runs(**kwargs, agent_function=world_yes_dirt, loss_function="dirt", agent_reset_function=world_yes_dirt_reset)
    # fmt:on

    print("world_no_actions:", scores[8])
    print("world_no_dirt:", scores[9])
    print("world_yes_actions:", scores[10])
    print("world_yes_dirt:", scores[11])

    return sum(scores)


if __name__ == "__main__":
    kwargs = {
        "map_width": 20,
        "max_steps": 50000,
        "knowledge": "world",
    }
    # run(**kwargs, agent_function=world_yes_dirt, loss_function="dirt")
    print(run_all(0))
