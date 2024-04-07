from random import shuffle
from heapq import heappop, heappush
from agent_common import dir_values


class PathFinder:
    def __init__(self, size, walls):
        self.vertices = tuple(
            (x, y) for x in range(size) for y in range(size) if (x, y) not in walls
        )
        self.walls = walls
        self.size = size

    def generate_random_permutation(self):
        perm = list(self.vertices)
        shuffle(perm)
        return perm

    def generate_random_path(self, start):
        perm = self.generate_random_permutation()
        perm.remove(start)
        path = [start]
        explored = set()
        while perm:
            next_vertex = perm.pop()
            if next_vertex in explored:
                continue
            new_section = self.A_star(start, next_vertex, explored)[1:]
            explored.update(new_section)
            path.extend(new_section)
            start = next_vertex
        return path
    
    def search_shortest_path(self, start, n):
        shortest = self.generate_random_path(start)
        for i in range(n):
            candidate = self.generate_random_path(start)
            if len(candidate) > len(shortest):
                shortest = candidate
        return shortest

    def get_neighbors(self, pos):
        x, y = pos
        return (
            (x + dx, y + dy)
            for dx, dy in dir_values
            if 0 <= x + dx < self.size
            and 0 <= y + dy < self.size
            and (x + dx, y + dy) not in self.walls
        )

    def heuristic(self, pos, goal):
        return sum(abs(p - g) for p, g in zip(pos, goal))

    def A_star(self, start, goal, discouraged):
        frontier = []
        explored = set()
        heappush(frontier, (0, start, []))  # (f, state, path)
        while frontier:
            _, current_state, path = heappop(frontier)

            if current_state == goal:
                return path + [current_state]

            if current_state in explored:
                continue
            explored.add(tuple(current_state))

            for neighbor in self.get_neighbors(current_state):
                if tuple(neighbor) in explored:
                    continue

                g = len(path) + 1
                h = self.heuristic(neighbor, goal)
                if neighbor in discouraged:
                    # artificially amp up the cost to go onto already explored nodes
                    h += 4
                f = g + h
                heappush(frontier, (f, neighbor, path + [current_state]))
        return None


def print_board(size, walls, start=(-1, -1), goal=(-1, -1), path=None):
    if path is None:
        path = []
    for y in range(size - 1, -1, -1):
        for x in range(6):
            print(" ", end="")
            if (x, y) == start:
                print("o", end="")
            elif (x, y) == goal:
                print("x", end="")
            elif (x, y) in walls:
                print("H", end="")
            elif (x, y) in path:
                print(str(path.index((x, y)))[-1], end="")
            else:
                print(" ", end="")
            print(" ", end="")
        print(f"|{y}")

    print("-" * size * 3)
    print(" " + "  ".join(str(i)[-1] for i in range(size)))
    print()


if __name__ == "__main__":
    walls = [(3, 3), (3, 2), (4, 2), (0, 4), (1, 3), (3, 4)]
    size = 6

    pathFinder = PathFinder(size, walls)
    start = (0, 0)
    path = pathFinder.search_shortest_path(start, 10000)
    for i in range(0, len(path), 5):
        print_board(size, walls, start=path[i], path=path[i+1:i+6])
