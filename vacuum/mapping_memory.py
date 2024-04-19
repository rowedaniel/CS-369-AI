from mapping_node import MappingNode
from agent_common import directions, dir_values, opposite_direction
from random import shuffle


class MappingMemory:
    def __init__(
        self,
        path_weight=1,
        likely_border_weight=100,
        possible_border_weight=2,
        singleton_weight=1,
    ):
        self.reset(path_weight, likely_border_weight, possible_border_weight, singleton_weight)

        # caches to save time
        self.border = {}

    def reset(self, path_weight, likely_border_weight, possible_border_weight, singleton_weight):

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
        self.frontier = [] # max size: map size + 2 in both dimensions (since it will assume borders are a set of walls)
        self.explored = {(0, 0): self.current_target} # some overlap with frontier, but always smaller than map size.

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
