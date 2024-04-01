"""
Use this script file to define your robot vacuum agents.

The run function will generate a map showing the animation of the robot, and return the output of the loss function at the end of the run. The many_runs function will run the simulation multiple times without the visualization and return the average loss.

You will need to implement a run_all function, which executes the many_runs function for all 12 of your agents (with the correct parameters) and sums up their returned losses as a single value. Your run_all function should use the following parameters for each agent: map_width=20, max_steps=50000 runs=100.
"""

from vacuum import *

directions = ["north", "south", "east", "west"]
dir_values = [(0, 1), (0, -1), (1, 0), (-1, 0)]
prevdirection = "null"


def opposite_direction(direction):
    return direction ^ 1


def random_agent(percept):
    if percept[0] == "dirt":
        return "clean"

    return random.choice(directions)


class MappingNode:
    def __init__(
        self,
        parent=None,
        pos=(0, 0),
        cost=-1,
    ):
        self.parent = parent
        self.children = [None for i in range(4)]
        self.pos = pos
        self.cost = cost
        self.isWall = False
        self.direction = None

    def set_parent(self, parent, direction):
        if parent is None:
            return
        if parent.get_parent() == self:
            print("attempting to create a circular parent chain")
            raise ValueError("AAAA")
        if self.parent is not None:
            print("attempting to set parent when one already exists")
            new_dir = opposite_direction(self.direction)
            self.add_child(self.parent, new_dir)
        self.parent = parent
        self.set_cost(parent.get_cost() + 1)
        self.set_direction(direction)
        if self.pos == (0, 1):
            print("0,1 has children:")
            print(["None" if c is None else c.get_pos() for c in self.children])
        # if self.direction is not None:
        #     x, y = parent.get_pos()
        #     ox, oy = dir_values[self.direction]
        #     self.pos = (x + ox, y + oy)

    def remove_parent(self):
        self.parent = None
        self.direction = -1

    def get_parent(self):
        return self.parent

    def add_child(self, child, direction):
        if self.isWall:
            print("attempting to add a child to a wall >:[")
        if self.children[direction] is not None:
            print("attempting to set child when one already exists")
        self.children[direction] = child

    def swap_root(self, direction):
        # print(
        #     "swapping",
        #     "(",
        #     self,
        #     ")",
        #     self.get_pos(),
        #     "with direction",
        #     directions[direction],
        # )
        if self.parent is not None:
            print("attempting to swap root with a node that is not a root")
            return
        child = self.children[direction]
        if child is None:
            print("attempting to swap root with a null child")
            return
        # print("( old root:", self, "new root:", child, ")", child.get_pos())

        self.children[direction] = None
        child.remove_parent()
        op_dir = opposite_direction(direction)
        child.add_child(self, op_dir)
        self.set_parent(child, op_dir)
        # print("finished swapping")
        return child

    def get_children(self):
        return self.children

    def has_child(self, direction):
        return self.children[direction] is not None

    def get_pos(self):
        return self.pos

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        self.cost = cost
        for child in self.children:
            if child is None:
                continue
            if child.get_parent() != self:
                continue
            child.set_cost(self.cost + 1)

    def is_wall(self):
        return self.isWall

    def set_wall(self, isWall):
        self.isWall = isWall

    def get_direction(self):
        return self.direction

    def set_direction(self, direction):
        self.direction = direction


class MappingMemory:
    def __init__(self):
        # TODO: make frontier a priority queue which prioritizes unexplored nearby nodes
        self.frontier = []
        self.explored = {}
        self.current_node = MappingNode(cost=0)
        self.current_pos = (0, 0)
        self.prev_dir = None

    def get_next_direction(self):
        if self.current_node is None:
            return -1
        node = self.current_node
        direction = -1

        # print(
        #     "getting next direction. Goal is",
        #     self.current_node.get_pos(),
        #     "and currently at",
        #     self.current_pos,
        #     end=". ",
        # )

        while node.get_pos() != self.current_pos:
            parent = node.get_parent()
            if parent is None:
                print("Reached a parent of 'None'")
                break
            direction = node.get_direction()
            node = parent
        # print(f"decided to go {direction} ({directions[direction]})")
        return direction

    def move(self, direction):
        x, y = self.current_pos
        ox, oy = dir_values[direction]
        self.prev_dir = direction
        self.current_pos = (x + ox, y + oy)
        print(f"moving from {(x,y)} to {self.current_pos} ({directions[direction]})")
        self.update_costs((x, y), direction)

    def undo_move(self):
        if self.prev_dir is None:
            print("Cannot undo")
            return
        self.move(opposite_direction(self.prev_dir))
        self.prev_dir = None

    def update_costs(self, prev_pos, direction):
        # update costs
        if prev_pos not in self.explored:
            print("somehow just left an unexplored tile??")
            return
        prev_root = self.explored[prev_pos]

        new_root = prev_root.swap_root(direction)
        if new_root is None:
            print("failed to swap")
            raise ValueError("AAAA")
            return
        new_root.set_cost(0)
        for node in new_root.get_children():
            # print("updating", node, "route")
            self.update_cheaper_routes(node)

        # self.traverse_tree(new_root)

    def traverse_tree(self, node, prefix="", explored=None):
        if node is None:
            return
        if explored is None:
            explored = set()
        if node in explored:
            return
        explored.add(node)
        print(f"{prefix}{node.get_pos()}:")
        parent = node.get_parent()
        if parent is None:
            print(f"{prefix} parent: None")
        else:
            print(f"{prefix} parent: {parent.get_pos()}")
        for i, child in enumerate(node.get_children()):
            print(f"{prefix} child {i}:")
            self.traverse_tree(child, prefix + " ", explored)

    def set_wall(self):
        self.current_node.set_wall(True)
        self.explored[self.current_node.get_pos()] = self.current_node
        self.current_node = None
        self.undo_move()

    def get_current_node(self):
        return self.current_node

    def pop_frontier(self):
        if self.current_node is not None:
            print("attempting to pop frontier when already pursuing a node")
            return
        random.shuffle(self.frontier)
        self.current_node = self.frontier.pop()

    def update_cheaper_routes(self, node):
        if node is None:
            return
        print("updating route for node", node.get_pos())
        # check for cheaper routes
        # print("-" * 100 + "updating routes")
        x, y = node.get_pos()
        for i, (ox, oy) in enumerate(dir_values):
            possible_parent_pos = (x - ox, y - oy)
            if possible_parent_pos not in self.explored:
                continue
            possible_parent = self.explored[possible_parent_pos]
            current_parent = node.get_parent()
            if (
                current_parent is None
                or possible_parent.get_cost() < current_parent.get_cost()
            ):
                print("cheaper route found")
                node.set_parent(possible_parent, i)
                for child in node.get_children():
                    print("recursing")
                    self.update_cheaper_routes(child)
            else:
                print("no cheaper route found than through", current_parent.get_pos())
                print(
                    "cost", possible_parent.get_cost(), "vs", current_parent.get_cost()
                )

    def expand_node(self):
        node = self.current_node
        if node is None:
            return

        print("expanding node", node.get_pos())
        self.explored[node.get_pos()] = node
        self.current_node = None

        # add nodes to frontier
        x, y = node.get_pos()
        named_frontier = {n.get_pos(): n for n in self.frontier}
        for i, new_pos in enumerate((x + ox, y + oy) for ox, oy in dir_values):
            # check if node already accounted for
            if new_pos in self.explored or new_pos in named_frontier:
                if new_pos in self.explored:
                    child = self.explored[new_pos]
                if new_pos in named_frontier:
                    child = named_frontier[new_pos]
                if child != node.get_parent():
                    # provided it's not the current node's parent, then connect it
                    node.add_child(child, i)
                    old_parent = child.get_parent()
                    if node.get_cost() < old_parent.get_cost():
                        print(
                            f"clobbering old parent {child.get_parent().get_pos()} of node {child.get_pos()} in favor of {node.get_pos()}"
                        )
                        child.set_parent(node, i)
                continue

            new_node = MappingNode(pos=new_pos)
            node.add_child(new_node, i)
            new_node.set_parent(node, i)
            self.update_cheaper_routes(new_node)
            self.frontier.append(new_node)
            # print("new node on frontier:", new_pos, i, f"({directions[i]})")

    def print_status(self):
        print("\n\ntarget node:", self.current_node)
        print("explored:")
        for pos, node in self.explored.items():
            print(
                f"    {pos}: pos {node.pos} direction {node.direction} cost {node.cost}"
            )
        print("frontier:")
        for node in self.frontier:
            print(f"    pos {node.pos} direction {node.direction} cost {node.cost}")


mappingMemory = MappingMemory()


def mapping_agent_sight_1(percept):
    # input()
    if percept[0] == "wall":
        print("Error: actor is on a wall")
        return random.choice(directions)

    if percept[0] == "dirt":
        # on dirt, assume made it to destination.
        mappingMemory.expand_node()
        mappingMemory.pop_frontier()
        # mappingMemory.print_status()
        print("cleaning")
        return "clean"

    # on a clean tile
    # print("on a clean tile")
    next_dir = mappingMemory.get_next_direction()

    if next_dir == -1:
        # print("on a clean tile when a dirty tile was expected")
        # expected to be on a dirty tile
        mappingMemory.set_wall()
        mappingMemory.pop_frontier()
        # mappingMemory.print_status()
        next_dir = mappingMemory.get_next_direction()
    if next_dir == -1:
        print(
            "still don't know where going, despite having recalculated route just now??"
        )

    mappingMemory.move(next_dir)
    return directions[next_dir]


# TODO: debug only
mapping_agent_sight_1.mem = mappingMemory

## input args for run: map_width, max_steps, agent_function, loss_function

random.seed(0)
run(20, 50000, mapping_agent_sight_1, "actions")

## input args for many_runs: map_width, max_steps, runs, agent_function, loss_function

# print(many_runs(20, 50000, 10, random_agent, 'dirt'))
