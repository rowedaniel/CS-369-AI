from agent_common import dir_values, opposite_direction


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
            raise ValueError("AAAA")
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
