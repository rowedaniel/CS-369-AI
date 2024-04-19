"""
Use this script file to define your robot vacuum agents.

The run function will generate a map showing the animation of the robot, and return the output of the loss function at the end of the run. The many_runs function will run the simulation multiple times without the visualization and return the average loss.

You will need to implement a run_all function, which executes the many_runs function for all 12 of your agents (with the correct parameters) and sums up their returned losses as a single value. Your run_all function should use the following parameters for each agent: map_width=20, max_steps=50000 runs=100.
"""

from vacuum import run, many_runs
import random
from mapping_memory import MappingMemory
from agent_common import directions, dir_values
from tsp import TSP, mapping_agent_sight_5, sight_5_random_agent
from A_star import A_star

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


# size will be number of non-wall spaces in world
routeplanMemory = [-1]


def clear_routeplanMemory():
    routeplanMemory.clear()
    routeplanMemory.append(-1)


def tsp_agent(percept, loss="dirt", weights=None, depth=4, score_cutoff=150000):
    world, pos = percept
    if len(routeplanMemory) == 0:
        x, y = pos
        return sight_1_random_agent(world[x][y])

    if len(routeplanMemory) == 1 and routeplanMemory[0] == -1:
        routeplanMemory.pop()
        size = len(world)
        walls = [
            (x, y) for x in range(size) for y in range(size) if world[x][y] == "wall"
        ]
        tsp = TSP(len(world), walls, loss)
        route = tsp.search_path_OPT2(pos, n=depth, score_cutoff=score_cutoff)
        routeplanMemory.extend([pos for _, pos in reversed(route)])
    x, y = pos
    if routeplanMemory and pos == routeplanMemory[-1]:
        routeplanMemory.pop()
        if routeplanMemory:
            goal_x, goal_y = routeplanMemory[-1]
            while world[goal_x][goal_y] == "clean":
                routeplanMemory.pop()
                if not routeplanMemory:
                    break
                goal_x, goal_y = routeplanMemory[-1]
    if world[x][y] == "dirt":
        return "clean"
    path = A_star(world, pos, routeplanMemory[-1])
    if path is None or len(path) < 2:
        return sight_1_random_agent(False)
    xp, yp = path[1]
    next_dir = (xp - x, yp - y)
    if next_dir not in dir_values:
        return sight_1_random_agent(False)
    return directions[dir_values.index(next_dir)]


single_no_actions = sight_1_random_agent
single_no_dirt = sight_1_random_agent
single_yes_actions = mapping_agent_sight_1
# TODO: update weights
single_actions_reset = lambda: mappingMemory.reset(1, 100, 4, 2)
single_yes_dirt = mapping_agent_sight_1
# TODO: update weights
single_dirt_reset = lambda: mappingMemory.reset(1, 100, 4, 2)

surrounding_no_actions = sight_5_random_agent
surrounding_no_dirt = sight_5_random_agent
surrounding_yes_actions = lambda percept: mapping_agent_sight_5(percept, mappingMemory)
# TODO: update weights
surrounding_actions_reset = lambda: mappingMemory.reset(1, 100, 4, 2)
surrounding_yes_dirt = lambda percept: mapping_agent_sight_5(percept, mappingMemory)
# TODO: update weights
surrounding_dirt_reset = lambda: mappingMemory.reset(1, 100, 4, 2)

map_no_actions = None
map_no_dirt = None
map_yes_actions = lambda percept: tsp_agent(percept, loss="actions", depth=0)
map_yes_dirt = lambda percept: tsp_agent(
    percept, loss="dirt", depth=3, score_cutoff=152000
)
map_yes_reset = clear_routeplanMemory

## input args for run: map_width, max_steps, agent_function, loss_function

# run(20, 50000, random_agent, "actions")
# TODO: better singleton management, prevent looping back on self when pushing the only possible frontier
import time

seed = 0

t1 = time.time()
random.seed(seed)
surrounding_dirt_reset()
loss = run(20, 50000, single_no_dirt, "dirt", knowledge="single", animate=False)
print("single no dirt loss:", loss, "time:", time.time() - t1)

t1 = time.time()
random.seed(seed)
surrounding_dirt_reset()
loss = run(
    20, 50000, surrounding_no_dirt, "dirt", knowledge="surrounding", animate=False
)
print("surrounding no dirt loss:", loss, "time:", time.time() - t1)

t1 = time.time()
random.seed(seed)
surrounding_dirt_reset()
loss = run(20, 50000, single_yes_dirt, "dirt", knowledge="single", animate=False)
print("single yes dirt loss:", loss, "time:", time.time() - t1)

t1 = time.time()
random.seed(seed)
surrounding_dirt_reset()
loss = run(
    20, 50000, surrounding_yes_dirt, "dirt", knowledge="surrounding", animate=False
)
print("surrounding yes dirt loss:", loss, "time:", time.time() - t1)

map_yes_reset()
t1 = time.time()
random.seed(seed)
loss = run(20, 50000, map_yes_dirt, "dirt", knowledge="world", animate=False)
print("whole world yes dirt loss:", loss, "time:", time.time() - t1)

t1 = time.time()
avg_loss = many_runs(
    20,
    50000,
    100,
    map_yes_dirt,
    "dirt",
    agent_reset_function=map_yes_reset,
    knowledge="world",
)
print("for many runs of map_yes_dirt:", avg_loss, "time:", time.time() - t1)


# t1 = time.time()
# random.seed(0)
# avg_loss = many_runs(
#     20,
#     50000,
#     10,
#     map_yes_dirt,
#     "dirt",
#     agent_reset_function=map_yes_reset,
#     knowledge="world",
# )
# print("loss:", avg_loss, "time:", time.time() - t1)

# t1 = time.time()
# surrounding_loss = run(20, 50000, surrounding_yes_dirt, "dirt", knowledge="surrounding", animate=False)
# print("surrounding loss:", surrounding_loss, "took", time.time() - t1)
# random.seed(0)
# t1 = time.time()
# all_loss = run(20, 50000, map_yes_dirt, "dirt", knowledge="world", animate=False)
# print("all loss:", all_loss, "took", time.time() - t1)
