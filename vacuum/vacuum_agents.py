"""
Use this script file to define your robot vacuum agents.

The run function will generate a map showing the animation of the robot, and return the output of the loss function at the end of the run. The many_runs function will run the simulation multiple times without the visualization and return the average loss.

You will need to implement a run_all function, which executes the many_runs function for all 12 of your agents (with the correct parameters) and sums up their returned losses as a single value. Your run_all function should use the following parameters for each agent: map_width=20, max_steps=50000 runs=100.
"""

from vacuum import run, many_runs
import random
from mapping_memory import MappingMemory
from agent_common import directions, dir_values


def random_agent(percept):
    if percept:
        return "clean"

    return random.choice(directions)


mappingMemory = MappingMemory()


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
        return random_agent(percept)
    return directions[new_dir]


def mapping_agent_sight_5(percept):
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
        # This only runs if there's an unreachable dirty tile
        return random_agent(percept[0] == "dirt")
    return directions[new_dir]


## input args for run: map_width, max_steps, agent_function, loss_function

# run(20, 50000, random_agent, "actions")
# TODO: better singleton management, prevent looping back on self when pushing the only possible frontier
random.seed(9)
mappingMemory.likely_border_weight = 2
mappingMemory.possible_border_weight = 1
run(20, 50000, mapping_agent_sight_5, "actions", knowledge="surrounding")
# random.seed(0)
# print(
#     "daniel:",
#     many_runs(
#         20,
#         50000,
#         20,
#         mapping_agent_sight_1,
#         "actions",
#         agent_reset_function=mappingMemory.reset,
#         knowledge="single",
#     ),
# )

## input args for many_runs: map_width, max_steps, runs, agent_function, loss_function


# NOTE:
# for:
# run_params = {
# "map_width": 20,
# "max_steps": 10000,
# "runs": 40,
# "loss_function": "dirt",
# "knowledge": "single",
# }
# best hyperparams were

# for
# run_params = {
#     "map_width": 20,
#     "max_steps": 10000,
#     "runs": 40,
#     "loss_function": "actions",
#     "knowledge": "single",
# }
# best hyperparams were likely=100, possible=4

# seed = 0

# likely_weights = (2, 4, 8, 16, 32, 64, 80, 90, 100, 110, 128)
# possible_weights = (0, 1, 2, 3, 4, 5, 6, 7, 8, 16)
# res = [[0 for _ in likely_weights] for _ in possible_weights]

# run_params = {
#     "map_width": 20,
#     "max_steps": 10000,
#     "runs": 20,
#     "loss_function": "actions",
#     "knowledge": "single",
# }

# random.seed(seed)
# print(
#     "random:",
#     many_runs(
#         run_params["map_width"],
#         run_params["max_steps"],
#         run_params["runs"],
#         random_agent,
#         run_params["loss_function"],
#         knowledge=run_params["knowledge"],
#     ),
# )
# for i, possible_weight in enumerate(possible_weights):
#     for j, likely_weight in enumerate(likely_weights):
#         mappingMemory.likely_border_weight = likely_weight
#         mappingMemory.possible_border_weight = possible_weight

#         random.seed(seed)
#         res[i][j] = many_runs(
#             run_params["map_width"],
#             run_params["max_steps"],
#             run_params["runs"],
#             mapping_agent_sight_1,
#             run_params["loss_function"],
#             agent_reset_function=mappingMemory.reset,
#             knowledge=run_params["knowledge"],
#         )
#         print(
#             f"daniel with likely weight: {likely_weight} and possible weight: {possible_weight}",
#             res[i][j],
#         )

# import matplotlib.pyplot as plt
# import numpy as np

# fig, ax = plt.subplots()
# im = ax.imshow(res)

# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(likely_weights)), labels=likely_weights)
# ax.set_yticks(np.arange(len(possible_weights)), labels=possible_weights)
# cbar = ax.figure.colorbar(im, ax=ax)
# for i in range(len(possible_weights)):
#     for j in range(len(likely_weights)):
#         score = res[i][j] / 1000
#         text = ax.text(j, i, f"{score:.3f}", ha="center", va="center", color="w")
# plt.show()
