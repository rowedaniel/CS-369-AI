'''
Use this script file to define your robot vacuum agents.

The run function will generate a map showing the animation of the robot, and return the output of the loss function at the end of the run. The many_runs function will run the simulation multiple times without the visualization and return the average loss. 

You will need to implement a run_all function, which executes the many_runs function for all 12 of your agents (with the correct parameters) and sums up their returned losses as a single value. Your run_all function should use the following parameters for each agent: map_width=20, max_steps=50000 runs=100.
'''

from vacuum import *

directions = ['north', 'south', 'east', 'west']
prevdirection = 'null'


def random_agent(percept):
    if (percept):
        return 'clean'

    return random.choice(directions)


## input args for run: map_width, max_steps, agent_function, loss_function

run(20, 50000, random_agent, 'actions')

## input args for many_runs: map_width, max_steps, runs, agent_function, loss_function

# print(many_runs(20, 50000, 10, random_agent, 'dirt'))
