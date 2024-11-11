import manavlib.io.xml_io as new_io
import manavlib.gen.tasks as agents
import manavlib.common.params as params
import manavlib.gen.maps as grid
import os
import sys
from dec_tswap.example_agent import RandomParams, SmartRandomParams, ShortestPathParams
from dec_tswap.base_tswap_agent import BaseTSWAPParams
from dec_tswap.naive_dec_tswap_agent import NaiveDecTSWAPParams
from dec_tswap.dec_tswap_agent import DecTSWAPParams

TASK_POSTFIX = "_task.xml"
CONFIG_POSTFIX = "_config.xml"
TASK_DIR = "./tasks"
MAP_PATH = os.path.join(TASK_DIR, "map.xml")

agents_num = 200
task_num = 250
h_map = 50
w_map = 50
r_vis = 3
ag_size = 0.3
max_steps = 500

default_ag_params = params.BaseDiscreteAgentParams()
default_ag_params.size = ag_size
default_ag_params.r_vis = r_vis

all_alg_params = [
    RandomParams(),
    SmartRandomParams(),
    ShortestPathParams(),
    BaseTSWAPParams(),
    NaiveDecTSWAPParams(),
    DecTSWAPParams()
]

exp_params = params.ExperimentParams()
exp_params.max_steps = max_steps
exp_params.timestep = 1.0
exp_params.xy_goal_tolerance = 0

cell_size = 1.0
h, w, grid_map = grid.create_empty_grid(w_map, h_map, cell_size)
new_io.create_map_file(MAP_PATH, grid_map, cell_size)

for alg_params in all_alg_params:
    config_path = os.path.join(TASK_DIR, f"{alg_params.alg_name}{CONFIG_POSTFIX}")
    new_io.create_config_file(config_path, alg_params, exp_params)

for task_id in range(task_num):
    starts, goal = agents.create_random_empty_instance(agents_num, h, w, cell_size, False, True)
    task_file = f"{task_id}{TASK_POSTFIX}"
    task_path = os.path.join(TASK_DIR, task_file)
    new_io.create_agents_file(task_path, starts, goal, default_ag_params)
    print(f"Task {task_file} generated in {TASK_DIR}")
