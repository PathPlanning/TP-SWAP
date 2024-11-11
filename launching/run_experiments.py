import os
from pathlib import Path
from itertools import product
import datetime
from multiprocessing import Pool, freeze_support, RLock
from multiprocessing import current_process
import time
import utils
import manavlib.io.xml_io as xml_io
from dec_tswap.path_table import PathTable
from tqdm import tqdm


import argparse

BASE_TASK_DIR = "tasks"
BASE_RESULTS_DIR = "results"
BASE_CONFIG_FILE = "_config"
BASE_RESULT_FILE = "result"
MAP_FILE = "map.xml"
TASK_POSTFIX = "_task.xml"


def run_experiment(arg):

    current = current_process()
    pos = current._identity[0] - 1

    map_name, alg_name, (task_num, agents_from, agents_step, agents_to) = arg
    print(f'Processing -- map name: "{map_name}"; alg. name: "{alg_name}"')
    alg_name_score = alg_name.replace("_", "-")
    result_dir = os.path.join(BASE_RESULTS_DIR, map_name, alg_name_score)
    now = datetime.datetime.now()
    now_str = now.strftime("%d_%m_%y_%H_%M_%S")
    result_file = f"{BASE_RESULT_FILE}_{now_str}.txt"

    if not os.path.exists(result_dir):
        path = Path(result_dir)
        path.mkdir(parents=True)

    task_dir = os.path.join(BASE_TASK_DIR, map_name)
    config_name = f"{alg_name}{BASE_CONFIG_FILE}.xml"
    map_path = os.path.join(task_dir, MAP_FILE)
    config_path = os.path.join(task_dir, config_name)
    exp_params, alg_params = xml_io.read_xml_config(config_path)

    task_dir = os.path.join(BASE_TASK_DIR, map_name)
    config_name = f"{alg_name}{BASE_CONFIG_FILE}.xml"
    h, w, cs, grid_map = xml_io.read_xml_map(map_path)

    result_path = os.path.join(result_dir, result_file)
    result_file = open(result_path, "w")
    result_file.write(utils.Summary.header() + "\n")

    for task_id in range(task_num):

        task_file = f"{task_id}{TASK_POSTFIX}"
        task_path = os.path.join(task_dir, task_file)
        default_params, starts, goals, ag_params = xml_io.read_xml_agents(task_path)
        path_table = PathTable(grid_map, goals[:agents_to])
        for agents_num in range(agents_from, agents_to + 1, agents_step):
            summary, steps_log, neighbors_log = utils.run_experiment(
                starts,
                goals,
                grid_map,
                cs,
                agents_num,
                utils.get_agent_type(type(alg_params)),
                ag_params,
                alg_params,
                exp_params,
                path_table,
                False,
            )
            print(f"Task{task_id+1:>4} / {task_num:<4}-> ", str(summary), f"    (Algorithm:{alg_name:^15}, Map:{map_name})")
            result_file.write(str(summary) + "\n")
    result_file.close()
    return result_path


def main(args):

    task_num = args.task_num
    agents_from = args.agents_from
    agents_step = args.agents_step
    agents_to = args.agents_to

    maps = args.maps
    algs = args.algoritms

    parallel_threads = args.parallel_threads

    print("Precheck...")
    for map_name, alg_name in product(maps, algs):
        task_dir = os.path.join(BASE_TASK_DIR, map_name)
        config_name = f"{alg_name}{BASE_CONFIG_FILE}.xml"

        if not os.path.exists(task_dir):
            print("Task dir does not exist:", task_dir)
            exit(-1)

        config_path = os.path.join(task_dir, config_name)

        if not os.path.exists(config_path):
            print("Config file does not exist:", config_path)
            exit(-1)

        map_path = os.path.join(task_dir, MAP_FILE)
        if not os.path.exists(map_path):
            print("Map file does not exist:", map_path)
            exit(-1)

    print("Done")

    before = time.time()
    with Pool(parallel_threads) as p:
        files = p.map(
            run_experiment,
            list(
                product(maps, algs, [(task_num, agents_from, agents_step, agents_to)])
            ),
        )
        print(files)
    after = time.time()
    print()
    print(
        f"Exp. time: {int(after-before)//3600}h {int(after-before)%3600//60}m {int(after-before) % 60}s",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="run_experiments",
        description="""Script to run experiments with AMAPF algorithms. 
                    The experiments are run using pyhon and the result of 
                    running each series of experiments is saved in a text file.""",
    )

    parser.add_argument(
        "-n",
        "--task_num",
        type=int,
        required=True,
        help="The number of tasks for every map",
    )
    parser.add_argument(
        "-f",
        "--agents_from",
        type=int,
        required=True,
        help="The number of agents varies starting from this value",
    )
    parser.add_argument(
        "-s",
        "--agents_step",
        type=int,
        required=True,
        help="The number of agents is varied in steps equal to this value",
    )
    parser.add_argument(
        "-t",
        "--agents_to",
        type=int,
        required=True,
        help="The number of agents varies up to this value",
    )

    parser.add_argument(
        "-m",
        "--maps",
        required=True,
        nargs="*",
        help=f"The maps on which the experiments are conducted. The names of the maps should correspond to the names of the subfolders in '{BASE_TASK_DIR}' folder containing the tasks for the experiment.",
    )
    parser.add_argument(
        "-a",
        "--algoritms",
        choices=[
            "naive_dec_tswap",
            "base_tswap",
            "random",
            "shortest_path",
            "dec_tswap",
            "smart_random",
        ],
        required=True,
        nargs="*",
        help="The algorithms that will participate in the experiments. The next options are available t this moment: ['naive_dec_tswap', 'base_tswap', 'random', 'shortest_path', 'dec_tswap', 'smart_random'].",
    )

    parser.add_argument(
        "-p",
        "--parallel_threads",
        type=int,
        required=False,
        default=2,
        help="The value specifies the number of parallel threads that are used to run the experiments. Default is 2.",
    )

    args = parser.parse_args()

    main(args)
