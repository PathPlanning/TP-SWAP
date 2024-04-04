import sys
from const import *
import time

sys.path.append("../")
import numpy as np
from sklearn.neighbors import KDTree
import numpy.typing as npt
from typing import List, Dict, Set, Tuple, Type
from dec_tswap.agent import Agent, DecTSWAPParams, Message
from manavlib.gen.params import ExperimentParams, DiscreteAgentParams


class Summary:
    def __init__(self, success: bool = False, collision: int = 0, collision_obst: int = 0, makespan: int = 0,
                 flowtime: int = 0,
                 runtime: float = 0):
        self.success: bool = success
        self.collision: int = collision
        self.collision_obst: int = collision_obst
        self.makespan: int = makespan
        self.flowtime: int = flowtime
        self.runtime: float = runtime

    def __str__(self):
        return f"{self.success:>7} {self.collision:>10} {self.collision_obst:>15} {self.makespan:>10} {self.flowtime:>10} {self.runtime:>10.3f}"

    def header(self):
        success_str = "success"
        collision_str = "collision"
        collision_obst_str = "collision_obst"
        makespan_str = "makespan"
        flowtime_str = "flowtime"
        runtime_str = "runtime"
        return f"{success_str:>7} {collision_str:>10} {collision_obst_str:>15} {makespan_str:>10} {flowtime_str:>10} {runtime_str:>10}"


def run_experiment(start_states: npt.NDArray,
                   goal_states: npt.NDArray,
                   grid_map: npt.NDArray,
                   cell_size: float,
                   agents_num: int,
                   agent_type: Type[Agent],
                   agents_params: List[DiscreteAgentParams],
                   alg_params: DecTSWAPParams,
                   exp_params: ExperimentParams,
                   save_log: bool = True):
    (
        agents,
        current_states,
        goal_states,
        actions,
        max_steps,
        steps_log,
        t,
        collisions,
        collisions_obst,
        agents_r_vis,
        pos_time_table
    ) = init_exp(start_states, goal_states, grid_map, cell_size, agents_num, agent_type, agents_params, alg_params,
                 exp_params)

    result = Summary()
    start_time = time.time()
    for step in range(max_steps):
        update_states_info(agents, current_states, agents_r_vis)
        actions = compute_actions(agents, actions)
        execute_actions(current_states, actions)
        t += 1

        update_pos_time_table(current_states, t, pos_time_table)
        collisions += check_collisions(current_states, actions, pos_time_table, t)
        collisions_obst += check_collisions_obst(current_states, grid_map)

        if collisions_obst:
            result = (Summary
                      (False,
                       collisions,
                       collisions_obst,
                       step,
                       step * agents_num,
                       float(time.time() - start_time))
                      )
            steps_log = steps_log[:t]
            break

        update_log(current_states, steps_log, save_log, t)

        if collisions:
            result = (Summary
                      (False,
                       collisions,
                       collisions_obst,
                       step,
                       step * agents_num,
                       float(time.time() - start_time))
                      )
            steps_log = steps_log[:t + 1]
            break

        if check_success(current_states, goal_states):
            result = (Summary
                      (True,
                       collisions,
                       collisions_obst,
                       step,
                       -1,
                       float(time.time() - start_time))
                      )
            steps_log = steps_log[:t + 1]
            break

    if not save_log:
        steps_log = None
    return result, steps_log


def init_exp(start_states: npt.NDArray,
             goal_states: npt.NDArray,
             grid_map: npt.NDArray,
             cell_size: float,
             agents_num: int,
             agent_type: Type[Agent],
             agents_params: List[DiscreteAgentParams],
             alg_params: DecTSWAPParams,
             exp_params: ExperimentParams):
    max_steps = exp_params.max_steps
    if len(start_states) < agents_num or len(goal_states) < agents_num:
        raise ValueError("Number of agents must be less than or equal to the number of starting or goal states")
    current_states = start_states[:agents_num].copy()
    goal_states = goal_states[:agents_num].copy()
    actions = np.zeros((agents_num, ACTION_DIM), dtype=np.int64)
    agents = [agent_type(a_id, agents_params[a_id], alg_params, grid_map, goal_states) for a_id in range(agents_num)]
    steps_log = np.zeros((max_steps + 1, agents_num, POS_DIM))
    steps_log[0, :, :] = current_states
    t = 0
    collisions = 0
    collisions_obst = 0
    agents_r_vis = [agents_params[a_id].r_vis for a_id in range(agents_num)]
    pos_time_table = dict()
    update_pos_time_table(current_states, t, pos_time_table)

    result = (
        agents,
        current_states,
        goal_states,
        actions,
        max_steps,
        steps_log,
        t,
        collisions,
        collisions_obst,
        agents_r_vis,
        pos_time_table
    )

    return result


def update_states_info(agents: List[Agent],
                       current_states: npt.NDArray,
                       agents_r_vis: List[int]):
    neighbors_info = update_neighbors_info(agents, current_states, agents_r_vis)
    for a_id, agent in enumerate(agents):
        agent.update_state_info(current_states[a_id])
        agent.update_neighbors_info(neighbors_info[a_id])


def compute_actions(agents: List[Agent],
                    actions: npt.NDArray) -> npt.NDArray:
    for a_id, agent in enumerate(agents):
        actions[a_id] = agent.compute_action()
    return actions


def execute_actions(current_states: npt.NDArray,
                    actions: npt.NDArray):
    for a_id in range(len(actions)):
        current_states[a_id] += actions[a_id]


def check_collisions(current_states: npt.NDArray,
                     actions: npt.NDArray,
                     pos_time_table: Dict[Tuple[int, int, int], int],
                     t: int) -> int:
    return (check_vertex_conflicts(current_states, pos_time_table, t) + check_edge_conflict(current_states, actions,
                                                                                           pos_time_table, t)) // 2


def check_vertex_conflicts(current_states: npt.NDArray,
                           pos_time_table: Dict[Tuple[int, int, int], int],
                           t: int) -> int:
    collisions = 0
    for a_id, (i, j) in enumerate(current_states):
        if pos_time_table[(i, j, t)] == COLLISION:
            collisions += 1
    return collisions


def check_edge_conflict(current_states: npt.NDArray,
                        actions: npt.NDArray,
                        pos_time_table: Dict[Tuple[int, int, int], int],
                        t: int) -> int:
    if t == 0:
        return 0
    collisions = 0
    for a_id, pos in enumerate(current_states):
        old_pos = pos - actions[a_id]
        i1, j1 = old_pos
        i2, j2 = pos
        if (i1, j1) == (i2, j2):
            continue
        if (i2, j2, t - 1) not in pos_time_table or pos_time_table[(i2, j2, t - 1)] == COLLISION:
            continue
        if (i1, j1, t) not in pos_time_table or pos_time_table[(i1, j1, t)] == COLLISION:
            continue
        collisions += (pos_time_table[(i2, j2, t - 1)] == pos_time_table[(i1, j1, t)])
    return collisions


def check_collisions_obst(current_states: npt.NDArray, grid_map: npt.NDArray):
    collisions_obst = 0
    for pos in current_states:
        collisions_obst += not (pos_on_map(pos, grid_map) and pos_is_traversable(pos, grid_map))
    return collisions_obst


def pos_on_map(pos: npt.NDArray, grid_map: npt.NDArray) -> bool:
    i, j = pos
    h, w = grid_map.shape
    return (0 <= i < h) and (0 <= j < w)


def pos_is_traversable(pos: npt.NDArray, grid_map: npt.NDArray) -> bool:
    i, j = pos
    return grid_map[i, j] == MAP_TRAV


def check_success(current_states, goal_states) -> bool:
    # TODO: maybe should be optimized
    goals_reached = np.zeros(len(current_states), dtype=bool)
    goals_dict = {(g_state[0], g_state[1]): g_id for g_id, g_state in enumerate(goal_states)}
    for a_id, (i, j) in enumerate(current_states):

        if (i, j) in goals_dict and not goals_reached[goals_dict[(i, j)]]:
            goals_reached[goals_dict[(i, j)]] = True
    return np.all(goals_reached)


def update_neighbors_info(agents: List[Agent],
                          current_states: npt.NDArray,
                          agents_r_vis: List[int]) -> List[List[Message]]:
    neighbors = compute_neighbors(current_states, agents_r_vis)
    groups = compute_neighbors_networks(neighbors)

    agents_num = len(current_states)
    neighbors_info = [list() for _ in range(agents_num)]
    messages = []
    for a_id in range(agents_num):
        messages.append(agents[a_id].send_message())

    for group in groups:
        group_messages = []
        for a_id in group:
            group_messages.append(messages[a_id])
        for a_id in group:
            neighbors_info[a_id] = group_messages
    return neighbors_info


def update_log(current_states: npt.NDArray,
               steps_log: npt.NDArray,
               save_log: bool,
               t: int):
    if save_log:
        steps_log[t] = current_states


def compute_neighbors(current_states: npt.NDArray, agents_r_vis: List[int]) -> List[Set[int]]:
    agents_num = len(current_states)
    neighbors = [set() for _ in range(agents_num)]
    tree = KDTree(current_states, metric='chebyshev')

    for a_id, pos in enumerate(current_states):
        n_ind = tree.query_radius([pos], agents_r_vis[a_id])
        neighbors[a_id].update(n_ind[0])

    return neighbors


def compute_neighbors_networks(neighbors: List[Set[int]]) -> List[Set[int]]:
    groups = []
    agents_num = len(neighbors)
    considered = set()
    for a_id in range(agents_num):
        if a_id in considered:
            continue
        considered.add(a_id)
        group = set()
        queue = [a_id]
        while len(queue):
            current = queue.pop()
            group.add(current)
            for n_id in neighbors[a_id]:
                if n_id not in group:
                    queue.append(n_id)
        groups.append(group)
        considered.update(group)
    return groups


def update_pos_time_table(current_states: npt.NDArray,
                          t: int,
                          pos_time_table: Dict[Tuple[int, int, int], int]
                          ) -> None:
    for a_id, (i, j) in enumerate(current_states):
        if (i, j, t) in pos_time_table:
            pos_time_table[(i, j, t)] = COLLISION
        else:
            pos_time_table[(i, j, t)] = a_id
