import copy
from const import *
import time
import numpy as np
from sklearn.neighbors import KDTree
import numpy.typing as npt
from typing import List, Dict, Set, Tuple, Type, Optional
from dec_tswap.agent import Agent
from dec_tswap.example_agent import (
    RandomParams,
    SmartRandomParams,
    ShortestPathParams,
    RandomAgent,
    SmartRandomAgent,
    ShortestPathAgent,
)
from dec_tswap.base_tswap_agent import BaseTSWAPParams, BaseTSWAPAgent
from dec_tswap.dec_tswap_agent import DecTSWAPParams, DecTSWAPAgent
from dec_tswap.naive_dec_tswap_agent import NaiveDecTSWAPParams, NaiveDecTSWAPAgent

from dec_tswap.message import Message
from manavlib.common.params import (
    ExperimentParams,
    BaseDiscreteAgentParams,
    BaseAlgParams,
)
from dec_tswap.path_table import PathTable


class Summary:
    """
    A class to store and summarize the results of an experiment.

    Parameters
    ----------
    agents_num : int, optional
        The number of agents involved in the experiment (default is 0).
    success : bool, optional
        Indicates whether the experiment was successful (default is False).
    collision : int, optional
        The total number of agent-to-agent collisions (default is 0).
    collision_obst : int, optional
        The total number of agent-to-obstacle collisions (default is 0).
    makespan : int, optional
        The total time taken for all agents to reach their goals (default is 0).
    flowtime : int, optional
        The sum of the time steps taken by all agents (default is 0).
    runtime : float, optional
        The total runtime of the experiment in seconds (default is 0).
    mean_groups : float, optional
        The average number of groups formed during the experiment (default is 0).
    mean_groups_size : float, optional
        The average size of the groups formed during the experiment (default is 0).
    """

    def __init__(
        self,
        agents_num: int = 0,
        success: bool = False,
        collision: int = 0,
        collision_obst: int = 0,
        makespan: int = 0,
        flowtime: int = 0,
        runtime: float = 0,
        mean_groups: float = 0,
        mean_groups_size: float = 0,
    ):
        self.agents_num: int = agents_num
        self.success: bool = success
        self.collision: int = collision
        self.collision_obst: int = collision_obst
        self.makespan: int = makespan
        self.flowtime: int = flowtime
        self.runtime: float = runtime
        self.mean_groups: float = mean_groups
        self.mean_groups_size: float = mean_groups_size

    def __str__(self):
        return f"{self.success:>7} {self.collision:>10} {self.collision_obst:>15} {self.makespan:>10} {self.flowtime:>10} {self.runtime:>10.3f} {self.mean_groups:>15.3f} {self.mean_groups_size:>20.3} {self.agents_num:>10}"

    @staticmethod
    def header():
        success_str = "success"
        collision_str = "collision"
        collision_obst_str = "collision_obst"
        makespan_str = "makespan"
        flowtime_str = "flowtime"
        runtime_str = "runtime"
        ag_num_str = "number"
        mean_groups_str = "mean_groups"
        mean_groups_size_str = "mean_groups_size"
        return f"{success_str:>7} {collision_str:>10} {collision_obst_str:>15} {makespan_str:>10} {flowtime_str:>10} {runtime_str:>10} {mean_groups_str:>15} {mean_groups_size_str:>20} {ag_num_str:>10}"


def get_agent_type(agents_params: BaseAlgParams) -> Type[Agent]:
    """
    Returns the agent type based on the provided algorithm parameters.

    Parameters
    ----------
    agents_params : BaseAlgParams
        The parameters defining the type of algorithm used by the agent.

    Returns
    -------
    Type[Agent]
        The class type of the agent corresponding to the provided parameters.
    """
    agent_types = {
        BaseAlgParams: Agent,
        RandomParams: RandomAgent,
        SmartRandomParams: SmartRandomAgent,
        ShortestPathParams: ShortestPathAgent,
        BaseTSWAPParams: BaseTSWAPAgent,
        DecTSWAPParams: DecTSWAPAgent,
        NaiveDecTSWAPParams: NaiveDecTSWAPAgent,
    }
    return agent_types[agents_params]


def run_experiment(
    start_states: npt.NDArray,
    goal_states: npt.NDArray,
    grid_map: npt.NDArray,
    cell_size: float,
    agents_num: int,
    agent_type: Type[Agent],
    agents_params: List[BaseDiscreteAgentParams],
    alg_params: BaseAlgParams,
    exp_params: ExperimentParams,
    search_object: PathTable,
    save_log: bool = True,
) -> Tuple[Summary, npt.NDArray, Optional[List[List[Set[int]]]]]:
    """
    Runs a multi-agent navigation experiment using the specified algorithm and parameters.

    Parameters
    ----------
    start_states : np.ndarray
        The initial positions of the agents.
    goal_states : np.ndarray
        The goal positions of the agents.
    grid_map : np.ndarray
        The grid map of the environment.
    cell_size : float
        The size of each grid cell.
    agents_num : int
        The number of agents participating in the experiment.
    agent_type : Type[Agent]
        The type of agent used in the experiment.
    agents_params : List[BaseDiscreteAgentParams]
        The list of agent-specific parameters.
    alg_params : BaseAlgParams
        The parameters specific to the chosen algorithm.
    exp_params : ExperimentParams
        The experiment parameters.
    search_object : PathTable
        The precomputed pathfinding object.
    save_log : bool, optional
        Whether to save the experiment logs (default is True).

    Returns
    -------
    Summary
        The summary of the experiment results.
    np.ndarray
        The log of agent positions at each time step.
    List[List[Set[int]]], optional
        The log of neighbor relationships at each time step.
    """

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
        pos_time_table,
        neighbors_log,
        goals_reached_time,
    ) = init_exp(
        start_states,
        goal_states,
        grid_map,
        agents_num,
        agent_type,
        agents_params,
        alg_params,
        exp_params,
        search_object,
    )

    groups_sum = 0
    grops_sizes_sum = 0
    result = Summary()
    start_time = time.time()
    if not initialize_agents(agents):
        result = Summary(
            agents_num,
            False,
            collisions,
            collisions_obst,
            t,
            t * agents_num,
            float(time.time() - start_time),
            0.0,
            0.0,
        )
        steps_log = steps_log[:t]
        return result, steps_log, neighbors_log

    for step in range(max_steps):
        groups_num, agents_in_groups = update_states_info(
            agents, current_states, agents_r_vis, save_log, neighbors_log
        )
        actions = compute_actions(agents, actions)
        execute_actions(current_states, actions)
        t += 1

        update_pos_time_table(current_states, t, pos_time_table)
        collisions += check_collisions(current_states, actions, pos_time_table, t)
        collisions_obst += check_collisions_obst(current_states, grid_map)
        groups_sum += groups_num
        grops_sizes_sum += agents_in_groups

        if collisions_obst:
            result = Summary(
                agents_num,
                False,
                collisions,
                collisions_obst,
                t,
                t * agents_num,
                float(time.time() - start_time),
                groups_sum / t,
                grops_sizes_sum / t,
            )
            steps_log = steps_log[:t]
            break

        update_log(current_states, steps_log, save_log, t)

        if collisions:
            result = Summary(
                agents_num,
                False,
                collisions,
                collisions_obst,
                t,
                t * agents_num,
                float(time.time() - start_time),
                groups_sum / t,
                grops_sizes_sum / t,
            )
            steps_log = steps_log[: t + 1]
            break

        if check_success(current_states, goal_states, goals_reached_time, t):
            result = Summary(
                agents_num,
                True,
                collisions,
                collisions_obst,
                t,
                np.sum(goals_reached_time),
                float(time.time() - start_time),
                groups_sum / t,
                grops_sizes_sum / t,
            )
            steps_log = steps_log[: t + 1]
            break
    else:
        result = Summary(
            agents_num,
            False,
            collisions,
            collisions_obst,
            t,
            t * agents_num,
            float(time.time() - start_time),
            groups_sum / t,
            grops_sizes_sum / t,
        )
    if not save_log:
        steps_log = None
        neighbors_log = None

    return result, steps_log, neighbors_log


def init_exp(
    start_states: npt.NDArray,
    goal_states: npt.NDArray,
    grid_map: npt.NDArray,
    agents_num: int,
    agent_type: Type[Agent],
    agents_params: List[BaseDiscreteAgentParams],
    alg_params: BaseAlgParams,
    exp_params: ExperimentParams,
    search_object,
):
    max_steps = exp_params.max_steps
    if len(start_states) < agents_num or len(goal_states) < agents_num:
        raise ValueError(
            "Number of agents must be less than or equal to the number of starting or goal states"
        )
    current_states = start_states[:agents_num].copy()
    goal_states = goal_states[:agents_num].copy()
    actions = np.zeros((agents_num, ACTION_DIM), dtype=np.int64)
    agents = [
        agent_type(
            a_id,
            current_states[a_id],
            agents_params[a_id],
            alg_params,
            grid_map,
            goal_states,
            search_object,
        )
        for a_id in range(agents_num)
    ]
    steps_log = np.zeros((max_steps + 1, agents_num, POS_DIM))
    steps_log[0, :, :] = current_states
    t = 0
    collisions = 0
    collisions_obst = 0

    if type(alg_params) is BaseTSWAPParams and alg_params.centralized:
        agents_r_vis = [100000 for a_id in range(agents_num)]
    else:
        agents_r_vis = [agents_params[a_id].r_vis for a_id in range(agents_num)]

    pos_time_table = dict()
    update_pos_time_table(current_states, t, pos_time_table)
    neighbors_log = []
    goals_reached_time = np.zeros(len(current_states), dtype=np.int32)

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
        pos_time_table,
        neighbors_log,
        goals_reached_time,
    )

    return result


def initialize_agents(agents) -> bool:
    for agent in agents:
        if not agent.initialize():
            return False

    return True


def update_states_info(
    agents: List[Agent],
    current_states: npt.NDArray,
    agents_r_vis: List[int],
    save_log: bool = False,
    neighbors_log: List[List[Set[int]]] = None,
) -> Tuple[int, float]:

    for a_id, agent in enumerate(agents):
        agent.update_state_info(current_states[a_id])

    neighbors_info, neighbors_ids, groups_num, agents_in_groups = update_neighbors_info(
        agents, current_states, agents_r_vis
    )

    for a_id, agent in enumerate(agents):
        agent.update_neighbors_info(copy.deepcopy(neighbors_info[a_id]))
    if save_log:
        neighbors_log.append(neighbors_ids)
    return groups_num, agents_in_groups


def compute_actions(agents: List[Agent], actions: npt.NDArray) -> npt.NDArray:
    for a_id, agent in enumerate(agents):
        actions[a_id] = agent.compute_action()
    return actions


def execute_actions(current_states: npt.NDArray, actions: npt.NDArray):
    for a_id in range(len(actions)):
        current_states[a_id] += actions[a_id]


def check_collisions(
    current_states: npt.NDArray,
    actions: npt.NDArray,
    pos_time_table: Dict[Tuple[int, int, int], int],
    t: int,
) -> int:
    return (
        check_vertex_conflicts(current_states, pos_time_table, t)
        + check_edge_conflict(current_states, actions, pos_time_table, t)
    ) // 2


def check_vertex_conflicts(
    current_states: npt.NDArray, pos_time_table: Dict[Tuple[int, int, int], int], t: int
) -> int:
    collisions = 0
    for a_id, (i, j) in enumerate(current_states):
        if pos_time_table[(i, j, t)] == COLLISION:
            collisions += 1
    return collisions


def check_edge_conflict(
    current_states: npt.NDArray,
    actions: npt.NDArray,
    pos_time_table: Dict[Tuple[int, int, int], int],
    t: int,
) -> int:
    if t == 0:
        return 0
    collisions = 0
    for a_id, pos in enumerate(current_states):
        old_pos = pos - actions[a_id]
        i1, j1 = old_pos
        i2, j2 = pos
        if (i1, j1) == (i2, j2):
            continue
        if (i2, j2, t - 1) not in pos_time_table or pos_time_table[
            (i2, j2, t - 1)
        ] == COLLISION:
            continue
        if (i1, j1, t) not in pos_time_table or pos_time_table[
            (i1, j1, t)
        ] == COLLISION:
            continue
        collisions += pos_time_table[(i2, j2, t - 1)] == pos_time_table[(i1, j1, t)]
    return collisions


def check_collisions_obst(current_states: npt.NDArray, grid_map: npt.NDArray):
    collisions_obst = 0
    for pos in current_states:
        collisions_obst += not (
            pos_on_map(pos, grid_map) and pos_is_traversable(pos, grid_map)
        )
    return collisions_obst


def pos_on_map(pos: npt.NDArray, grid_map: npt.NDArray) -> bool:
    i, j = pos
    h, w = grid_map.shape
    return (0 <= i < h) and (0 <= j < w)


def pos_is_traversable(pos: npt.NDArray, grid_map: npt.NDArray) -> bool:
    i, j = pos
    return grid_map[i, j] == MAP_TRAV


def check_success(
    current_states: npt.NDArray,
    goal_states: npt.NDArray,
    goals_reached_time: npt.NDArray,
    t: int,
) -> bool:

    goals_reached = np.zeros(len(current_states), dtype=bool)
    goals_dict = {
        (g_state[0], g_state[1]): g_id for g_id, g_state in enumerate(goal_states)
    }
    for a_id, (i, j) in enumerate(current_states):

        if (i, j) in goals_dict and not goals_reached[goals_dict[(i, j)]]:
            goals_reached[goals_dict[(i, j)]] = True
            if goals_reached_time[a_id] == -1:
                goals_reached_time[a_id] = t
        else:
            goals_reached_time[a_id] = -1
    return np.all(goals_reached)


def update_neighbors_info(
    agents: List[Agent], current_states: npt.NDArray, agents_r_vis: List[int]
) -> Tuple[List[List[Message]], List[Set[int]], int, float]:
    neighbors = compute_neighbors(current_states, agents_r_vis)
    groups = compute_neighbors_networks(neighbors)
    groups_num = len(groups)

    agents_num = len(current_states)
    neighbors_info = [list() for _ in range(agents_num)]
    neighbors_ids = [set() for _ in range(agents_num)]
    messages = []
    for a_id in range(agents_num):
        messages.append(agents[a_id].send_message())

    agents_in_group = 0
    for group in groups:
        group_messages = []
        agents_in_group += len(group)
        for a_id in group:
            group_messages.append(messages[a_id])

        for a_id in group:
            neighbors_info[a_id] = group_messages
            neighbors_ids[a_id] = group
    agents_in_group /= groups_num
    return neighbors_info, neighbors_ids, groups_num, agents_in_group


def update_log(
    current_states: npt.NDArray, steps_log: npt.NDArray, save_log: bool, t: int
):
    if save_log:
        steps_log[t] = current_states


def compute_neighbors(
    current_states: npt.NDArray, agents_r_vis: List[int]
) -> List[Set[int]]:
    agents_num = len(current_states)
    neighbors = [set() for _ in range(agents_num)]
    tree = KDTree(current_states, metric="chebyshev")

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
            for n_id in neighbors[current]:
                if n_id not in group:
                    queue.append(n_id)
        groups.append(group)
        considered.update(group)
    return groups


def update_pos_time_table(
    current_states: npt.NDArray, t: int, pos_time_table: Dict[Tuple[int, int, int], int]
) -> None:
    for a_id, (i, j) in enumerate(current_states):
        if (i, j, t) in pos_time_table:
            pos_time_table[(i, j, t)] = COLLISION
        else:
            pos_time_table[(i, j, t)] = a_id
