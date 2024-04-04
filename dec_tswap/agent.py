from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from manavlib.gen.params import DiscreteAgentParams, BaseAlgParams
import numpy.typing as npt
from enum import Enum
import numpy as np
import random

from dec_tswap.map import Map
from dec_tswap.astar_algorithm import astar_search, manhattan_distance, make_path


class DecTSWAPParams(BaseAlgParams):
    def __init__(self) -> None:
        super().__init__()
        pass


class Message:
    def __init__(self):
        self.pos: npt.NDArray | None = None
        self.next_pos: npt.NDArray | None = None
        self.priority: int | None = None


class Action(Enum):
    WAIT = (0, 0)
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Agent:
    def __init__(self,
                 a_id: int,
                 ag_params: DiscreteAgentParams,
                 alg_params: BaseAlgParams,
                 grid_map: npt.NDArray,
                 goals: npt.NDArray):
        self.a_id = a_id
        self.ag_params = ag_params
        self.alg_params = alg_params
        self.grid_map = grid_map
        self.goals = goals

    def update_neighbors_info(self, neighbors_info: List[Message]) -> None:
        raise NotImplementedError

    def compute_action(self) -> Action:
        raise NotImplementedError

    def update_state_info(self, new_pos: npt.NDArray) -> None:
        raise NotImplementedError

    def send_message(self) -> Message:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self,
                 a_id: int,
                 ag_params: DiscreteAgentParams,
                 alg_params: BaseAlgParams,
                 grid_map: npt.NDArray,
                 goals: npt.NDArray):
        super().__init__(a_id, ag_params, alg_params, grid_map, goals)
        pass

    def update_neighbors_info(self, neighbors_info: List[Message]) -> None:
        pass

    def compute_action(self) -> npt.NDArray:
        return np.array(random.choice(list(Action)).value)

    def update_state_info(self, new_pos: npt.NDArray) -> None:
        pass

    def send_message(self) -> Message:
        return Message()


class SmartRandomAgent(Agent):
    def __init__(self,
                 a_id: int,
                 ag_params: DiscreteAgentParams,
                 alg_params: BaseAlgParams,
                 grid_map: npt.NDArray,
                 goals: npt.NDArray):
        super().__init__(a_id, ag_params, alg_params, grid_map, goals)
        self.pos = None

    def update_neighbors_info(self, neighbors_info: List[Message]) -> None:
        pass

    def compute_action(self) -> npt.NDArray:
        actions = list(Action)
        actions.remove(Action.WAIT)
        while len(actions):
            action = random.choice(actions)
            actions.remove(action)
            action = np.array(action.value)
            predicted_pos = self.pos + action
            h, w = self.grid_map.shape
            i, j = predicted_pos
            if not ((0 <= i < h) and (0 <= j < w)):
                continue
            if self.grid_map[i, j]:
                continue

            return action

        return np.array(Action.WAIT.value)

    def update_state_info(self, new_pos: npt.NDArray) -> None:
        self.pos = new_pos

    def send_message(self) -> Message:
        return Message()


class AStarAgent(Agent):
    def __init__(self,
                 a_id: int,
                 ag_params: DiscreteAgentParams,
                 alg_params: BaseAlgParams,
                 grid_map: npt.NDArray,
                 goals: npt.NDArray):
        super().__init__(a_id, ag_params, alg_params, grid_map, goals)
        self.pos = None
        self.neighbors_info = None
        self.path = []
        self.goal_chosen = False
        self.goal = None
        self.search_map = Map(self.grid_map)
        self.path_exist = False

    def update_neighbors_info(self, neighbors_info: List[Message]) -> None:
        self.neighbors_info = neighbors_info

    def compute_action(self) -> npt.NDArray:

        if not self.goal_chosen:
            self.choose_goal()
            start_i, start_j = self.pos
            goal_i, goal_j = self.goal
            path_found, last_node, length = astar_search(self.search_map, start_i, start_j, goal_i, goal_j,
                                                         manhattan_distance)
            self.path = make_path(last_node)[:-1]

        if not self.path_exist or len(self.path) == 0:
            return np.array(Action.WAIT.value)
        next_pos = np.array(self.path.pop())
        action = (next_pos - self.pos)
        return action

    def update_state_info(self, new_pos: npt.NDArray) -> None:
        self.pos = new_pos

    def send_message(self) -> Message:
        message = Message()
        message.pos = self.pos
        return message

    def choose_goal(self) -> None:
        if self.goal_chosen:
            return

        start_i, start_j = self.pos
        min_len = np.inf

        for goal_i, goal_j in self.goals:
            path_found, last_node, length = astar_search(self.search_map, start_i, start_j, goal_i, goal_j,
                                                         manhattan_distance)
            if not path_found:
                continue
            self.path_exist = True
            if length < min_len:
                min_len = length
                self.goal = np.array((goal_i, goal_j))
