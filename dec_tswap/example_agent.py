import copy
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, Any
from manavlib.common.params import BaseDiscreteAgentParams, BaseAlgParams
import numpy.typing as npt
import numpy as np
import random
from dec_tswap.message import Message
from dec_tswap.map import Map
from dec_tswap.action import Action
from dec_tswap.path_table import PathTable
from dec_tswap.agent import Agent


class RandomParams(BaseAlgParams):
    alg_name = "random"

    def __init__(self) -> None:
        super().__init__()


class SmartRandomParams(BaseAlgParams):
    alg_name = "smart_random"

    def __init__(self) -> None:
        super().__init__()


class ShortestPathParams(BaseAlgParams):
    alg_name = "shortest_path"

    def __init__(self) -> None:
        super().__init__()


class RandomAgent(Agent):
    """
    An agent that selects actions randomly for movement on a grid map.
    """

    def __init__(
        self,
        a_id: int,
        pos: npt.NDArray,
        ag_params: BaseDiscreteAgentParams,
        alg_params: BaseAlgParams,
        grid_map: npt.NDArray,
        goals: npt.NDArray,
        search_object: Optional[PathTable],
    ):
        super().__init__(
            a_id, pos, ag_params, alg_params, grid_map, goals, search_object
        )
        pass

    def initialize(self) -> bool:
        return True

    def update_neighbors_info(self, neighbors_info: List[Message]) -> None:
        pass

    def compute_action(self) -> npt.NDArray:
        return np.array(random.choice(list(Action)).value)

    def update_state_info(self, new_pos: npt.NDArray) -> None:
        pass

    def send_message(self) -> Message:
        return Message()


class SmartRandomAgent(Agent):
    """
    An agent that selects actions intelligently to avoid obstacles, choosing a random valid action.
    """

    def __init__(
        self,
        a_id: int,
        pos: npt.NDArray,
        ag_params: BaseDiscreteAgentParams,
        alg_params: BaseAlgParams,
        grid_map: npt.NDArray,
        goals: npt.NDArray,
        search_object: Optional[PathTable],
    ):
        super().__init__(
            a_id, pos, ag_params, alg_params, grid_map, goals, search_object
        )

    def initialize(self) -> bool:
        return True

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


class ShortestPathAgent(Agent):
    """
    An agent that navigates using the shortest path to a goal.
    """

    def __init__(
        self,
        a_id: int,
        pos: npt.NDArray,
        ag_params: BaseDiscreteAgentParams,
        alg_params: BaseAlgParams,
        grid_map: npt.NDArray,
        goals: npt.NDArray,
        search_object: PathTable,
    ):
        super().__init__(
            a_id, pos, ag_params, alg_params, grid_map, goals, search_object
        )
        self.neighbors_info = None
        self.goal_chosen = False
        self.goal = None
        self.path_exist = False

    def initialize(self) -> bool:
        if not self.goal_chosen:
            self.choose_goal()
            result = self.search_object.find_length(self.pos, self.goal)
            if result is None:
                self.path_exist = False
            else:
                self.path_exist = True
        return self.path_exist

    def update_neighbors_info(self, neighbors_info: List[Message]) -> None:
        self.neighbors_info = neighbors_info

    def compute_action(self) -> npt.NDArray:

        if not self.path_exist:
            return np.array(Action.WAIT.value)
        next_pos = self.search_object.find_next(self.pos, self.goal)
        action = next_pos - self.pos
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
        min_len = np.inf

        for goal in self.goals:
            result = self.search_object.find_length(self.pos, goal)
            if result is None:
                continue
            self.path_exist = True
            if result < min_len:
                min_len = result
                self.goal = np.array(goal)
