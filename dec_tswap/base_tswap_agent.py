from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from manavlib.common.params import BaseDiscreteAgentParams, BaseAlgParams
import numpy.typing as npt
import numpy as np
from dec_tswap.agent import Agent
from dec_tswap.message import Message
from dec_tswap.map import Map
from dec_tswap.action import Action


class BaseTSWAPParams(BaseAlgParams):
    alg_name = "base_tswap"

    def __init__(self) -> None:
        super().__init__()
        self.centralized = False


class BaseTSWAPAgent(Agent):
    """
    A class that implements a basic adaptation of the TSWAP algorithm for 
    decentralized operation, supporting distributed execution, but 
    requiring a consistent initial assignment. Also supports centralized mode.
    """
    def __init__(
        self,
        a_id: int,
        pos: npt.NDArray,
        ag_params: BaseDiscreteAgentParams,
        alg_params: BaseTSWAPParams,
        grid_map: npt.NDArray,
        goals: npt.NDArray,
        search_object,
    ):
        super().__init__(
            a_id, pos, ag_params, alg_params, grid_map, goals, search_object
        )
        self.neighbors_info: List[Message] | None = None
        self.goal_chosen = False
        self.goal = None
        self.search_map = Map(self.grid_map)
        self.path_exist = False
        self.my_message_it = None
        self.goal_updated = False
        self.updated_pos = None
        self.next_pos = None

    def initialize(self) -> bool:
        self.goal = self.goals[self.a_id]
        self.find_next()
        return self.path_exist

    def update_neighbors_info(self, neighbors_info: List[Message]) -> None:
        self.neighbors_info = neighbors_info
        self.my_message_it = None
        for i, n_info in enumerate(neighbors_info):
            if n_info.id == self.a_id:
                self.my_message_it = i
        assert self.my_message_it is not None

    def compute_action(self) -> npt.NDArray:

        self.update_goals()
        if not self.path_exist:
            return np.array(Action.WAIT.value)
        action = self.updated_pos - self.pos
        return action

    def update_state_info(self, new_pos: npt.NDArray) -> None:
        self.pos = new_pos
        self.find_next()

    def send_message(self) -> Message:
        message = Message()
        message.id = self.a_id
        message.pos = self.pos.copy()
        if self.path_exist:
            message.goal = self.goal.copy()
            message.next_pos = self.next_pos.copy()
        return message

    def find_next(self):
        result = self.search_object.find_next(self.pos, self.goal)
        if result is not None:
            self.path_exist = True
            self.next_pos = result
        else:
            self.path_exist = False
            self.next_pos = None

    def update_goals(self):

        self.goal_updated = False

        def update_next(search_map, pos, goal):
            next_pos = self.search_object.find_next(pos, goal)
            assert next_pos is not None
            return next_pos

        pos_table = {tuple(a.pos): a for a in self.neighbors_info}
        for a_info in self.neighbors_info:
            if tuple(a_info.next_pos) not in pos_table:
                del pos_table[tuple(a_info.pos)]
                pos_table[tuple(a_info.next_pos)] = a_info
                a_info.pos = a_info.next_pos
                a_info.next_pos = update_next(self.search_map, a_info.pos, a_info.goal)
                continue

            if np.all(a_info.pos == a_info.goal):
                continue

            b_info = pos_table[tuple(a_info.next_pos)]

            if np.all(a_info.next_pos == b_info.goal):
                a_info.goal, b_info.goal = b_info.goal, a_info.goal
                a_info.next_pos = a_info.goal
                b_info.next_pos = update_next(self.search_map, b_info.pos, b_info.goal)
                continue

            elif (deadlock_queue := self.check_deadlock(a_info, pos_table)) is not None:
                prev_goal = deadlock_queue[-1].goal
                for i in range(len(deadlock_queue)):
                    c_info = deadlock_queue[i]
                    tmp_goal = c_info.goal
                    c_info.goal = prev_goal
                    c_info.next_pos = update_next(
                        self.search_map, c_info.pos, c_info.goal
                    )
                    prev_goal = tmp_goal

        self.updated_pos = self.neighbors_info[self.my_message_it].pos
        if not np.all(self.neighbors_info[self.my_message_it].goal == self.goal):
            self.goal = self.neighbors_info[self.my_message_it].goal
            self.goal_updated = True

    def check_deadlock(self, a_info, pos_table):
        deadlock_queue = [a_info]
        deadlock_set = {a_info.id}

        def get_next(a_i: Message) -> Message | None:
            if tuple(a_i.next_pos) in pos_table:
                return pos_table[tuple(a_i.next_pos)]
            return None

        b_info = a_info
        while True:
            b_info = get_next(b_info)
            if b_info is None or np.all(b_info.pos == b_info.goal):
                return None
            if a_info.id == b_info.id:
                return deadlock_queue
            if b_info.id in deadlock_set:
                return None
            deadlock_queue.append(b_info)
            deadlock_set.add(b_info.id)
