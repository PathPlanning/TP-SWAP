from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from manavlib.common.params import BaseDiscreteAgentParams, BaseAlgParams
import numpy.typing as npt
import numpy as np
from dec_tswap.agent import Agent
from dec_tswap.message import Message
from dec_tswap.map import Map
from dec_tswap.action import Action
from dec_tswap.path_table import PathTable
import heapq as pq


class NaiveDecTSWAPParams(BaseAlgParams):
    alg_name = "naive_dec_tswap"

    def __init__(self) -> None:
        super().__init__()


class NaiveDecTSWAPAgent(Agent):
    """
    A class that implements a naive adaptation of the TSWAP algorithm for a fully decentralized setting.
    """
    
    def __init__(
            self,
            a_id: int,
            pos: npt.NDArray,
            ag_params: BaseDiscreteAgentParams,
            alg_params: NaiveDecTSWAPParams,
            grid_map: npt.NDArray,
            goals: npt.NDArray,
            search_object: PathTable
    ):
        super().__init__(a_id, pos, ag_params, alg_params, grid_map, goals, search_object)
        self.neighbors_info: List[Message] | None = None
        self.goal_chosen = False
        self.goal = None
        self.search_map = Map(self.grid_map)
        self.path_exist = False
        self.my_message_it = None
        self.goal_updated = False
        self.updated_pos = None
        self.next_pos = None
        self.achieved_goals = set()

    def initialize(self) -> bool:
        if not self.goal_chosen:
            self.choose_goal()
            self.find_next()
            if self.path_exist:
                self.goal_chosen = True
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
        message.achieved_goals = self.achieved_goals
        if self.path_exist:
            message.goal = self.goal.copy()
            message.next_pos = self.next_pos.copy()
        return message

    def choose_goal(self) -> None:
        if self.goal_chosen:
            return
        min_len = np.inf
        for goal in self.goals:
            length = self.search_object.find_length(self.pos, goal)
            if length is None:
                continue
            self.path_exist = True
            if length < min_len:
                min_len = length
                self.goal = np.array(goal)

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
        self.remove_incons(pos_table)

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
                a_info.priority, b_info.priority = b_info.priority, a_info.priority

                a_info.next_pos = a_info.goal
                b_info.next_pos = update_next(self.search_map, b_info.pos, b_info.goal)

                continue

            elif (deadlock_queue := self.check_deadlock(a_info, pos_table)) is not None:

                prev_goal = deadlock_queue[-1].goal
                prev_priority = deadlock_queue[-1].priority
                for i in range(len(deadlock_queue)):
                    c_info = deadlock_queue[i]
                    tmp_goal = c_info.goal
                    tmp_priority = c_info.priority
                    c_info.goal = prev_goal
                    c_info.priority = prev_priority
                    c_info.next_pos = update_next(
                        self.search_map, c_info.pos, c_info.goal
                    )
                    prev_goal = tmp_goal
                    prev_priority = tmp_priority

        self.updated_pos = self.neighbors_info[self.my_message_it].pos
        self.achieved_goals = self.neighbors_info[self.my_message_it].achieved_goals
        if not np.all(self.neighbors_info[self.my_message_it].goal == self.goal):
            self.goal = self.neighbors_info[self.my_message_it].goal
            self.goal_updated = True

    def remove_incons(self, pos_table):
        merged_achieved_goals = set()
        for a_info in self.neighbors_info:
            merged_achieved_goals = merged_achieved_goals.union(a_info.achieved_goals)

        for a_info in self.neighbors_info:
            a_info.achieved_goals = merged_achieved_goals
            curr_pos = tuple(a_info.pos)
            curr_next_pos = tuple(a_info.next_pos)
            curr_goal = tuple(a_info.goal)

            if curr_pos == curr_goal:
                continue
            if curr_next_pos != curr_goal:
                continue
            if curr_next_pos not in pos_table:
                continue
            if tuple(pos_table[curr_next_pos].goal) != curr_goal:
                continue

            a_info.achieved_goals.add(tuple(curr_goal))
            closest_goal = (-1, -1)
            closest_goal_dist = np.inf
            for goal in self.goals:

                if tuple(goal) in a_info.achieved_goals:
                    continue

                goal_dist = self.search_object.find_length(a_info.pos, goal)
                if closest_goal_dist < goal_dist:
                    continue
                closest_goal = goal
                closest_goal_dist = goal_dist

            assert not (tuple(closest_goal) == (-1, -1))
            assert closest_goal_dist < np.inf
            a_info.goal = closest_goal
            a_info.next_pos = self.search_object.find_next(a_info.pos, a_info.goal)


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
