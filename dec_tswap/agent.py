from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, Any
from manavlib.common.params import BaseDiscreteAgentParams, BaseAlgParams
import numpy.typing as npt
import numpy as np
from dec_tswap.message import Message
from dec_tswap.action import Action
from dec_tswap.path_table import PathTable


class Agent:
    """
    A base class for representing an agent in a multi-agent navigation system.
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
        """
        Initialize agent's attributes.

        Parameters
        ----------
        a_id : int
            The agent's unique identifier.
        pos : np.ndarray
            The current position of the agent on the grid.
        ag_params : BaseDiscreteAgentParams
            Agent-specific parameters.
        alg_params : BaseAlgParams
            Algorithm-related parameters.
        grid_map : np.ndarray
            The grid map where the agent navigates.
        goals : np.ndarray
            Array containing the agent's possible goal positions.
        search_object : Optional[PathTable]
            Search object for precomputed shortest paths, if needed.
        """
        self.a_id = a_id
        self.pos = pos
        self.ag_params = ag_params
        self.alg_params = alg_params
        self.grid_map = grid_map
        self.goals = goals
        self.search_object = search_object

    def initialize(self) -> bool:
        """Initializes the agent"""
        raise NotImplementedError

    def update_neighbors_info(self, neighbors_info: List[Message]) -> None:
        """
        Updates information about neighboring agents based on received messages.

        Parameters
        ----------
        neighbors_info : List[Message]
            List of messages containing state information from neighboring agents.
        """
        raise NotImplementedError

    def compute_action(self) -> Action:
        """
        Computes the next action for the agent based on its current state and environment.

        Returns
        -------
        Action
            The computed action that the agent should take.
        """
        raise NotImplementedError

    def update_state_info(self, new_pos: npt.NDArray) -> None:
        """
        Updates the agent's state information, specifically its position on the grid.

        Parameters
        ----------
        new_pos : np.ndarray
            The new position of the agent as a NumPy array [i, j].
        """
        raise NotImplementedError

    def send_message(self) -> Message:
        """
        Creates and returns a message containing the agent's current state information.

        Returns
        -------
        Message
            A message containing the agent's current position, goal, and other relevant data.
        """
        raise NotImplementedError
