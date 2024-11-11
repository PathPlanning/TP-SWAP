import numpy.typing as npt
from typing import Dict, Tuple, Set

class Message:
    """
    A class to represent a message in a multi-agent navigation system, containing information about an agent's 
    current position, next position, goal, priority, etc.

    Attributes
    ----------
    id : Optional[int]
        The unique identifier of the agent.
    pos : Optional[np.ndarray]
        The current position of the agent as a NumPy array (e.g., [i, j] for a grid position).
    next_pos : Optional[np.ndarray]
        The next intended position of the agent as a NumPy array.
    goal : Optional[np.ndarray]
        The target goal position of the agent as a NumPy array.
    priority : Optional[int]
        The priority level of the agent, used for resolving conflicts.
    goals_priorities : Optional[Dict[Tuple[int, int], int]]
        A dictionary mapping each goal (represented as a tuple of coordinates) to the priority of the agent that moves to this goal
    achieved_goals : Optional[Set[Tuple[int, int]]]
        A set of achieved goals, each represented as a tuple of coordinates.
    """
    def __init__(self):
        self.id: int | None = None
        self.pos: npt.NDArray | None = None
        self.next_pos: npt.NDArray | None = None
        self.goal: npt.NDArray | None = None
        self.priority: int | None = None
        self.goals_priorities: Dict[Tuple[int, int], int] | None = None
        self.achieved_goals: Set[Tuple[int, int]] | None = None

    def __str__(self):
        return f"(id: {self.id}, pos: {self.pos}, next: {self.next_pos}, goal: {self.goal})"

    def __repr__(self):
        return f"(id: {self.id}, pos: {self.pos}, next: {self.next_pos}, goal: {self.goal})"
