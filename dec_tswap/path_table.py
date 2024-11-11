from dec_tswap.map import Map, compute_cost
from dec_tswap.search_tree import SearchTree
from dec_tswap.node import Node
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import numpy as np
import numpy.typing as npt


def dijkstra_search(
    task_map: Map, start_i: int, start_j: int
) -> Tuple[bool, Optional[Node], Optional[int]]:
    """
    Performs Dijkstra's search algorithm to find the shortest path in a grid map.

    Parameters
    ----------
    task_map : Map
        The map on which the search is conducted.
    start_i : int
        The starting row index for the search.
    start_j : int
        The starting column index for the search.

    Returns
    -------
    tuple
        A tuple containing:
        - bool : Indicates whether a path was found.
        - Optional[List[Node]] : List of expanded nodes if the path exists; otherwise, None.
        - int : The number of steps taken during the search.
    """
    ast = SearchTree()
    steps = 0
    start_node = Node(start_i, start_j, g=0)
    ast.add_to_open(start_node)

    while not ast.open_is_empty():
        current = ast.get_best_node_from_open()

        if current is None:
            break

        ast.add_to_closed(current)

        for i, j in task_map.get_neighbors(current.i, current.j):
            new_node = Node(i, j)
            if not ast.was_expanded(new_node):
                new_node.g = current.g + compute_cost(current.i, current.j, i, j)
                new_node.f = new_node.g
                new_node.parent = current
                ast.add_to_open(new_node)

        steps += 1

    return ast.expanded


class PathTable:
    """
    A class to store and retrieve precomputed shortest paths to various goals on a grid map.

    This class uses Dijkstra's algorithm to precompute and store the shortest paths from any position
    on the map to specified goal positions. The paths are stored in a dictionary for efficient lookup,
    enabling fast retrieval of the next step or the path length from any starting point to a given goal.
    """

    def __init__(self, grid_map: npt.NDArray, all_goals: npt.NDArray):
        """
        Initializes a PathTable object, which stores precomputed shortest paths to goals.

        Parameters
        ----------
        grid_map : np.ndarray
            The grid map where pathfinding is performed.
        all_goals : np.ndarray
            A 2D array of shape (n, 2) with list of n goal positions (i, j) for pathfinding.
        """
        self.search_map = Map(grid_map)
        self.goals_tables = dict()

        for goal in all_goals:
            expanded = dijkstra_search(self.search_map, goal[0], goal[1])
            self.goals_tables[tuple(goal)] = dict()
            for node in expanded:
                self.goals_tables[tuple(goal)][(node.i, node.j)] = (node.g, node.parent)

    def find_next(self, pos: npt.NDArray, goal: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Finds the next position on the path to the goal from a given position.

        Parameters
        ----------
        pos : np.ndarray
            The current position as an array (i, j).
        goal : np.ndarray
            The target goal position as an array [i, j].

        Returns
        -------
        Optional[np.ndarray]
            The next position as an array [i, j] if found; otherwise, None.
        """
        if np.all(pos == goal):
            return goal
        if tuple(pos) in self.goals_tables[tuple(goal)]:
            node = self.goals_tables[tuple(goal)][tuple(pos)][1]
            if node is None:
                return None
            return np.array([node.i, node.j], dtype=np.int32)
        else:
            return None

    def find_length(self, pos: npt.NDArray, goal: npt.NDArray) -> Optional[int]:
        """
        Finds the shortest path length from a given position to the goal.

        Parameters
        ----------
        pos : np.ndarray
            The current position as an array [i, j].
        goal : np.ndarray
            The target goal position as an array [i, j].

        Returns
        -------
        Optional[int]
            The path length to the goal if found; otherwise, None.
        """

        if tuple(pos) in self.goals_tables[tuple(goal)]:
            return self.goals_tables[tuple(goal)][tuple(pos)][0]
        else:
            return None
