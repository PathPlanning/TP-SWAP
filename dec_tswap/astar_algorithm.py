from dec_tswap.map import Map, compute_cost
from dec_tswap.search_tree import SearchTree
from dec_tswap.node import Node
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import numpy as np


def manhattan_distance(i1: int, j1: int, i2: int, j2: int) -> int:
    """
    Computes the Manhattan distance between two cells on a grid.

    Parameters
    ----------
    i1, j1 : int
        (i, j) coordinates of the first cell on the grid.
    i2, j2 : int
        (i, j) coordinates of the second cell on the grid.

    Returns
    -------
    int
        Manhattan distance between the two cells.
    """
    return abs(i1 - i2) + abs(j1 - j2)


def astar_search(
        task_map: Map,
        start_i: int,
        start_j: int,
        goal_i: int,
        goal_j: int,
        heuristic_func: Callable
) -> Tuple[bool, Optional[Node], Optional[int]]:
    """
    Implements the A* search algorithm.

    Parameters
    ----------
    task_map : Map
        The grid or map being searched.
    start_i, start_j : int, int
        Starting coordinates.
    goal_i, goal_j : int, int
        Goal coordinates.
    heuristic_func : Callable
        Heuristic function for estimating the distance from a node to the goal.

    Returns
    -------
    Tuple[bool, Optional[Node], int, int, Optional[Iterable[Node]], Optional[Iterable[Node]]]
        Tuple containing:
        - A boolean indicating if a path was found.
        - The last node in the found path or None.
        - Path length
    """
    ast = SearchTree()
    steps = 0
    start_node = Node(start_i, start_j, g=0, h=heuristic_func(start_i, start_j, goal_i, goal_j))
    ast.add_to_open(start_node)

    while not ast.open_is_empty():
        current = ast.get_best_node_from_open()

        if current is None:
            break

        ast.add_to_closed(current)

        if (current.i, current.j) == (goal_i, goal_j):
            return True, current, current.g

        for i, j in task_map.get_neighbors(current.i, current.j):
            new_node = Node(i, j)
            if not ast.was_expanded(new_node):
                new_node.g = current.g + compute_cost(current.i, current.j, i, j)
                new_node.h = heuristic_func(i, j, goal_i, goal_j)
                new_node.f = new_node.g + new_node.h
                new_node.parent = current
                ast.add_to_open(new_node)

        steps += 1

    return False, None, None


def make_path(goal: Node) -> List[Node]:
    """
    Creates a path by tracing parent pointers from the goal node to the start node.
    It also returns the path's length.

    Parameters
    ----------
    goal : Node
        Pointer to the goal node in the search tree.

    Returns
    -------
    Tuple[List[Node], float]
        Path and its length.
    """
    current = goal
    path = []
    while current.parent:
        path.append((current.i, current.j))
        current = current.parent
    path.append(current)
    return path
