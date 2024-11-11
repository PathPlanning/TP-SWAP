from PIL import Image, ImageDraw, ImageOps, ImageFont
from PIL import ImageFilter
from random import randint
from typing import List, Optional, Tuple, Union, Set
import numpy.typing as npt
import numpy as np
from const import *
import cv2


def draw_grid(draw_obj: ImageDraw, grid_map: npt.NDArray, scale: Union[float, int]):
    """
    Draws static obstacles on the grid using the provided drawing object.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object used to render the grid elements.
    grid_map : np.ndarray
        A 2D array representing the grid map, where obstacles are marked as '1'.
    scale : float or int
        The scaling factor for drawing elements on the grid.
    """
    height, width = grid_map.shape
    gap = 0.15
    for row in range(height):
        for col in range(width):
            if grid_map[row, col] == MAP_OBSTACLE:
                top_left = ((col + gap) * scale, (row + gap) * scale)
                bottom_right = (
                    (col - gap + 1) * scale - 1,
                    (row - gap + 1) * scale - 1,
                )

                draw_obj.rounded_rectangle(
                    [top_left, bottom_right], fill=(70, 80, 80), width=0.0, radius=1
                )


def draw_goal(draw_obj: ImageDraw, goal: npt.NDArray, scale: Union[float, int]):
    """
    Draws goal positions on the grid using the provided drawing object.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object used to render the goal cell.
    goal : np.ndarray
        The coordinates of the goal cell.
    scale : float or int
        The scaling factor for drawing elements on the grid.
    """

    def draw_cell(cell, fill_color):
        top_left = ((cell[1] + 0.1) * scale, (cell[0] + 0.1) * scale)
        bottom_right = ((cell[1] + 0.9) * scale - 1, (cell[0] + 0.9) * scale - 1)
        draw_obj.rectangle([top_left, bottom_right], fill=fill_color, width=0.0)

    draw_cell(goal, (231, 76, 60))


def draw_dyn_object(
    draw_obj: ImageDraw,
    path: npt.NDArray,
    step: int,
    frame_num: int,
    frames_per_step: int,
    scale: Union[float, int],
    color: Tuple[int, int, int],
    outline_color: Optional[Tuple[int, int, int]],
    outline_width: int,
    circle: bool,
    neighbors: List[List[npt.NDArray]],
    draw_links: bool = True,
):
    """
    Draws a dynamic object (agent) at a specific position on the grid.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object used to render the dynamic object.
    path : np.ndarray
        The path of the dynamic object as an array of coordinates.
    step : int
        The current step index along the path.
    frame_num : int
        The current frame number for interpolation.
    frames_per_step : int
        The number of frames per movement step.
    scale : float or int
        The scaling factor for drawing.
    color : Tuple[int, int, int]
        The fill color for the dynamic object.
    outline_color : Optional[Tuple[int, int, int]]
        The outline color for the dynamic object.
    outline_width : int
        The width of the outline.
    circle : bool
        Whether to draw the object as a circle.
    neighbors : List[List[np.ndarray]]
        List of neighbors' positions for drawing links.
    draw_links : bool, optional
        Whether to draw lines connecting the agent to its neighbors (default is True).
    """
    path_len = len(path)
    curr_i, curr_j = path[min(path_len - 1, step)]
    next_i, next_j = path[min(path_len - 1, step + min(frame_num, 1))]

    di = frame_num * (next_i - curr_i) / frames_per_step
    dj = frame_num * (next_j - curr_j) / frames_per_step

    top_left = (float(curr_j + dj + 0.1) * scale, float(curr_i + di + 0.1) * scale)
    bottom_right = (
        float(curr_j + dj + 0.9) * scale - 1,
        float(curr_i + di + 0.9) * scale - 1,
    )

    outline_color = color if outline_color is None else outline_color

    if circle:
        draw_obj.ellipse(
            [top_left, bottom_right],
            fill=color,
            outline=outline_color,
            width=outline_width,
        )
    else:
        draw_obj.rectangle(
            [top_left, bottom_right],
            fill=color,
            outline=outline_color,
            width=outline_width,
        )

    if neighbors is None:
        return

    agent_pos = (float(curr_j + dj + 0.5) * scale, float(curr_i + di + 0.5) * scale)
    if draw_links:
        for neighbor_ij, neighbor_next_ij in neighbors:
            n_di = frame_num * (neighbor_next_ij[0] - neighbor_ij[0]) / frames_per_step
            n_dj = frame_num * (neighbor_next_ij[1] - neighbor_ij[1]) / frames_per_step
            neighbor_pos = (
                float(neighbor_ij[1] + n_dj + 0.5) * scale,
                float(neighbor_ij[0] + n_di + +0.5) * scale,
            )
            draw_obj.line(
                (agent_pos[0], agent_pos[1], neighbor_pos[0], neighbor_pos[1]),
                fill=(100, 100, 100),
            )


def create_frame(
    im: Image.Image,
    scale: int,
    step: int,
    quality: int,
    goals: npt.NDArray,
    paths: npt.NDArray,
    agent_colors: List[str],
    outline_colors: List[str],
    outline_width: int,
    neighbors: List[List[npt.NDArray]],
    draw_links: bool = True,
) -> List[Image.Image]:
    """
    Creates the animation frames with agents and their paths for one time step.

    Parameters
    ----------
    im : Image.Image
        The base image to draw on.
    scale : int
        The scaling factor for drawing elements.
    step : int
        The current step in the animation sequence.
    quality : int
        The quality level for interpolation between steps (number of frames per step).
    starts : np.ndarray
        Starting positions of the agents.
    goals : np.ndarray
        Goal positions of the agents.
    paths : np.ndarray
        Paths of the agents.
    agent_colors : List[str]
        Colors for the agents.
    outline_colors : List[str]
        Outline colors for the agents.
    outline_width : int
        Width of the outline.
    neighbors : List[List[np.ndarray]]
        Neighbor positions for each agent.
    draw_links : bool, optional
        Whether to draw links between agents and their neighbors (default is True).

    Returns
    -------
    List[Image.Image]
        A list of generated animation frames.
    """
    frames = []

    for n in range(quality):
        im_copy = im.copy()
        draw = ImageDraw.Draw(im_copy)
        agents_num = len(paths[0])
        for a_id in range(agents_num):
            goal = goals[a_id]
            draw_goal(draw, goal, scale)

        for a_id in range(agents_num):
            path = paths[:, a_id, :]
            agent_color = agent_colors[a_id % len(agent_colors)]
            outline_color = outline_colors[a_id % len(outline_colors)]
            draw_dyn_object(
                draw,
                path,
                step,
                n,
                quality,
                scale,
                agent_color,
                outline_color,
                outline_width,
                True,
                neighbors[a_id] if neighbors is not None else None,
                draw_links,
            )
        im_copy = im_copy.filter(ImageFilter.SMOOTH_MORE)
        frames.append(im_copy)
    return frames


def save_animation(images: List[Image.Image], output_filename: str):
    """
    Saves the generated animation frames as a video file.

    Parameters
    ----------
    images : List[Image.Image]
        List of frames to be saved as a video.
    output_filename : str
        The filename for the output video.
    """
    videodims = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video = cv2.VideoWriter(output_filename, fourcc, 30, videodims)
    for im in images:
        video.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    video.release()


def draw(
    grid_map: npt.NDArray,
    goals: npt.NDArray,
    paths: npt.NDArray,
    neighbors: List[List[Set[int]]],
    scale: int,
    outline_width: int,
    output_filename: str = "animated_trajectories",
):
    """
    Visualizes the environment and agent paths.

    Parameters
    ----------
    grid_map : np.ndarray
        A 2D array representing the grid map, where obstacles are marked as '1'.
    goals : np.ndarray
        Goal positions of the agents.
    paths : np.ndarray
        The paths of agents from start to goal.
    neighbors : List[List[Set[int]]]
        List of neighbor sets for each agent at each time step.
    scale : int
        The scaling factor for drawing.
    outline_width : int
        The width of the outline around agents.
    output_filename : str, optional
        The filename for the output animation (default is "animated_trajectories").
    """
    quality = 5
    height, width = grid_map.shape
    agents_num = len(paths[0])
    outline_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    agent_colors = [
        "#1f77b4c8",
        "#ff7f0ec8",
        "#2ca02cc8",
        "#d62728c8",
        "#9467bdc8",
        "#8c564bc8",
        "#e377c2c8",
        "#7f7f7fc8",
        "#bcbd22c8",
        "#17becfc8",
    ]

    max_time = max((len(paths[:, a_id, :]) for a_id in range(agents_num)), default=1)
    neighbors_poses = compute_neighbors_poses(paths, neighbors)
    images = []
    im = Image.new("RGBA", (width * scale, height * scale), color=(255, 255, 255))
    draw_orig = ImageDraw.Draw(im)
    draw_grid(draw_orig, grid_map, scale)
    for step in range(75):
        images.extend(
            create_frame(
                im,
                scale,
                0,
                1,
                goals,
                paths,
                agent_colors,
                outline_colors,
                outline_width,
                neighbors_poses[0] if 0 < len(neighbors_poses) else None,
                False,
            )
        )

    for step in range(max_time):
        images.extend(
            create_frame(
                im,
                scale,
                step,
                quality,
                goals,
                paths,
                agent_colors,
                outline_colors,
                outline_width,
                neighbors_poses[step] if step < len(neighbors_poses) else None,
            )
        )

    for step in range(75):
        images.extend(
            create_frame(
                im,
                scale,
                -1,
                1,
                goals,
                paths,
                agent_colors,
                outline_colors,
                outline_width,
                neighbors_poses[0] if 0 < len(neighbors_poses) else None,
                False,
            )
        )
    save_animation(images, output_filename)


def compute_neighbors_poses(
    step_log: npt.NDArray, neighbors_ids: List[List[Set[int]]]
) -> List[List[Tuple[npt.NDArray, npt.NDArray]]]:
    """
    Computes the positions of neighbors for each agent at each time step.

    Parameters
    ----------
    step_log : np.ndarray
        The log of agent positions for each time step.
    neighbors_ids : List[List[Set[int]]]
        List of sets of neighbor indices for each agent at each time step.

    Returns
    -------
    List[List[Tuple[np.ndarray, np.ndarray]]]
        A list of lists containing tuples of current and next positions for each neighbor.
    """
    neighbors_poses = []
    for step in range(len(step_log) - 1):
        neighbors_ids_step = neighbors_ids[step]
        neighbors_poses_step = []
        for a_id in range(len(step_log[step])):
            neighbors_poses_agent = []
            for n_id in neighbors_ids_step[a_id]:
                neighbors_poses_agent.append(
                    (step_log[step][n_id], step_log[step + 1][n_id])
                )
            neighbors_poses_step.append(neighbors_poses_agent)
        neighbors_poses.append(neighbors_poses_step)
    return neighbors_poses
