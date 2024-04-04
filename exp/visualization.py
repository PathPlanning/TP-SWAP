import time

from PIL import Image, ImageDraw, ImageOps
from PIL import ImageFilter
from random import randint
from typing import List, Optional, Tuple, Union
import numpy.typing as npt
from const import *
import copy


def draw_grid(draw_obj: ImageDraw, grid_map: npt.NDArray, scale: Union[float, int]):
    """
    Draws static obstacles on the grid using draw_obj.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object to use.
    grid_map : np.ndarray
        The grid map containing obstacle information.
    scale : float or int
        The scaling factor for drawing.
    """
    height, width = grid_map.shape
    for row in range(height):
        for col in range(width):
            if grid_map[row, col] == MAP_OBSTACLE:
                top_left = (col * scale, row * scale)
                bottom_right = ((col + 1) * scale - 1, (row + 1) * scale - 1)
                draw_obj.rectangle(
                    [top_left, bottom_right], fill=(70, 80, 80), width=0.0
                )


def draw_start_goal(
        draw_obj: ImageDraw,
        start: npt.NDArray,
        goal: npt.NDArray,
        scale: Union[float, int],
):
    """
    Draws start and goal cells on the grid using draw_obj.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object to use.
    start : np.ndarray
        The start cell coordinates.
    goal : np.ndarray
        The goal cell coordinates.
    scale : float or int
        The scaling factor for drawing.
    """

    def draw_cell(cell, fill_color):
        top_left = ((cell[1] + 0.1) * scale, (cell[0] + 0.1) * scale)
        bottom_right = ((cell[1] + 0.9) * scale - 1, (cell[0] + 0.9) * scale - 1)
        draw_obj.rounded_rectangle(
            [top_left, bottom_right], fill=fill_color, width=0.0, radius=scale * 0.22
        )

    draw_cell(start, (40, 180, 99))  # Start cell color
    draw_cell(goal, (231, 76, 60))  # Goal cell color


def draw_dyn_object(
        draw_obj: ImageDraw,
        path: npt.NDArray,
        step: int,
        frame_num: int,
        frames_per_step: int,
        scale: Union[float, int],
        color: Tuple[int, int, int],
        outline_color: Tuple[int, int, int] | None,
        outline_width: int,
        circle: bool,
):
    """
    Draws the position of a dynamic object at a specific time using draw_obj.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object to use.
    path : np.ndarray
        The path of the dynamic object.
    step : int
        The current step in the path.
    frame_num : int
        The current frame number.
    frames_per_step : int
        The number of frames per step.
    scale : float or int
        The scaling factor for drawing.
    color : Tuple[int, int, int]
        The fill color for the object.
    circle : bool
        Whether to draw the object as a circle.
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
        draw_obj.ellipse([top_left, bottom_right], fill=color, outline=outline_color, width=outline_width)
    else:
        draw_obj.rectangle([top_left, bottom_right], fill=color, outline=outline_color, width=outline_width)


def create_frame(
        grid_map,
        scale,
        width,
        height,
        step,
        quality,
        starts,
        goals,
        paths,
        agent_colors,
        outline_colors,
        outline_width
):
    frames = []
    im = Image.new("RGBA", (width * scale, height * scale), color=(234, 237, 237))
    draw_orig = ImageDraw.Draw(im)
    draw_grid(draw_orig, grid_map, scale)

    for n in range(quality):
        im_copy = im.copy()
        draw = ImageDraw.Draw(im_copy)
        agents_num = len(paths[0])
        for a_id in range(agents_num):
            start = starts[a_id]
            goal = goals[a_id]
            draw_start_goal(draw, start, goal, scale)

        for a_id in range(agents_num):
            # for path, agent_color in zip(paths, agent_colors):
            path = paths[:, a_id, :]
            agent_color = agent_colors[a_id % len(agent_colors)]
            outline_color = outline_colors[a_id % len(outline_colors)]
            draw_dyn_object(
                draw, path, step, n, quality, scale, agent_color, outline_color, outline_width, True
            )

        im_copy = ImageOps.expand(im_copy, border=2, fill="black")
        im_copy = im_copy.filter(ImageFilter.SMOOTH_MORE)
        frames.append(im_copy)
    return frames


def save_animation(images, output_filename, quality):
    # Maybe replace with openCV and add antialiasing
    images[0].save(
        f"{output_filename}.png",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        loop=0,
    )


def draw(
        grid_map: npt.NDArray,
        starts: npt.NDArray,
        goals: npt.NDArray,
        paths: npt.NDArray,
        scale: int,
        outline_width: int,
        output_filename: str = "animated_trajectories",
):
    """
    Visualizes the environment, agent paths, and dynamic obstacles trajectories.

    Parameters
    ----------
    grid_map : Map
        Environment representation as a grid.
    starts : np.ndarray
        Starting positions of agents.
    goals : np.ndarray
        Goal positions of agents.
    paths : np.ndarray
        Paths of agents between start and goal positions.
    scale : int
        TODO
    output_filename : str
        Name of the file for the resulting animated visualization.
    """
    quality = 5
    height, width = grid_map.shape
    agents_num = len(paths[0])
    outline_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf']
    agent_colors = ['#1f77b4c8', '#ff7f0ec8', '#2ca02cc8', '#d62728c8', '#9467bdc8', '#8c564bc8', '#e377c2c8',
                    '#7f7f7fc8',
                    '#bcbd22c8', '#17becfc8']

    max_time = max((len(paths[:, a_id, :]) for a_id in range(agents_num)), default=1)
    images = []
    for step in range(max_time):
        images.extend(
            create_frame(
                grid_map,
                scale,
                width,
                height,
                step,
                quality,
                starts,
                goals,
                paths,
                agent_colors,
                outline_colors,
                outline_width
            )
        )
    save_animation(images, output_filename, quality)
