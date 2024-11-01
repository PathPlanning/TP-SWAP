# Decentralized Unlabeled Multi-agent Pathfinding Via Target And Priority Swapping: Experiments

> [!Important]  
> **This repository contains code and data for the paper:**
>
> Dergachev S., Yakovlev K. *Decentralized Unlabeled Multi-agent Pathfinding via Target and Priority Swapping*. Proceedings of the 27th European Conference on Artificial Intelligence (ECAI 2024). IOS Press, 2024, pp. 4344–4351.
>
> **[[Full text in Proceedings of ECAI 2024](https://ebooks.iospress.nl/volumearticle/70105)]**
>
> **[[Full text with supplementary materials on arXiv](https://arxiv.org/abs/2408.14948)]**

## Description

This repository includes experimental results and code for processing and visualizing these results, as detailed in the above paper.

## Table of Content

1. [Repository Structure](#repository-structure)
2. [Requirements](#requirements)
3. [Experimantal Results](#experimantal-results)
4. [References](#references)

## Repository Structure

* `results/`: Contains experimental results organized by map and algorithm:
  * `*map_name*/`: Each map used in the experiments has its own folder.
    * `*algorithm_name*/`: Within each map folder, there are subfolders named after the evaluated algorithms.
      * `result.txt`: Contains evaluation metrics for each algorithm, including **__flowtime__** and **__makespan__**.
  * `all/`: This subfolder contains all intermediate results of experiments obtained during the preparation of the paper.
* `processing/`: Contains code for processing experimental results.
  * `results_processing.ipynb`: A Jupyter notebook for processing the data in `results/` and generating tables and plots to visualize the results.
  * `img/`: Contains plots generated from processing the experimental results.

## Requirements

To use the repository, install the following software and libraries:

* `Python 3.11`
* `IPython and Jupyter Notebook`
* `numpy`
* `pandas`
* `matplotlib`

## Experimantal Results

The `results/` folder contains experimental data for various grid maps and AMAPF solvers. The following maps from the MovingAI benchmarks [[1](https://arxiv.org/abs/1906.08291), [2](https://www.movingai.com/)] were used:

![Maps](img/maps.svg)

The next algorithms were evaluated in the experiments:

- **`dec-tswap` or `TP-SWAP`**: The proposed decentralized AMAPF solver.
  - `dec-tswap-2`, `dec-tswap-5`, and `dec-tswap-10`: Variants with different communication ranges:
    - `dec-tswap-2`: Sight radius of `2` cells, equivalent to a `5x5` communication range.
    - `dec-tswap-5`: Sight radius of `5` cells, equivalent to an `11x11` communication range.
    - `dec-tswap-10`: Sight radius of `10` cells, equivalent to a `21x21` communication range.
- **`origin-tswap` or `C-TSWAP`**: A centralized AMAPF solver (state-of-the-art) [[3](https://arxiv.org/abs/2109.04264)], with results obtained using the original implementation [[4](https://github.com/Kei18/tswap)].
- **`base-tswap` or `D-TSWAP-C`**: A semi-decentralized AMAPF solver based on TSWAP with a consistent initial assignment and decentralized operation.
- **`naive-dec-tswap` or `D-TSWAP-N`**: A naive, fully decentralized AMAPF solver based on TSWAP.


Each `result.txt` file contains metrics from multiple experimental runs for the corresponding algorithm on the specific map. Each line in the file represents one experimental run and includes the following columns:

* `success`: Indicates whether the experiment successfully completed (`1` for success, `0` for failure).
* `collision`: Total number of collisions between agents.
* `collision_obst`: Number of collisions between agents and obstacles.
* `makespan`: The total time (steps) taken until the last agent reaches its target.
* `flowtime`: The sum of all agents’ travel times (in steps).
* `runtime`: Computation time for the experiment.
* `mean_groups`: Average number of groups formed during the experiment.
* `mean_groups_size`: Average size of each group.
* `number`: Number of agents involved in the experiment.

The notebook `processing/results_processing.ipynb` can be used to load, process, and visualize this data. Run the notebook to produce tables and plots for metrics like **__flowtime__** and **__makespan__**, as presented in the paper. Plots generated by processing the results are saved in the `processing/img/` subfolder.

## References

1. [Stern R. et al. *Multi-agent pathfinding: Definitions, variants, and benchmarks*. Proceedings of the 12th Annual Symposium on Combinatorial Search (SoCS 2019), 2019, pp. 151–158.](https://arxiv.org/abs/1906.08291)
2. [MovingAI Lab](https://www.movingai.com/)
3. [K. Défago X. *Solving Simultaneous Target Assignment and Path Planning Efficiently with Time-Independent Execution*. Proceedings of the International Conference on Automated Planning and Scheduling (ICAPS 2022), vol. 32, 2022, pp. 270–278.](https://arxiv.org/abs/2109.04264)
4. [GitHub repository with implementation of TSWAP algorithm](https://github.com/Kei18/tswap)
  