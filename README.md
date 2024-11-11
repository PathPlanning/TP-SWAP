# Decentralized Unlabeled Multi-agent Pathfinding Via Target And Priority Swapping

<p align="center">
<img src="img/animation.gif" alt="Animation" width="500"/>
<p\>

> [!Important]  
> **This repository contains code for the paper:**
>
> Dergachev S., Yakovlev K. *Decentralized Unlabeled Multi-agent Pathfinding via Target and Priority Swapping*. Proceedings of the 27th European Conference on Artificial Intelligence (ECAI 2024). IOS Press, 2024, pp. 4344–4351.
>
> **[[Full text in Proceedings of ECAI 2024](https://ebooks.iospress.nl/volumearticle/70105)]**
>
> **[[Full text with supplementary materials on arXiv](https://arxiv.org/abs/2408.14948)]**

## Overview

This repository contains the source code for algorithms designed for decentralized anonymous/unlabeled multi-agent pathfinding (AMAPF), as described in the referenced paper. The focus of this project is on decentralized approaches, and it includes several baseline algorithms as well as the proposed **TP-SWAP** method. A more detailed description of these algorithms can be found in the paper.

### Supported Algorithms

The following baseline policies and AMAPF algorithms are implemented:

* `random`: A baseline policy where agents move randomly without a specific strategy.
* `smart_random`: An improved `random` policy where agents agents move randomly, but avoid  collisions with obstacles.
* `shortest_path`: Agents follow the shortest path to their closest targets using precomputed paths.
* `base_tswap` or `D-TSWAP-C`: A semi-decentralized solver based on TSWAP with consistent initial assignment and decentralized operation.
* `naive_dec_tswap` or `D-TSWAP-N`: A naive fully decentralized solver based on TSWAP.
* `dec_tswap` or `TP-SWAP`: The proposed fully decentralized solver.

This repository can also be used as a base framework for implementing custom AMAPF algorithms.

## Repository Structure

The repository is organized into several branches:

* **`main` branch** [[**Link**](https://github.com/PathPlanning/TP-SWAP/tree/main)]: Contains the core implementation of the AMAPF algorithms, including all supported methods.
* **`experiments` branch** [[**Link**](https://github.com/PathPlanning/TP-SWAP/tree/experiments)]: Includes scripts and resources for running full-scale experiments, as described in the referenced paper. This branch provides tools for task generation, experiment execution, and result analysis.
* **`supplementary` branch** [[**Link**](https://github.com/PathPlanning/TP-SWAP/tree/supplementary)]: Contains extended experimental results analysis. This branch stands as supplementary material for the paper.

## Installation and Launch

### Main Requirements

To use the repository, install the following software and libraries:

* `Python 3.11`
* `setuptools`
* `numpy`
* `manavlib (v1.0)`

The `manavlib` library provides utility functions for multi-agent navigation experiments, including tools for handling XML configuration files, generating tasks, and creating maps.

**Installation:** You can find detailed installation instructions in the `manavlib` GitHub repository. In brief, to install `manavlib` directly from its [[**GitHub**]](https://github.com/haiot4105/multi-agent-nav-lib) source (on Linux or macOS), use the following command:

```bash
git clone git@github.com:haiot4105/multi-agent-nav-lib.git
cd multi-agent-nav-lib
pip install -e .
```

### Installation

After all requirements was installed, you should clone this repo in separate folder and run installation process (tested on Linux and macOS):

To set up this repository, follow these steps (tested on Linux and macOS):

1. Clone the repository:

```bash
git clone git@github.com:PathPlanning/TP-SWAP.git tp-swap
```

2. Install the package:

```bash
cd tp-swap
pip install -e .
```

### Launching Algorithms

To run and evaluate the algorithms, clone this reporitory into separate folder second time and switch to the `experiments` branch, which includes all necessary scripts and tools for conducting experiments.

```bash
cd ..
git clone git@github.com:PathPlanning/TP-SWAP.git tp-swap-exp
cd tp-swap-exp
git checkout experiments
```

Detailed instructions for running experiments, generating tasks, and processing results can be found in the README of the `experiments` branch.

## Implementing Your Own Algorithm

You can extend this repository by implementing a custom AMAPF algorithm. Here’s a brief guide:

1. Create a New Agent Class:
Implement your algorithm by creating a new agent class (e.g., `MyAlgAgent`). Inherit from the `Agent` base class (`dec_tswap/agent.py`) and override its methods. For examples, refer to `RandomAgent` and `SmartRandomAgent` in `dec_tswap/example_agent.py`.
2. Create a Parameter Class:
Define a corresponding parameter class (e.g., `MyAlgParams`) that inherits from `BaseAlgParams` in `manavlib`. This class should specify the necessary parameters for your algorithm. Refer to the `dec_tswap/example_agent.py` for examples.

Use the instructions from `experiments` branch  to evaluate your algorithm.

## Citing This Work

If you use this repository in your research, please cite the following paper:

```bibtex
@inproceedings{dergachev2024decentralized,
  title={Decentralized Unlabeled Multi-agent Pathfinding via Target and Priority Swapping},
  author={Dergachev, S. and Yakovlev, K.},
  booktitle={Proceedings of the 27th European Conference on Artificial Intelligence (ECAI 2024)},
  year={2024},
  pages={4344--4351},
  publisher={IOS Press}
}
```

## Contact

For questions or further information, please contact:

* Stepan Dergachev (*dergachev@isa.ru* or *sadergachev@hse.ru*)
