# BIG Hype Algorithm (Best Intervention in Games using Hypergradients)
Big Hype is an algorithm for solving bilevel games/optimization problems with convergence guarantees.
## Setup
To setup this repo you need to:
- Create a Python [virtual environment](https://docs.python.org/3/library/venv.html).
- Activate the environment.
- Install the necessary Python modules by running:
    ```
    pip install -r /path/to/requirements.txt
    ```
## Contents
This repository contains the following:
- `hypergradient` folder includes the core solver functionality:
  - `polyhedral_proj.py` implements a class for to compute the projection onto generic polyhedra, and also computing the Jacobian of this projection. Additional classes are provided, such as projections onto boxes and simplices, but without the Jacobian computation functionality.
  - `*_vi.py` files implement solvers for the Variational Inequality (VI) problem associated with the computation of Nash Equilibria, using the projected pseudo-gradient method.
  - `upper.py` implements a simple projected gradient scheme with the possibility of adding relaxation steps, tailored to bilevel games.
  - `hypergrad.py` and `*_hg.py` implement the overall hypergradient scheme including both VI and upper-level solvers. A number of different implementations are included depending on the problem setting: Linear-Quadratic, Aggregative, General, and combinations of those. For more information see our paper.
- `toy_examples` folder contains simple problems that serve as tests and demonstrations for BIG Hype. It is a good place to start when trying to understand how to work with the hypergradient classes.
- `dr_python` folders contains a Demand Response (DR) problem setup. It is based on the paper "A Stackelberg game for incentive-based demand response in energy markets".

## Citing BIG Hype
If you use BIG Hype in your work please cite our [paper](https://arxiv.org/abs/2303.01101):
```
@misc{grontas2023,
      title={BIG Hype: Best Intervention in Games via Distributed Hypergradient Descent}, 
      author={Panagiotis D. Grontas and Giuseppe Belgioioso and Carlo Cenedese and Marta Fochesato and John Lygeros and Florian Dörfler},
      year={2023},
      eprint={2303.01101},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```