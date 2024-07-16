"""
   Copyright 2024 ETH Zurich, Panagiotis Grontas

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# %% Add hypergradient to path
import os
import sys
from os import listdir
from os.path import isfile, join
import re

# Use this when running as script
# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../hypergradient"))
# )

# Use this when running in interactive
# os.chdir("multilevel/dr_python")
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname("__file__"), "../hypergradient"))
)
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
import general_hg
import parametric_price_plotting
import general_agg_hg
import json
import times_plotting
import polyhedral_proj
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm

# %% Enable Running and Plotting
dist_save_tot = False
save_objective = False
save_inner_objective = False


# %%  Auxiliary Functions (for matrix building)
def choose_i_vec(i, all):
    vec_tmp = np.zeros(all)
    vec_tmp[i] = 1.0
    return vec_tmp


# %% Import Demand Data
# The file "net_d_i.npy" can be found in https://www.research-collection.ethz.ch/handle/20.500.11850/583560
net_d_i = np.load("net_d_i.npy")
# %% Problem Setup
# Number of buildings/agents
n_ag = 2
building_reps = (np.floor((n_ag - 1) / 9) + 1).astype(int)
net_d_i = np.tile(net_d_i, (1, building_reps))[:, :n_ag]
# Time horizon
tsteps = 24
# Grid capacity
g_scale = 1.0 + 2.0
g_cap = g_scale * np.sum(net_d_i[:tsteps, :n_ag], axis=1)
# %% Problem Parameters
eff_C = 1.0
eff_DC = 1.0
delta = 0.0
E_max_i = np.array([80, 80, 80, 80, 80, 30, 60, 60, 60])
E_max_i = np.tile(E_max_i, building_reps)
P_max_i = E_max_i.copy()
E_0_i = E_max_i.copy()
# Average marginal price
c1_min = 5e-4
c1_max = 1.5e-3
c0_min = 0.05e0
c0_max = 0.1e0
n_ineq_i = (7 + 1) * tsteps  # 7 local + 1 parametric coupling inequality constraints
n_eq_i = 2 * tsteps
# n_ineq_coup = tsteps
# %% Algorithm Parameters
# Total lower-level dimension
dim_y = 4 * tsteps * n_ag
# Each follower's dimension
dims_y = np.ones(n_ag, dtype=int) * 4 * tsteps
# Practical leader's dimension (c_0 and c_1)
dim_x = 2 * tsteps
# Augmented leader's dimension (including parametric coupling constraints)
dvim = dim_x + n_ag
# %% Followers' Local Constraints
# Followers' decision vector xi := (p_i, e_i, p^C_i, p^DC_i)
# State of Charge: e_{i,tau} = e_{i,tau-1} + (eff_C * p_{i,tau}^C - eff_DC * p_{i,tau}^DC)
# Power balanace: p_{i,tau} - p^C_{i,tau} + p^DC_{i,tau} = d_{i,tau} - s_{i,tau}

# Equality Constraints
beq_i = np.zeros((2 * tsteps, n_ag))
Aeq_i = np.block(
    [
        [
            np.zeros((tsteps, tsteps)),
            np.eye(tsteps),
            -eff_C * np.tril(np.ones((tsteps, tsteps))),
            eff_DC * np.tril(np.ones((tsteps, tsteps))),
        ],  # State of Charge
        [
            np.eye(tsteps),
            np.zeros((tsteps, tsteps)),
            -np.eye(tsteps),
            np.eye(tsteps),
        ],  # Power Balance
    ]
)
for ag in range(n_ag):
    # State of Charge rhs
    bsoc = np.tile(np.array([E_0_i[ag]]), (tsteps,))
    # Power balance rhs
    bpb = net_d_i[:tsteps, ag]
    beq_i[:, ag] = np.concatenate((bsoc, bpb))

# Inequality Constraints
# Upper and Lower Bounds on Variables
b_ineq_i = np.zeros((n_ineq_i, n_ag))
A_ineq_i = np.block(
    [
        [-np.eye(tsteps), np.zeros((tsteps, 3 * tsteps))],  # p_i >= 0
        [
            np.zeros((tsteps, tsteps)),
            -np.eye(tsteps),
            np.zeros((tsteps, 2 * tsteps)),
        ],  # e_i >= 0
        [
            np.zeros((tsteps, tsteps)),
            np.eye(tsteps),
            np.zeros((tsteps, 2 * tsteps)),
        ],  # e_i <= e_i^max
        [
            np.zeros((tsteps, 2 * tsteps)),
            -np.eye(tsteps),
            np.zeros((tsteps, tsteps)),
        ],  # p_i^C >= 0
        [
            np.zeros((tsteps, 2 * tsteps)),
            np.eye(tsteps),
            np.zeros((tsteps, tsteps)),
        ],  # p_i^C <= p_i^max
        [np.zeros((tsteps, 3 * tsteps)), -np.eye(tsteps)],  # p_i^DC >= 0
        [np.zeros((tsteps, 3 * tsteps)), np.eye(tsteps)],  # p_i^DC <= p_i^max
        [np.eye(tsteps), np.zeros((tsteps, 3 * tsteps))],  # sum(p_i) <= g
    ]
)
for ag in range(n_ag):
    b_ineq_i[:, ag] = np.concatenate(
        (
            np.zeros(tsteps),  # p_i >= 0
            np.zeros(tsteps),  # e_i >= 0
            np.ones(tsteps) * E_max_i[ag],  # e_i <= e_i^max
            np.zeros(tsteps),  # p_i^C >= 0
            np.ones(tsteps) * P_max_i[ag],  # p_i^C <= p_i^max
            np.zeros(tsteps),  # p_i^DC >= 0
            np.ones(tsteps) * P_max_i[ag],  # p_i^DC <= p_i^max
            np.zeros(tsteps),  # sum(p_i) <= g
        )
    )
G_ineq_i = np.block(
    [
        [np.zeros((7 * tsteps, dvim))],
        [
            np.zeros((tsteps, dim_x)),
            np.kron(
                np.eye(n_ag)[ag],
                np.expand_dims(g_cap, axis=1),
            ),
        ],
    ]
)
# %% Define leader's ingredients
# x := (c_0, c_1, coupling)
C_1 = lambda x: np.diag(x[tsteps : 2 * tsteps])

# Leader's Objective
ones_p = np.tile(np.block([np.eye(tsteps), np.zeros((tsteps, 3 * tsteps))]), (1, n_ag))


def np_obj(x, y):
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
        x = np.expand_dims(x, axis=1)

    n_iters = y.shape[1]
    objs = np.zeros(n_iters)
    for ii in range(n_iters):
        objs[ii] = -(C_1(x) @ ones_p @ y[:, ii] + x[:tsteps]) @ ones_p @ y[:, ii]
    return objs


def np_obj_tmp(x, y):
    # if len(y.shape) == 1:
    #     y = np.expand_dims(y, axis=1)
    #     x = np.expand_dims(x, axis=1)

    objs = -(C_1(x) @ ones_p @ y + x[:tsteps]) @ ones_p @ y
    return objs


# Leader's Constraints
simplex_projectors = True
gc_param_low = -np.ones(n_ag) / n_ag * 0.8
if not (simplex_projectors):
    Ax_ineq = np.block(
        [
            [-np.eye(tsteps), np.zeros((tsteps, tsteps + n_ag))],  # c0 >= c0_min
            [np.eye(tsteps), np.zeros((tsteps, tsteps + n_ag))],  # c0 <= c0_max
            [
                np.ones(tsteps),
                np.zeros((1, tsteps + n_ag)),
            ],  # average(c0) <= 0.5 (c0_min + c0_max)
            [
                np.zeros((tsteps, tsteps)),
                -np.eye(tsteps),
                np.zeros((tsteps, n_ag)),
            ],  # c1 >= c1_min
            [
                np.zeros((tsteps, tsteps)),
                np.eye(tsteps),
                np.zeros((tsteps, n_ag)),
            ],  # c1 <= c1_max
            [
                np.zeros((1, tsteps)),
                np.ones(tsteps),
                np.zeros((1, n_ag)),
            ],  # average(c1) <= 0.5 (c1_min + c1_max)
            # Here we have the coupling constraints. I will switch them to simplex projectors eventually
            [np.zeros((n_ag, 2 * tsteps)), -np.eye(n_ag)],  # theta_i >= 0
            [np.zeros((1, 2 * tsteps)), np.ones((1, n_ag))],  # sum(theta_i) <= 1
        ]
    )

    bx_ineq = np.concatenate(
        (
            -1 * np.ones(tsteps) * c0_min,  # c0 >= c0_min
            np.ones(tsteps) * c0_max,  # c0 <= c0_max
            tsteps
            * np.array(
                [0.5 * (c0_min + c0_max)]
            ),  # average(c0) <= 0.5 (c0_min + c0_max)
            -1 * np.ones(tsteps) * c1_min,  # c1 >= c1_min
            # -1 * np.ones(tsteps) * (c1_min + c1_max) * 0.5,  # c1 == constant
            np.ones(tsteps) * c1_max,  # c1 <= c1_max
            tsteps
            * np.array(
                [0.5 * (c1_min + c1_max)]
            ),  # average(c1) <= 0.5 (c1_min + c1_max)
            # np.zeros(n_ag), # theta_i >= 0
            gc_param_low,
            np.ones(1),  # sum(theta_i) <= 1
        )
    )

    # All constraints are solved as QPs.
    qp_inds = None
    simp_proj = None
else:
    # Simplex projection is used for the coupling constraints
    Ax_ineq = np.block(
        [
            [-np.eye(tsteps), np.zeros((tsteps, tsteps))],  # c0 >= c0_min
            [np.eye(tsteps), np.zeros((tsteps, tsteps))],  # c0 <= c0_max
            [
                np.ones(tsteps),
                np.zeros((1, tsteps)),
            ],  # average(c0) <= 0.5 (c0_min + c0_max)
            [
                np.zeros((tsteps, tsteps)),
                -np.eye(tsteps),
            ],  # c1 >= c1_min
            [
                np.zeros((tsteps, tsteps)),
                np.eye(tsteps),
            ],  # c1 <= c1_max
            [
                np.zeros((1, tsteps)),
                np.ones(tsteps),
            ],  # average(c1) <= 0.5 (c1_min + c1_max)
        ]
    )

    bx_ineq = np.concatenate(
        (
            -1 * np.ones(tsteps) * c0_min,  # c0 >= c0_min
            np.ones(tsteps) * c0_max,  # c0 <= c0_max
            tsteps
            * np.array(
                [0.5 * (c0_min + c0_max)]
            ),  # average(c0) <= 0.5 (c0_min + c0_max)
            -1 * np.ones(tsteps) * c1_min,  # c1 >= c1_min
            np.ones(tsteps) * c1_max,  # c1 <= c1_max
            tsteps
            * np.array(
                [0.5 * (c1_min + c1_max)]
            ),  # average(c1) <= 0.5 (c1_min + c1_max)
        )
    )

    # Enable simplex projections
    qp_inds = np.concatenate(
        (
            np.ones(2 * tsteps, dtype=bool),
            np.zeros(n_ag, dtype=bool),
        )
    )
    # Variable indices for grid capacity coupling constraints
    var_inds_gc = np.concatenate(
        (
            np.zeros(2 * tsteps, dtype=bool),
            np.ones(n_ag, dtype=bool),
        )
    )
    # Simplex projector for grid capacity
    smp_gc = polyhedral_proj.simplex_proj(
        var_inds=var_inds_gc, lb=-gc_param_low[0], sum_to=1.0, method="active"
    )

    simp_proj = [smp_gc]

ones_p_sparse = csc_array(ones_p)


# Leader's Gradient w.r.t. x
def phiJ1(xin, yin):
    sum_p = ones_p @ yin
    tmp = np.concatenate((-sum_p, -(sum_p**2), np.zeros(n_ag)))
    return tmp


# Leader's Gradient w.r.t. y
def phiJ2(xin, yin):
    sum_p = ones_p @ yin
    tmp = -2 * ones_p.transpose() @ C_1(xin) @ sum_p - ones_p.transpose() @ xin[:tsteps]
    return tmp


# %% Define followers' ingredients
# Quadratic regularization constant
reg = 1e-5
pgrad = []
pjacob1 = []
pjacob2 = []
A_eqdvi = []
b_eqdvi = []
H_eqdvi = []
A_ineqdvi = []
b_ineqdvi = []
G_ineqdvi = []
for ag in range(n_ag):
    # Pseudo-gradient plus regularization term
    pg = (
        lambda x, y, *, ag=ag: np.concatenate(
            (
                C_1(x) @ (y[ag] + np.sum(y, axis=0))[:tsteps] + x[:tsteps],
                np.zeros(3 * tsteps),
            )
        )
        + 2 * reg * np.eye(dims_y[ag]) @ y[ag]
        # + 2 * reg * np.diag(np.concatenate((np.zeros(tsteps), np.ones(2 * tsteps), np.zeros(tsteps)))) @ y[ag]
    )
    pgrad.append(pg)

    # Jacobian w.r.t. x
    jacx = lambda x, y, *, ag=ag: np.block(
        [
            [
                np.eye(tsteps),
                np.diag(y[ag][:tsteps] + np.sum(y, axis=0)[:tsteps]),
                np.zeros((tsteps, n_ag)),
            ],
            [np.zeros((3 * tsteps, dvim))],
        ]
    )
    pjacob1.append(jacx)

    # Jacobian w.r.t. y
    base_jacy = lambda x: np.block(
        [
            [
                C_1(x),
                np.zeros((tsteps, 3 * tsteps)),
            ],
        ]
    )
    jacy = lambda x, y, *, ag=ag: np.block(
        [
            [np.kron(np.ones((1, n_ag)) + choose_i_vec(ag, n_ag), base_jacy(x))],
            [np.zeros((3 * tsteps, n_ag * 4 * tsteps))],
            # [np.zeros((2 * tsteps, tsteps)), 2 * reg * np.eye(2 * tsteps), np.zeros((2 * tsteps, (4 * (n_ag - 1) + 1) * tsteps))],
            # [np.zeros((tsteps, n_ag * 4 * tsteps))]
        ]
    )
    pjacob2.append(jacy)

    # Equality Constraints
    A_eqdvi.append(Aeq_i)
    b_eqdvi.append(beq_i[:, ag])
    H_eqdvi.append(None)

    # Inequality Constraints
    A_ineqdvi.append(A_ineq_i)
    b_ineqdvi.append(b_ineq_i[:, ag])
    G_ineqdvi_tmp = np.block(
        [
            [np.zeros((7 * tsteps, dvim))],
            [
                np.zeros((tsteps, dim_x)),
                np.kron(
                    np.eye(n_ag)[ag],
                    np.expand_dims(g_cap, axis=1),
                ),
            ],
        ]
    )
    G_ineqdvi.append(G_ineqdvi_tmp)
# %% Define General Solver Class
# Guesses of constants that ensure convergence of the lower level
mu = 0.6e2
lf = 1.0
# Analytical Estimation
# mu = reg
# lf = np.square(n_ag ** 2 + n_ag) * c1_max
step_guess = 0.6e2
gstep_precond = np.concatenate(
    (
        1e1 * np.ones(tsteps),
        1e0 * np.ones(tsteps),
        1e1 * np.ones(n_ag),
    )
)
# Gradient step for the leader (alpha^k)
gstep_dvi = lambda k: 3e-6 / ((k + 1) ** 0.51) * gstep_precond
# Relaxation step for the leader (beta^k)
rstep = lambda k: 1.0
# Small Tolerance
# tol_y_dvi = lambda k: 2e-3 / (k + 1) ** 0.51
# tol_s_dvi = lambda k: 2.5e1 / (k + 1) ** 0.51
# # Medium Tolerance
# tol_y_dvi = lambda k: 2e-2 / (k + 1) ** 0.51
# tol_s_dvi = lambda k: 5e1 / (k + 1) ** 0.51
# # Large Tolerance
tol_y_dvi = lambda k: 1e-1 / (k + 1) ** 0.51
tol_s_dvi = lambda k: 5e2 / (k + 1) ** 0.51
# agg_lead = False

genhg = general_hg.general_hypergradient(
    dims_y=dims_y,
    dim_x=dvim,
    pgrad=pgrad,
    pjacob1=pjacob1,
    pjacob2=pjacob2,
    mu=mu,
    lf=lf,
    A_ineq=A_ineqdvi,
    b_ineq=b_ineqdvi,
    G_ineq=G_ineqdvi,
    A_eq=A_eqdvi,
    b_eq=b_eqdvi,
    H_eq=H_eqdvi,
    upper_obj=np_obj_tmp,
    Ax_ineq=Ax_ineq,
    bx_ineq=bx_ineq,
    Ax_eq=None,
    bx_eq=None,
    qp_inds=qp_inds,
    simp_proj=simp_proj,
    phiJ1=phiJ1,
    phiJ2=phiJ2,
    gstep=gstep_dvi,
    rstep=rstep,
    tol_y=tol_y_dvi,
    tol_s=tol_s_dvi,
    step_guess=step_guess,
)

x0_worst = np.concatenate(
    (
        np.ones(tsteps) * c0_min,
        np.ones(tsteps) * c1_min,
        1.0 * np.ones(n_ag) / n_ag,
    )
)
x0_rand_feas = np.concatenate(
    (
        np.random.rand(tsteps) * (c0_max - c0_min) + c0_min,
        np.random.rand(tsteps) * (c1_max - c1_min) + c1_min,
        1.0 * np.ones(n_ag) / n_ag,
    )
)

genhg.hypergrad.x = x0_worst.copy()

genhg.hypergrad.x_log[:, 0] = genhg.hypergrad.x.copy()
# %% Run Distributed Solver
# genhg.run_fixed(inner_iters=1, n_iters=2000, log_data=False)
# %% Run General Setup Iteration
genhg.run_general(n_iters=500, timing=False, log_data=False)
# %% Plot Results
plt.figure()
genhg.plot_equilibrium_error()
plt.show()
plt.figure()
genhg.plot_perturbed_sensitivity_error()
plt.show()
# %% Plot "Relative Suboptimality"
every = 1
inner = 10
objs = genhg.compute_objective(every=every, inner=inner)
best_obj = np.min(objs)
times = np.arange(genhg.up_off, objs.size * every + genhg.up_off, every)
plt.figure()
plt.plot(objs)
plt.grid()
plt.show()
# %%
plt.figure()
plt.plot(genhg.low2up, objs, linewidth=2, color="blue", marker="o", markersize=4)

plt.grid()
plt.title("Leader's Objective")
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
# %%
plt.figure()
eps = -1e-2
subopt = (objs - best_obj - eps) / np.abs(best_obj)
plt.semilogy(genhg.low2up, subopt, linewidth=2, color="blue", marker="o", markersize=4)
plt.grid()
plt.title("Relative Suboptimality")
plt.xlabel("Iterations")
plt.ylabel("Relative Suboptimality")
# %%
if save_objective:
    obj_path = "path/to/data.csv"
    objs = genhg.compute_objective(every=every, inner=inner)
    np.savetxt(obj_path, objs, delimiter=",")
# %%
if save_inner_objective:
    obj_path = "path/to/data_inner.csv"
    objs = genhg.compute_objective(every=every, inner=inner)
    np.savetxt(obj_path, np.vstack((genhg.low2up, objs)), delimiter=",")
# %% Plot the result
parametric_price_plotting.plot_parametric_price_problem(
    x=genhg.hypergrad.x,
    y=np.concatenate(genhg.vi.y, axis=0),
    tsteps=tsteps,
    n_ag=n_ag,
    net_d_i=net_d_i,
    E_max_i=E_max_i,
    c0_min=c0_min,
    c0_max=c0_max,
    c1_min=c1_min,
    c1_max=c1_max,
    g_cap=g_cap,
)
# %% Setup General Aggregative Hypergradient
# dvix = dvim
# dims_y = np.ones(n_ag, dtype=int) * dim
# dvim = dim_x
dvigrad_agg = []
pjacob1_agg = []
pjacob2_agg = []
pjacob3_agg = []
# A_eqdvi = []
# b_eqdvi = []
# H_eqdvi = []
# A_ineqdvi = []
# b_ineqdvi = []
# G_ineqdvi = []
for ag in range(n_ag):
    start = dims_y[:ag].sum()
    end = dims_y[: ag + 1].sum()
    # PPG
    ppg = (
        lambda x, yi, y_agg, *, ag=ag: np.concatenate(
            (
                C_1(x) @ (yi + y_agg)[:tsteps] + x[:tsteps],
                np.zeros(3 * tsteps),
            )
        )
        + 2 * reg * np.eye(dims_y[ag]) @ yi
    )
    dvigrad_agg.append(ppg)

    # Jacobian w.r.t. x
    jacx = lambda x, yi, y_agg, *, ag=ag: np.block(
        [
            [
                np.eye(tsteps),
                np.diag(yi[:tsteps] + y_agg[:tsteps]),
                np.zeros((tsteps, n_ag)),
            ],
            [np.zeros((3 * tsteps, dvim))],
        ]
    )
    pjacob1_agg.append(jacx)

    # Jacobian w.r.t. y_i
    base_jacy = lambda x: np.block(
        [
            [
                C_1(x),
                np.zeros((tsteps, 3 * tsteps)),
            ],
        ]
    )
    jacyi = lambda x, yi, y_agg, *, ag=ag: np.block(
        [
            [base_jacy(x)],
            [np.zeros((3 * tsteps, 4 * tsteps))],
        ]
    )
    pjacob2_agg.append(jacyi)

    # Jacobian w.r.t. aggregate y
    # The dependence of the PG on y_i and y_agg is identical
    pjacob3_agg.append(jacyi)


# Aggregative Leader's functions
# Leader's Gradient w.r.t. x
def phiJ1_agg(xin, y_agg):
    sum_p = y_agg[:tsteps]
    tmp = np.concatenate((-sum_p, -(sum_p**2), np.zeros(n_ag)))
    return tmp


# Leader's Gradient w.r.t. y_agg
def phiJ2_agg(xin, y_agg):
    sum_p = y_agg[:tsteps]
    tmp = np.concatenate((-2 * C_1(xin) @ sum_p - xin[:tsteps], np.zeros(3 * tsteps)))
    return tmp


# %% Define General Aggregative Solver Class
# Guesses of constants that ensure convergence of the lower level
mu = 0.6e2
lf = 1.0
step_guess = 0.6e2
gstep_precond = np.concatenate(
    (
        1e1 * np.ones(tsteps),
        1e0 * np.ones(tsteps),
        1e1 * np.ones(n_ag),
    )
)
gstep_dvi = lambda k: 3e-6 / ((k + 1) ** 0.51) * gstep_precond
# gstep_dvi = lambda k: 5e-6 / ((k + 1) ** 1.01)
# def gstep_dvi(k):
#     if k < 500:
#         return 1e-6 / ((k + 1) ** 0.51) * gstep_precond
#     else:
#         return 5e-7 / ((k + 1) ** 0.51) * gstep_precond
rstep = lambda k: 1.0
tol_y_dvi = lambda k: 2e-2 / (k + 1) ** 0.51
tol_s_dvi = lambda k: 5e1 / (k + 1) ** 0.51
agg_lead = True

gen_agg_hg = general_agg_hg.general_agg_hypergradient(
    dims_y=dims_y,
    dim_x=dvim,
    pgrad=dvigrad_agg,
    pjacob1=pjacob1_agg,
    pjacob2=pjacob2_agg,
    pjacob3=pjacob3_agg,
    mu=mu,
    lf=lf,
    A_ineq=A_ineqdvi,
    b_ineq=b_ineqdvi,
    G_ineq=G_ineqdvi,
    A_eq=A_eqdvi,
    b_eq=b_eqdvi,
    H_eq=H_eqdvi,
    upper_obj=np_obj_tmp,
    Ax_ineq=Ax_ineq,
    bx_ineq=bx_ineq,
    Ax_eq=None,
    bx_eq=None,
    qp_inds=qp_inds,
    simp_proj=simp_proj,
    phiJ1=phiJ1_agg,
    phiJ2=phiJ2_agg,
    gstep=gstep_dvi,
    rstep=rstep,
    tol_y=tol_y_dvi,
    tol_s=tol_s_dvi,
    step_guess=step_guess,
    agg_lead=agg_lead,
)

x0_worst = np.concatenate(
    (
        np.ones(tsteps) * c0_min,
        np.ones(tsteps) * c1_min,
        1.0 * np.ones(n_ag) / n_ag,
    )
)
x0_rand_feas = np.concatenate(
    (
        np.random.rand(tsteps) * (c0_max - c0_min) + c0_min,
        np.random.rand(tsteps) * (c1_max - c1_min) + c1_min,
        1.0 * np.ones(n_ag) / n_ag,
    )
)

gen_agg_hg.hypergrad.x = x0_worst.copy()

gen_agg_hg.hypergrad.x_log[:, 0] = gen_agg_hg.hypergrad.x.copy()
# %%
# saves_path_dishg = "path/to/distributed_" + str(n_ag) + "/"
# %%
reps = 1
for rep in range(reps):
    gen_agg_hg.run_fixed(inner_iters=1, n_iters=1000, timing=True)
    # dishg.run_with_tolerance(n_iters=100, timing=True)
    # plt.figure()
    # dishg.plot_relative_suboptimality()
    # plt.show()
    plt.figure(1)
    gen_agg_hg.plot_equilibrium_error()
    plt.show()
    plt.figure(2)
    gen_agg_hg.plot_sensitivity_error()
    plt.show()
    plt.figure(3)
    gen_agg_hg.plot_objective(every=10, inner=3)
    plt.show()
    plt.pause(0.1)
    every = 10
    inner = 10
    objs = gen_agg_hg.compute_objective(every=every, inner=inner)
    relative_change = np.abs(np.diff(objs, n=1) / objs[:-1])
    times = np.arange(gen_agg_hg.up_off, objs.size * every + gen_agg_hg.up_off, every)
    plt.figure()
    plt.semilogy(times[1:], relative_change)
    # filename = "data" + str(rep) + ".json"
    # save_file = os.path.abspath(os.path.join(curr_path, "../../hgm_saves/" + filename))
    # save_file = saves_path_dishg + filename
    # gen_agg_hg.save_to_json(save_file)
    # gen_agg_hg.clear_log()
    # %% Get files
    onlyfiles = [
        f
        for f in listdir(saves_path_dishg)
        if isfile(join(saves_path_dishg, f))
        if f.endswith(".json")
    ]
    onlyfiles.sort(key=lambda f: int(re.sub("\D", "", f)))
    filepaths = []
    for f in onlyfiles:
        filepaths.append(os.path.abspath(os.path.join(saves_path_dishg, f)))
    gen_agg_hg.load_json(filepaths)
    # %%
    if dist_save_tot:
        path = "path/to/comparison/big_hype.json"
        best_obj = np.min(gen_agg_hg.compute_objective(every=3, inner=20))
        gen_agg_hg.save_total_time(path, n_ag, tsteps, best_obj, distributed=False)
# %% Plot timing over different numbers of agents
data_path = "path/to/data"
tplot = times_plotting.times_plots(folder_path=data_path, agg=True)
plt.figure()
tplot.plot_leader_times(followers=True)
plt.show()

plt.figure()
tplot.plot_total_times()
plt.show()
# %% Solve using GUROBI
gpm = gp.Model("general")
# Followers' Variable
yi = gpm.addMVar(shape=dim_y, lb=-GRB.INFINITY, name="yi")
# Leader's Variable
xlead = gpm.addMVar(shape=dim_x, lb=-GRB.INFINITY, name="xlead")
# Local Dual Inequality Variables
n_ineq_all = n_ag * (7 * tsteps)
lambda_i = gpm.addMVar(shape=n_ineq_all, lb=-GRB.INFINITY, name="lambda_i")
# Local Dual Equality Variables
n_eq_all = n_ag * (2 * tsteps)
nu_i = gpm.addMVar(shape=n_eq_all, lb=-GRB.INFINITY, name="nu_i")
# Coupling Dual Inequality Variables
n_ineq_coup = tsteps
lambda_coup = gpm.addMVar(shape=n_ineq_coup, lb=-GRB.INFINITY, name="lambda_coup")
# Binary variables
bin_i = gpm.addMVar(shape=n_ineq_all, vtype=GRB.BINARY, name="bin_i")
bin_coup = gpm.addMVar(shape=n_ineq_coup, vtype=GRB.BINARY, name="bin_coup")
# Do optimization or find equilibrium for fixed x
find_eq = False
# % Define Necessary Matrices
# These could be made sparse
A_ineq_all = np.kron(np.eye(n_ag), A_ineq_i[:-tsteps, :])
b_ineq_all = b_ineq_i[:-tsteps, :].flatten("F")
A_eq_all = np.kron(np.eye(n_ag), Aeq_i)
b_eq_all = beq_i.flatten("F")
# Followers' Coupling Constraints
A_coup_i = np.block([np.eye(tsteps), np.zeros((tsteps, 3 * tsteps))])
A_ineq_coup = np.tile(A_coup_i, (1, n_ag))
b_ineq_coup = g_cap
# Leader's Constraints
Ax_ineq = np.block(
    [
        [-np.eye(tsteps), np.zeros((tsteps, tsteps))],  # c0 >= c0_min
        [np.eye(tsteps), np.zeros((tsteps, tsteps))],  # c0 <= c0_max
        [
            np.ones(tsteps),
            np.zeros((1, tsteps)),
        ],  # average(c0) <= 0.5 (c0_min + c0_max)
        [
            np.zeros((tsteps, tsteps)),
            -np.eye(tsteps),
        ],  # c1 >= c1_min
        [
            np.zeros((tsteps, tsteps)),
            np.eye(tsteps),
        ],  # c1 <= c1_max
        [
            np.zeros((1, tsteps)),
            np.ones(tsteps),
        ],  # average(c1) <= 0.5 (c1_min + c1_max)
    ]
)
bx_ineq = np.concatenate(
    (
        -1 * np.ones(tsteps) * c0_min,  # c0 >= c0_min
        np.ones(tsteps) * c0_max,  # c0 <= c0_max
        tsteps
        * np.array([0.5 * (c0_min + c0_max)]),  # average(c0) <= 0.5 (c0_min + c0_max)
        -1 * np.ones(tsteps) * c1_min,  # c1 >= c1_min
        np.ones(tsteps) * c1_max,  # c1 <= c1_max
        tsteps
        * np.array([0.5 * (c1_min + c1_max)]),  # average(c1) <= 0.5 (c1_min + c1_max)
    )
)

# Big-M variables
# (n_ag, tsteps) = (2, 24) -> 5e3
# (n_ag, tsteps) = (1, 12) -> 3e2
# (n_ag, tsteps) = (1, 8) -> 3e2
bigM_loc1 = 4e2
bigM_loc2 = 4e2
bigM_coup1 = 4e2
bigM_coup2 = 4e2
# % Define Constraints
# Lower-level local
gpm.addConstr(0 <= lambda_i)
gpm.addConstr(lambda_i <= bigM_loc1 * bin_i)
gpm.addConstr(0 <= b_ineq_all - A_ineq_all @ yi)
gpm.addConstr(b_ineq_all - A_ineq_all @ yi <= bigM_loc2 * (1 - bin_i))
gpm.addConstr(A_eq_all @ yi == b_eq_all)

# Lower-level Coupling
gpm.addConstr(0 <= lambda_coup)
gpm.addConstr(lambda_coup <= bigM_coup1 * bin_coup)
gpm.addConstr(0 <= b_ineq_coup - A_ineq_coup @ yi)
gpm.addConstr(b_ineq_coup - A_ineq_coup @ yi <= bigM_coup2 * (1 - bin_coup))
# Upper-level Constraints
if find_eq:
    # gpm.addConstr(xlead == x0_worst[:2 * tsteps])
    gpm.addConstr(xlead == x0_rand_feas[: 2 * tsteps])
    # gpm.addConstr(xlead == genhg.hypergrad.x[:2 * tsteps])
else:
    gpm.addConstr(Ax_ineq @ xlead <= bx_ineq)
# % Stationarity Constraints
# Pseudo-gradient
sum_yi = []
for ii in range(4 * tsteps):
    sum_tmp = 0
    for ag in range(n_ag):
        sum_tmp += yi[ag * (4 * tsteps) + ii]
    sum_yi.append(sum_tmp)

pseudog = []
dual_pg_ineq = []
dual_pg_eq = []
dual_pg_coup = []
stat_con = []
for ag in tqdm(range(n_ag)):
    for ii in range(4 * tsteps):
        # Pseudogradient Computation
        pg_tmp = 0
        idx = ag * (4 * tsteps) + ii
        if ii < tsteps:
            pg_tmp += xlead[tsteps + ii] * (yi[idx] + sum_yi[ii]) + xlead[ii]
        pg_tmp += 2 * reg * yi[idx]
        pseudog.append(pg_tmp)

        # Dual local inequality part computation
        dual_pg_ineq_tmp = 0
        for jj in range(n_ineq_all):
            dual_pg_ineq_tmp += A_ineq_all.transpose()[idx, jj] * lambda_i[jj]
        dual_pg_ineq.append(dual_pg_ineq_tmp)

        # Dual local equality part computation
        dual_pg_eq_tmp = 0
        for jj in range(n_eq_all):
            dual_pg_eq_tmp += A_eq_all.transpose()[idx, jj] * nu_i[jj]
        dual_pg_eq.append(dual_pg_eq_tmp)

        # Dual coupling inequality part computation
        dual_pg_coup_tmp = 0
        for jj in range(n_ineq_coup):
            dual_pg_coup_tmp += A_ineq_coup.transpose()[idx, jj] * lambda_coup[jj]
        dual_pg_coup.append(dual_pg_coup_tmp)

        stat_con.append(pg_tmp + dual_pg_ineq_tmp + dual_pg_eq_tmp + dual_pg_coup_tmp)
        gpm.addConstr(stat_con[-1] == 0)
# % Specify Objective
if find_eq:
    gpm.setObjective(1.0)
else:
    gp_objective = 0
    w_aux = gpm.addMVar(shape=tsteps, lb=-GRB.INFINITY, name="w_aux")
    for ii in range(tsteps):
        gpm.addConstr(w_aux[ii] == xlead[tsteps + ii] * sum_yi[ii])
        gp_objective -= (w_aux[ii] + xlead[ii]) * sum_yi[ii]
    gpm.setObjective(gp_objective)
# %% Set Parameters
gpm.update()
gpm.setParam(GRB.Param.FeasibilityTol, 1e-6)
gpm.setParam(GRB.Param.MIPGap, 0.05)
gpm.setParam(GRB.Param.CrossoverBasis, 0)
gpm.setParam(GRB.Param.NodefileDir, "")
gpm.setParam(GRB.Param.NonConvex, 2)
gpm.setParam(GRB.Param.PreSOS2BigM, 0)
gpm.setParam(GRB.Param.TuneTrials, 3)
gpm.setParam(GRB.Param.NumericFocus, 3)
gpm.setParam(GRB.Param.TimeLimit, 4 * 3600)
gpm.setParam(GRB.Param.Seed, 42)
# %%
gpm.optimize()
# %% Evaluate equilibrium based on leader's decision,
# when gurobi times out and the lower level variables are not a Nash equilibrium.
genhg.hypergrad.x = np.concatenate((xlead.X, np.ones(n_ag) / n_ag))
genhg.run_fixed(inner_iters=1, n_iters=1)
genhg.hypergrad.x = np.concatenate((xlead.X, np.ones(n_ag) / n_ag))
genhg.vi.run_projection(n_iter=1000)
# genhg.plot_equilibrium_error()
y_all = np.concatenate(genhg.vi.y, axis=0)
gurobi_obj = np_obj_tmp(xlead.X, y_all)
print(np_obj_tmp(xlead.X, yi.X))
print(gurobi_obj)
# %%
gurobi_save = False
if gurobi_save:
    new_row = np.array([[n_ag, tsteps, gpm.runtime, gpm.MIPGap, gurobi_obj]])

    miqp_times_path = "path/to/comparison/miqcqp.json"
    if os.path.isfile(miqp_times_path):
        mode = "r+"
    else:
        mode = "a+"
    with open(miqp_times_path, mode, encoding="utf-8") as file:
        if not (os.path.getsize(miqp_times_path) == 0):
            data_dict = json.loads(file.read())
            updated_array = np.append(np.array(data_dict["data"]), new_row, axis=0)
        else:
            updated_array = new_row

        data_dict = {"data": updated_array.tolist()}

    with open(miqp_times_path, "w", encoding="utf-8") as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4)
# %%
parametric_price_plotting.plot_parametric_price_problem(
    x=xlead.X,
    y=y_all,
    tsteps=tsteps,
    n_ag=n_ag,
    net_d_i=net_d_i,
    E_max_i=E_max_i,
    c0_min=c0_min,
    c0_max=c0_max,
    c1_min=c1_min,
    c1_max=c1_max,
    g_cap=g_cap,
)
# %%
genhg.hypergrad.x = np.concatenate((xlead.X, np.ones(n_ag) / n_ag))
genhg.run_fixed(inner_iters=1, n_iters=1)
genhg.hypergrad.x = np.concatenate((xlead.X, np.ones(n_ag) / n_ag))
genhg.vi.run_projection(n_iter=1000)
# genhg.plot_equilibrium_error()
y_all = np.concatenate(genhg.vi.y, axis=0)
np_obj_tmp(xlead.X, y_all)
# %%
parametric_price_plotting.plot_parametric_price_problem(
    x=genhg.hypergrad.x[: 2 * tsteps],
    y=np.concatenate(genhg.vi.y, axis=0),
    tsteps=tsteps,
    n_ag=n_ag,
    net_d_i=net_d_i,
    E_max_i=E_max_i,
    c0_min=c0_min,
    c0_max=c0_max,
    c1_min=c1_min,
    c1_max=c1_max,
    g_cap=g_cap,
)
