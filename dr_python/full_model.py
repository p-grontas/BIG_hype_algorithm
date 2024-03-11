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
import json
import times_plotting
from scipy.sparse import csc_array

# Use this when running as script
# curr_path = os.path.dirname(__file__)
# Use this when running in interactive
curr_path = os.path.dirname("__file__")

# Path to hypergradient folder
hg_path = os.path.abspath(os.path.join(curr_path, "../hypergradient"))
sys.path.insert(0, hg_path)
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import hypergrad
import polyhedral_proj
from full_model_plotting import plot_full_model
import distributed_hg

# %% Import Demand Data
# The file "net_d_i.npy" can be found in https://www.research-collection.ethz.ch/handle/20.500.11850/583560
data_path = os.path.abspath(os.path.join(curr_path, "net_d_i.npy"))
net_d_i = np.load(data_path)

# %% Enable Simulations
miqp_run = False
miqp_time = False
miqp_plot = False
dist_run = True
dist_plot = True
baseline = True
# %% Problem Setup
# Number of buildings/agents
n_ag = 45
building_reps = (np.floor((n_ag - 1) / 9) + 1).astype(int)
net_d_i = np.tile(net_d_i, (1, building_reps))
# Time horizon
tsteps = 24
# Flexibility request signal
coeff_r = 50.0
r = coeff_r * np.array(
    [0, 0, -5, -3, 3, 3, 3, 0, 0, 5, 0, -3, -3, 0, 0, 3, 3, -2, -2, -1, 0, 0, 0, 0]
)
r = r[:tsteps]
# Grid capacity
g_scale = 1.0 + 2.5
g_cap = g_scale * np.sum(net_d_i[:tsteps, :n_ag], axis=1)
# %% Parameters
eff_C = 1.0
eff_DC = 1.0
delta = 0.0
E_max_i = np.array([80, 80, 80, 80, 80, 30, 60, 60, 60])
E_max_i = np.tile(E_max_i, building_reps)
P_max_i = E_max_i.copy()
E_0_i = E_max_i.copy()
c1 = 1e-4
c0_min = 0.05
c0_max = 0.1
n_ineq_i = 11 * tsteps
n_eq_i = 2 * tsteps
n_ineq_coup = 3 * tsteps
p_reb = 1.5
p_res = 1.4
# %% Follower's Local Constraints
# Followers' decision vector xi := (p_i, y_i, e_i, p^C_i, p^DC_i, k_i)
# State of Charge: e_{i,tau} = e_{i,tau-1} + (eff_C * p_{i,tau}^C - eff_DC * p_{i,tau}^DC)
# Power balanace: p_{i,tau} - p^C_{i,tau} + p^DC_{i,tau} = d_{i,tau} - s_{i,tau}

# Equality Constraints
beq_i = np.zeros((2 * tsteps, n_ag))
Aeq_i = np.block(
    [
        [
            np.zeros((tsteps, 2 * tsteps)),
            np.eye(tsteps),
            -eff_C * np.tril(np.ones((tsteps, tsteps))),
            eff_DC * np.tril(np.ones((tsteps, tsteps))),
            np.zeros((tsteps, tsteps)),
        ],  # State of Charge
        [
            np.eye(tsteps),
            np.zeros((tsteps, 2 * tsteps)),
            -np.eye(tsteps),
            np.eye(tsteps),
            -np.eye(tsteps),
        ],  # Power Balance
    ]
)
for ag in range(n_ag):
    # State of Charge rhs
    bsoc = np.tile(np.array([E_0_i[ag]]), (tsteps,))
    # Power balance rhs
    bpb = net_d_i[:tsteps, ag]
    beq_i[:, ag] = np.concatenate((bsoc, bpb))

# Create Full Matrix of Equality Constraints
A_eq_all = np.kron(np.eye(n_ag), Aeq_i)
b_eq_all = beq_i.flatten("F")

# Inequality Constraints
# Upper and Lower Bounds on Variables
b_ineq_i = np.zeros((n_ineq_i, n_ag))
A_ineq_i = np.block(
    [
        [-np.eye(tsteps), np.zeros((tsteps, 5 * tsteps))],  # p_i >= 0
        [
            np.zeros((tsteps, tsteps)),
            -np.eye(tsteps),
            np.zeros((tsteps, 4 * tsteps)),
        ],  # y_i >= 0
        [
            np.zeros((tsteps, tsteps)),
            np.eye(tsteps),
            np.zeros((tsteps, 4 * tsteps)),
        ],  # y_i <= max(0, r)
        [
            np.zeros((tsteps, 2 * tsteps)),
            -np.eye(tsteps),
            np.zeros((tsteps, 3 * tsteps)),
        ],  # e_i >= 0
        [
            np.zeros((tsteps, 2 * tsteps)),
            np.eye(tsteps),
            np.zeros((tsteps, 3 * tsteps)),
        ],  # e_i <= e_i^max
        [
            np.zeros((tsteps, 3 * tsteps)),
            -np.eye(tsteps),
            np.zeros((tsteps, 2 * tsteps)),
        ],  # p_i^C >= 0
        [
            np.zeros((tsteps, 3 * tsteps)),
            np.eye(tsteps),
            np.zeros((tsteps, 2 * tsteps)),
        ],  # p_i^C <= p_i^max
        [
            np.zeros((tsteps, 4 * tsteps)),
            -np.eye(tsteps),
            np.zeros((tsteps, tsteps)),
        ],  # p_i^DC >= 0
        [
            np.zeros((tsteps, 4 * tsteps)),
            np.eye(tsteps),
            np.zeros((tsteps, tsteps)),
        ],  # p_i^DC <= p_i^max
        [np.zeros((tsteps, 5 * tsteps)), np.eye(tsteps)],  # k_i <= 0
        [np.zeros((tsteps, 5 * tsteps)), -np.eye(tsteps)],  # k_i >= min(r, 0)
    ]
)
for ag in range(n_ag):
    b_ineq_i[:, ag] = np.concatenate(
        (
            np.zeros(tsteps),  # p_i >= 0
            np.zeros(tsteps),  # y_i >= 0
            np.maximum(0, r),  # y_i <= max(0, r)
            np.zeros(tsteps),  # e_i >= 0
            np.ones(tsteps) * E_max_i[ag],  # e_i <= e_i^max
            np.zeros(tsteps),  # p_i^C >= 0
            np.ones(tsteps) * P_max_i[ag],  # p_i^C <= p_i^max
            np.zeros(tsteps),  # p_i^DC >= 0
            np.ones(tsteps) * P_max_i[ag],  # p_i^DC <= p_i^max
            np.zeros(tsteps),  # k_i <= 0
            -np.minimum(0, r),  # - k_i <= - min(r, 0)
        )
    )

# Create Full Matrix of Inequality Constraints
A_ineq_all = np.kron(np.eye(n_ag), A_ineq_i)
b_ineq_all = b_ineq_i.flatten("F")
# %% Followers' Coupling Constraints
# Inequality Constraints
# Grid Capacity: sum(p_i) - sum(k_i) <= g_i - min(0, r)
A_gc_each = np.block(
    [np.eye(tsteps), np.eye(tsteps), np.zeros((tsteps, 3 * tsteps)), -np.eye(tsteps)]
)
A_gc = np.tile(A_gc_each, (1, n_ag))
b_gc = g_cap[:tsteps] - np.minimum(0, r)

# Constraint on rebound flexibility provided
# - sum(k_i) <= - min(0, r)
A_reb_each = np.block([np.zeros((tsteps, 5 * tsteps)), -np.eye(tsteps)])
A_reb = np.tile(A_reb_each, (1, n_ag))
b_reb = -np.minimum(0, r)

# Constraint on response flexibility provided
# sum(y_i) <= max(0, r)
A_res_each = np.block(
    [np.zeros((tsteps, tsteps)), np.eye(tsteps), np.zeros((tsteps, 4 * tsteps))]
)
A_res = np.tile(A_res_each, (1, n_ag))
b_res = np.maximum(0, r)

# Aggregate matrices and vectors
A_ineq_coup = np.block([[A_gc], [A_reb], [A_res]])
b_ineq_coup = np.concatenate((b_gc, b_reb, b_res))
# %% Leader's Constraints
Ax_ineq = np.block(
    [
        [-np.eye(tsteps), np.zeros((tsteps, tsteps))],  # c0 >= c0_min
        [np.eye(tsteps), np.zeros((tsteps, tsteps))],  # c0 <= c0_max
        [
            np.ones(tsteps),
            np.zeros((1, tsteps)),
        ],  # average(c_0) <= 0.5 (c0_min + c0_max)
        [np.zeros((tsteps, tsteps)), -np.eye(tsteps)],  # alpha >= 0
        [np.zeros((tsteps, tsteps)), np.eye(tsteps)],  # alpha <= 1
    ]
)
alpha_max = np.ones(tsteps)
alpha_max[r <= 0] = 0.0
bx_ineq = np.concatenate(
    [
        -1 * np.ones(tsteps) * c0_min,  # c0 >= c0_min
        np.ones(tsteps) * c0_max,  # c0 <= c0_max
        tsteps * np.array([0.5 * (c0_min + c0_max)]),
        np.zeros(tsteps),
        alpha_max,
    ]
)
# %% Lower-level Stationarity Matrices
# For followers: Q @ x_i
reg = 5e-4
Q_each = c1 * np.kron(np.diag(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])), np.eye(tsteps))
C_ij = np.tile(Q_each, (n_ag, n_ag))
Q = np.kron(np.eye(n_ag), Q_each) + C_ij
Q += reg * np.eye(Q.shape[0])

# For leader: C @ v_1
C_i0 = np.block(
    [
        [np.eye(tsteps), np.zeros((tsteps, 5 * tsteps))],
        [
            np.zeros((tsteps, tsteps)),
            -p_res * np.eye(tsteps),
            np.zeros((tsteps, 4 * tsteps)),
        ],
    ]
).transpose()
C = np.tile(C_i0, (n_ag, 1))
# %% Define big M parameters
bigM_loc = 1e3
bigM_loc2 = 1e3
bigM_coup = 1e3
bigM_coup2 = 1e3
# %% Gurobi Interface Solver - Define Variables
m = gp.Model("dr")
# Followers' Variable, structured as xi = col((pi,pi^C,pi^DC,ei,k_i)_{i = 1,..., n_ag})
xi = m.addMVar(shape=n_ag * tsteps * 6, lb=-GRB.INFINITY, name="xi")
leaders = m.addMVar(shape=2 * tsteps, lb=-GRB.INFINITY, name="c_0_alpha")
# Dual Variables
lambda_i = m.addMVar(shape=n_ineq_i * n_ag, name="lambda_i")
nu_i = m.addMVar(shape=n_eq_i * n_ag, lb=-GRB.INFINITY, name="nu_i")
lambda_coup = m.addMVar(shape=n_ineq_coup, name="lambda_coup")
# Binary Variables
bin_i = m.addMVar(shape=n_ineq_i * n_ag, vtype=GRB.BINARY, name="bin_i")
bin_coup = m.addMVar(shape=n_ineq_coup, vtype=GRB.BINARY, name="bin_coup")
# %% Define Objective
# Ones matrix for prices
ones_p = np.tile(np.block([np.eye(tsteps), np.zeros((tsteps, 5 * tsteps))]), (1, n_ag))
# Ones matrix for rebound flexibility
ones_k = np.tile(np.block([np.zeros((1, 5 * tsteps)), np.ones(tsteps)]), (1, n_ag))
ones_y = np.tile(
    np.block(
        [np.zeros((tsteps, tsteps)), np.eye(tsteps), np.zeros((tsteps, 4 * tsteps))]
    ),
    (1, n_ag),
)
sum_tsteps = np.ones(tsteps)
# Matrices for leader's decisions
eye_c_0 = np.block([np.eye(tsteps), np.zeros((tsteps, tsteps))])
eye_alpha = np.block([np.zeros((tsteps, tsteps)), np.eye(tsteps)])
(
    m.setObjective(
        xi @ (-c1 * ones_p.transpose() @ ones_p) @ xi
        + leaders @ (-1 * eye_c_0.transpose() @ ones_p @ xi)
        + p_reb * ones_k @ xi
        + leaders @ (p_res * eye_alpha.transpose() @ ones_y @ xi)
        - p_res * sum_tsteps @ ones_y @ xi
    )
)
# %% Define Constraints
# Lower-level Stationarity
m.addConstr(
    Q @ xi
    + C @ leaders
    + A_eq_all.transpose() @ nu_i
    + A_ineq_all.transpose() @ lambda_i
    + A_ineq_coup.transpose() @ lambda_coup
    == 0
)

# Lower-level Local
m.addConstr(0 <= lambda_i)
m.addConstr(lambda_i <= bigM_loc * bin_i)
m.addConstr(0 <= b_ineq_all - A_ineq_all @ xi)
m.addConstr(b_ineq_all - A_ineq_all @ xi <= bigM_loc2 * (1 - bin_i))
m.addConstr(A_eq_all @ xi == b_eq_all)

# # Lower-level Coupling
m.addConstr(0 <= lambda_coup)
m.addConstr(lambda_coup <= bigM_coup * bin_coup)
m.addConstr(0 <= b_ineq_coup - A_ineq_coup @ xi)
m.addConstr(b_ineq_coup - A_ineq_coup @ xi <= bigM_coup2 * (1 - bin_coup))
# # Upper-level Constraints
m.addConstr(Ax_ineq @ leaders <= bx_ineq)
# m.addConstr(Ax_ineq @ v_1 <= bx_ineq)
m.update()
# %% Set Parameters (same as in Matlab YALMIP)
m.setParam(GRB.Param.MIPGap, 0.01)
m.setParam(GRB.Param.CrossoverBasis, 0)
m.setParam(GRB.Param.NodefileDir, "")
m.setParam(GRB.Param.NonConvex, 2)
m.setParam(GRB.Param.PreSOS2BigM, 0)
m.setParam(GRB.Param.TuneTrials, 3)
m.setParam(GRB.Param.NumericFocus, 3)
m.setParam(GRB.Param.TimeLimit, 3600 * 6)
m.setParam(GRB.Param.Seed, 33)
# %% Run
if miqp_run:
    m.optimize()

    if miqp_time:
        new_row = np.array([[n_ag, tsteps, m.runtime, m.MIPGap]])

        miqp_times_path = "path/to/miqp_2.json"
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


# %% Plot Demand Response Model (MIQP)
if miqp_plot:
    plot_full_model(
        x=leaders.X,
        y=xi.X,
        tsteps=tsteps,
        n_ag=n_ag,
        net_d_i=net_d_i,
        E_max_i=E_max_i,
        c0_min=c0_min,
        c0_max=c0_max,
        c1=c1,
        g_cap=g_cap,
        r=r,
    )
# %% Hypergradient
dim_y = 6 * tsteps * n_ag
dim_x = 2 * tsteps


def np_obj(x, y):
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
        x = np.expand_dims(x, axis=1)

    n_iters = y.shape[1]
    objs = np.zeros(n_iters)
    for ii in range(n_iters):
        objs[ii] = (
            y[:, ii].transpose() @ (-c1 * ones_p.transpose() @ ones_p) @ y[:, ii]
            - eye_c_0 @ x[:, ii].transpose() @ (ones_p @ y[:, ii])
            + p_reb * ones_k @ y[:, ii]
            - p_res
            * (
                (np.ones(tsteps) - eye_alpha @ x[:, ii]).transpose()
                @ (ones_y @ y[:, ii])
            )
        )
    return objs


# gstep = lambda k: 1e-6
rstep = lambda k: 1.0
tol_y_cent = lambda k: 1e-2 / (k + 1) ** 0.51
tol_s_cent = lambda k: 1e2 / (k + 1) ** 0.51
# %% Setup distributed model parameters
# Coupling constraints are handled by the leader or are fixed
parametric_lead = True
simplex_projectors = True
dims_y = np.ones(n_ag, dtype=int) * 6 * tsteps
Qs = []
Cs = []
cs = []
A_eqdvi = []
b_eqdvi = []
H_eqdvi = []
A_ineqdvi = []
b_ineqdvi = []
G_ineqdvi = []
if not (parametric_lead):
    dvim = dim_x
    for ag in range(n_ag):
        # Index of starting and ending row correspoding to each agent
        start = dims_y[:ag].sum()
        end = dims_y[: ag + 1].sum()

        # Equality Constraints
        A_eqdvi.append(Aeq_i)
        b_eqdvi.append(beq_i[:, ag])
        H_eqdvi.append(None)

        # Construct agent-specific matrices
        Qs.append(Q[start:end, :])
        Cs.append(C[start:end, :])
        cs.append(np.zeros(dims_y[ag]))

        # Inequality Constraints
        # Add local grid capacity constraints
        A_ineq_i_g = np.block(
            [
                [A_ineq_i],
                [
                    np.eye(tsteps),
                    np.eye(tsteps),
                    np.zeros((tsteps, 3 * tsteps)),
                    -np.eye(tsteps),
                ],
            ]
        )
        A_ineqdvi.append(A_ineq_i_g)
        # b_ineqdvi.append(np.concatenate((b_ineq_i[:, ag], g_scale * net_d_i[:tsteps, ag] - np.minimum(0, r))))
        # Each agent is allowed to access 1/n_ag of the total grid capacity
        b_ineqdvi.append(
            np.concatenate((b_ineq_i[:, ag], 1 / n_ag * (g_cap - np.minimum(0, r))))
        )
        G_ineqdvi.append(None)

    # Gradient of the leader's objective w.r.t. x and y
    J1dvi = lambda x, y: np.concatenate((-ones_p @ y, p_res * ones_y @ y))
    J2dvi = (
        lambda x, y: -c1 * 2 * (ones_p.transpose() @ ones_p) @ y
        - ones_p.transpose() @ eye_c_0 @ x
        + p_reb * np.squeeze(ones_k)
        - p_res * ones_y.transpose() @ (np.ones(tsteps) - eye_alpha @ x)
    )

    def J1dvi(xin, yin):
        tmp = np.concatenate((-ones_p @ yin, p_res * ones_y @ yin))
        return tmp

    y_coef_J2 = -c1 * 2 * (ones_p.transpose() @ ones_p)
    x_coef_J2 = -ones_p.transpose() @ eye_c_0 - p_res * ones_y.transpose() @ (
        -eye_alpha
    )
    const_J2 = p_reb * np.squeeze(ones_k) - p_res * ones_y.transpose() @ np.ones(tsteps)

    def J2dvi(xin, yin):
        tmp = y_coef_J2 @ yin + x_coef_J2 @ xin + const_J2
        return tmp

else:
    # One leader's variable (zeta) for the percentage of flexibility that
    # each agent may utilize
    # One leader's variable (theta) for the percentage of grid capacity that
    # each agent may utilize
    dvim = dim_x + 2 * n_ag
    for ag in range(n_ag):
        start = dims_y[:ag].sum()
        end = dims_y[: ag + 1].sum()

        # Equality Constraints
        A_eqdvi.append(Aeq_i)
        b_eqdvi.append(beq_i[:, ag])
        H_eqdvi.append(None)

        # Agent specific-matrices including additional coupling leader variables
        Qs.append(Q[start:end, :])
        Cs.append(np.block([C[start:end, :], np.zeros((dims_y[ag], 2 * n_ag))]))
        cs.append(np.zeros(dims_y[ag]))

        # Inequality Constraints
        # Add local grid capacity constraints
        A_ineq_i_g = np.block(
            [
                [A_ineq_i],
                [
                    np.eye(tsteps),
                    np.eye(tsteps),
                    np.zeros((tsteps, 3 * tsteps)),
                    -np.eye(tsteps),
                ],
            ]
        )
        A_ineqdvi.append(A_ineq_i_g)
        # b_ineqdvi.append(np.concatenate((b_ineq_i[:, ag], g_scale * net_d_i[:tsteps, ag] - np.minimum(0, r))))
        # We need a different rhs vector because for y_i, k_i
        # because the upper bounds are controlled by the leader
        b_ineqdvi_tmp = np.concatenate(
            (
                np.zeros(tsteps),  # p_i >= 0
                np.zeros(tsteps),  # y_i >= 0
                np.zeros(tsteps),  # y_i <= max(0, r) * zeta_i
                np.zeros(tsteps),  # e_i >= 0
                np.ones(tsteps) * E_max_i[ag],  # e_i <= e_i^max
                np.zeros(tsteps),  # p_i^C >= 0
                np.ones(tsteps) * P_max_i[ag],  # p_i^C <= p_i^max
                np.zeros(tsteps),  # p_i^DC >= 0
                np.ones(tsteps) * P_max_i[ag],  # p_i^DC <= p_i^max
                np.zeros(tsteps),  # k_i <= 0
                np.zeros(tsteps),  # - k_i <= - min(r, 0) * zeta_i
                np.zeros(tsteps),
            )
        )
        b_ineqdvi.append(b_ineqdvi_tmp)
        # Costruct matrix indicating which constraints are affected by the leader
        G_ineqdvi_tmp = np.block(
            [
                [np.zeros((2 * tsteps, dvim))],  # p_i, y_i >= 0
                [
                    np.zeros((tsteps, dim_x)),
                    np.kron(np.eye(n_ag)[ag], np.expand_dims(np.maximum(0, r), axis=1)),
                    np.zeros((tsteps, n_ag)),
                ],  # y_i <= max(0, r) * zeta_i
                [np.zeros((7 * tsteps, dvim))],  # e_i >= 0 ..  k_i <= 0
                [
                    np.zeros((tsteps, dim_x)),
                    np.kron(
                        np.eye(n_ag)[ag], np.expand_dims(-np.minimum(0, r), axis=1)
                    ),
                    np.zeros((tsteps, n_ag)),
                ],  # - k_i <= - min(r, 0) *
                [
                    np.zeros((tsteps, dim_x + n_ag)),
                    np.kron(
                        np.eye(n_ag)[ag],
                        np.expand_dims(g_cap - np.minimum(0, r), axis=1),
                    ),
                ],
            ]
        )
        G_ineqdvi.append(G_ineqdvi_tmp)

        # Define the objective function using the new leader's decision variable
        eye_c_0_dis = np.block(
            [np.eye(tsteps), np.zeros((tsteps, tsteps)), np.zeros((tsteps, 2 * n_ag))]
        )
        eye_alpha_dis = np.block(
            [np.zeros((tsteps, tsteps)), np.eye(tsteps), np.zeros((tsteps, 2 * n_ag))]
        )
        # J1dvi = lambda x, y: np.concatenate(
        #     (-ones_p @ y, p_res * ones_y @ y, np.zeros(2 * n_ag))
        # )
        # J2dvi = (
        #     lambda x, y: -c1 * 2 * (ones_p.transpose() @ ones_p) @ y
        #     - ones_p.transpose() @ eye_c_0_dis @ x
        #     + p_reb * np.squeeze(ones_k)
        #     - p_res * ones_y.transpose() @ (np.ones(tsteps) - eye_alpha_dis @ x)
        # )
        y_coef_J1 = np.block(
            [[-ones_p], [p_res * ones_y], [np.zeros((2 * n_ag, dim_y))]]
        )
        y_coef_J1 = csc_array(y_coef_J1)

        def J1dvi(xin, yin):
            # tmp = np.concatenate(
            #     (-ones_p @ yin, p_res * ones_y @ yin, np.zeros(2 * n_ag))
            # )
            tmp = y_coef_J1 @ yin
            return tmp

        y_coef_J2 = -c1 * 2 * (ones_p.transpose() @ ones_p)
        x_coef_J2 = -ones_p.transpose() @ eye_c_0_dis - p_res * ones_y.transpose() @ (
            -eye_alpha_dis
        )
        const_J2 = p_reb * np.squeeze(ones_k) - p_res * ones_y.transpose() @ np.ones(
            tsteps
        )

        y_coef_J2 = csc_array(y_coef_J2)
        x_coef_J2 = csc_array(x_coef_J2)

        def J2dvi(xin, yin):
            tmp = y_coef_J2 @ yin + x_coef_J2 @ xin + const_J2
            return tmp

        # Leader's constraints
        flex_param_low = -np.ones(n_ag) / n_ag * 0.8
        # Lower bound on the grid capacity percentage
        gc_param_low = -np.ones(n_ag) / n_ag * 0.8
        if not (simplex_projectors):
            Ax_ineq = np.block(
                [
                    [
                        -np.eye(tsteps),
                        np.zeros((tsteps, tsteps + 2 * n_ag)),
                    ],  # c0 >= c0_min
                    [
                        np.eye(tsteps),
                        np.zeros((tsteps, tsteps + 2 * n_ag)),
                    ],  # c0 <= c0_max
                    [
                        np.ones(tsteps),
                        np.zeros((1, tsteps + 2 * n_ag)),
                    ],  # average(c_0) <= 0.5 (c0_min + c0_max)
                    [
                        np.zeros((tsteps, tsteps)),
                        -np.eye(tsteps),
                        np.zeros((tsteps, 2 * n_ag)),
                    ],  # alpha >= 0
                    [
                        np.zeros((tsteps, tsteps)),
                        np.eye(tsteps),
                        np.zeros((tsteps, 2 * n_ag)),
                    ],  # alpha <= 1
                    [
                        np.zeros((n_ag, 2 * tsteps)),
                        -np.eye(n_ag),
                        np.zeros((n_ag, n_ag)),
                    ],  # zeta_i >= 0
                    [
                        np.zeros((1, 2 * tsteps)),
                        np.ones((1, n_ag)),
                        np.zeros((1, n_ag)),
                    ],  # \sum zeta_i <= 1
                    [
                        np.zeros((n_ag, 2 * tsteps + n_ag)),
                        -np.eye(n_ag),
                    ],  # theta_i >= 0
                    [
                        np.zeros((1, 2 * tsteps + n_ag)),
                        np.ones((1, n_ag)),
                    ],  # \sum theta_i <= 1
                ]
            )

            bx_ineq = np.concatenate(
                [
                    -1 * np.ones(tsteps) * c0_min,  # c0 >= c0_min
                    np.ones(tsteps) * c0_max,  # c0 <= c0_max
                    tsteps * np.array([0.5 * (c0_min + c0_max)]),
                    np.zeros(tsteps),  # alpha >= 0
                    alpha_max,  # alpha <= 1
                    flex_param_low,  # zeta_i >= 0
                    np.ones(1),  # \sum \zeta_i <= 1
                    gc_param_low,  # theta_i >= 0
                    np.ones(1),  # \sum theta_i <= 1
                ]
            )

            # Disable simplex projections
            qp_inds = None
            simp_proj = None
        else:
            Ax_ineq = np.block(
                [
                    [
                        -np.eye(tsteps),
                        np.zeros((tsteps, tsteps)),
                    ],  # c0 >= c0_min
                    [np.eye(tsteps), np.zeros((tsteps, tsteps))],  # c0 <= c0_max
                    [
                        np.ones(tsteps),
                        np.zeros((1, tsteps)),
                    ],  # average(c_0) <= 0.5 (c0_min + c0_max)
                    [
                        np.zeros((tsteps, tsteps)),
                        -np.eye(tsteps),
                    ],  # alpha >= 0
                    [
                        np.zeros((tsteps, tsteps)),
                        np.eye(tsteps),
                    ],  # alpha <= 1
                ]
            )

            bx_ineq = np.concatenate(
                [
                    -1 * np.ones(tsteps) * c0_min,  # c0 >= c0_min
                    np.ones(tsteps) * c0_max,  # c0 <= c0_max
                    tsteps * np.array([0.5 * (c0_min + c0_max)]),
                    np.zeros(tsteps),  # alpha >= 0
                    alpha_max,  # alpha <= 1
                ]
            )

            # Enable simplex projections
            qp_inds = np.concatenate(
                (
                    np.ones(2 * tsteps, dtype=bool),
                    np.zeros(2 * n_ag, dtype=bool),
                )
            )

            var_inds_flex = np.concatenate(
                (
                    np.zeros(2 * tsteps, dtype=bool),
                    np.ones(n_ag, dtype=bool),
                    np.zeros(n_ag, dtype=bool),
                )
            )
            var_inds_gc = np.concatenate(
                (
                    np.zeros(2 * tsteps, dtype=bool),
                    np.zeros(n_ag, dtype=bool),
                    np.ones(n_ag, dtype=bool),
                )
            )

            smp_flex = polyhedral_proj.simplex_proj(
                var_inds=var_inds_flex,
                lb=-flex_param_low[0],
                sum_to=1.0,
                method="active",
            )

            smp_gc = polyhedral_proj.simplex_proj(
                var_inds=var_inds_gc, lb=-gc_param_low[0], sum_to=1.0, method="active"
            )

            simp_proj = [smp_flex, smp_gc]


def np_obj_dis(x, y):
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
        x = np.expand_dims(x, axis=1)

    n_iters = y.shape[1]
    objs = np.zeros(n_iters)
    for ii in range(n_iters):
        objs[ii] = (
            y[:, ii].transpose() @ (-c1 * ones_p.transpose() @ ones_p) @ y[:, ii]
            - eye_c_0_dis @ x[:, ii].transpose() @ (ones_p @ y[:, ii])
            + p_reb * ones_k @ y[:, ii]
            - p_res
            * (
                (np.ones(tsteps) - eye_alpha_dis @ x[:, ii]).transpose()
                @ (ones_y @ y[:, ii])
            )
        )
    return objs


# %%
def gstep_dvi(k):
    if k < 100:
        return 3e-4 / ((k + 1) ** 0.51)
    else:
        return 5e-5 / ((k + 1) ** 0.51)


tol_y_dvi = lambda k: 1e-2 / (k + 1) ** 0.51
tol_s_dvi = lambda k: 1e2 / (k + 1) ** 0.51

dishg = distributed_hg.distributed_hypergradient(
    Q=Qs,
    C=Cs,
    c=cs,
    dims_y=dims_y,
    dim_x=dvim,
    A_ineq=A_ineqdvi,
    b_ineq=b_ineqdvi,
    G_ineq=G_ineqdvi,
    A_eq=A_eqdvi,
    b_eq=b_eqdvi,
    H_eq=H_eqdvi,
    upper_obj=np_obj_dis,
    Ax_ineq=Ax_ineq,
    bx_ineq=bx_ineq,
    Ax_eq=None,
    bx_eq=None,
    qp_inds=qp_inds,
    simp_proj=simp_proj,
    box_proj=None,
    phiJ1=J1dvi,
    phiJ2=J2dvi,
    gstep=gstep_dvi,
    rstep=rstep,
    tol_y=tol_y_dvi,
    tol_s=tol_s_dvi,
)
dishg.vi.step *= 2
dishg.hypergrad.x = np.concatenate(
    (
        np.ones(tsteps) * (c0_min + c0_max) / 2.0,
        np.ones(tsteps) * 0.5,
        1.0 * np.ones(n_ag) / n_ag,
        np.ones(n_ag) / n_ag,
    )
)
dishg.hypergrad.x_log[:, 0] = dishg.hypergrad.x.copy()
# dishg.exact_x = leaders.X
# dishg.exact_y = xi.X
# dishg.exact_val = m.ObjVal
# %%
saves_path = os.path.abspath(os.path.join(curr_path, "../../hgm_saves"))
# %%
if dist_run:
    reps = 3
    for rep in range(reps):
        dishg.run_fixed(inner_iters=1, n_iters=100, timing=True)
        # dishg.run_with_tolerance(n_iters=100, timing=True)
        plt.figure()
        dishg.plot_equilibrium_error()
        plt.show()
        plt.figure()
        dishg.plot_sensitivity_error()
        plt.show()
        plt.figure()
        dishg.plot_objective(every=3, inner=3)
        plt.show()
        plt.pause(0.1)
# %% Plotting
if dist_plot:
    # %% Parse Solved Problem Data
    plot_full_model(
        x=dishg.hypergrad.x[: -2 * n_ag],
        y=np.concatenate(dishg.vi.y, axis=0),
        tsteps=tsteps,
        n_ag=n_ag,
        net_d_i=net_d_i,
        E_max_i=E_max_i,
        c0_min=c0_min,
        c0_max=c0_max,
        c1=c1,
        g_cap=g_cap,
        r=r,
        flex_perc=dishg.hypergrad.x[dim_x : dim_x + n_ag],
    )

    # Plot timings
    plt.figure()
    dishg.plot_max_ag_times(pnd=True, sens=True, both=True)
    plt.show()
    plt.figure()
    dishg.plot_leader_times()
    plt.show()

    # %% Plot Parametric Variables
    # Maybe add these as a function to full_model_plotting
    plt.figure()
    plt.subplot(131)
    if n_ag > 9:
        first_n = 9
    else:
        first_n = n_ag
    for ag in range(first_n):
        par_con = dishg.hypergrad.x_log[dim_x + ag, :]
        plt.plot(1e2 * par_con, label="Agent " + str(ag + 1))
    plt.legend()
    plt.grid()
    plt.gca().set_title("Flexibility Percentages")
    plt.xlabel("Iterations")
    plt.ylabel("Percentage [%]")

    plt.subplot(132)
    for ag in range(first_n):
        par_con = dishg.hypergrad.x_log[dim_x + n_ag + ag, :]
        plt.plot(1e2 * par_con, label="Agent " + str(ag + 1))
    plt.gca().set_title("Grid Capacity Percentages")
    plt.xlabel("Iterations")
    plt.ylabel("Percentage [%]")
    plt.legend()
    plt.grid()

    plt.subplot(133)
    plt.plot(
        (np.arange(n_ag) + 1).astype(int),
        net_d_i.sum(axis=0)[:n_ag],
        drawstyle="steps",
        label="Total Demand",
        c="blue",
        linewidth=2,
    )
    plt.plot(
        (np.arange(n_ag) + 1).astype(int),
        E_max_i[:n_ag],
        drawstyle="steps",
        label="Battery Size",
        c="red",
        linewidth=2,
    )
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(n_ag))
    plt.gca().set_title("Agent Size")
    plt.xlabel("Agent Number")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.show()
