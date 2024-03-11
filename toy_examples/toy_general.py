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

# Use this when running as script
# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../hypergradient"))
# )

# Use this when running in interactive
# os.chdir("multilevel/toy_examples")
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname("__file__"), "../hypergradient"))
)
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import upper
import distributed_vi
import general_hg
from tqdm import tqdm
from matplotlib.patches import Rectangle
import general_agg_hg
import gurobipy as gp
from gurobipy import GRB

# %% Set problem parameters
dim = 2
n_ag = 3
n = dim * n_ag
lambd = 1.0
y_hat = np.concatenate((np.array([0.0, 0.0]), np.array([4.0, 0]), np.array([0, 2.0])))
dim_x = 4
x = np.array([0.2, 0.3, 1.0, 1.0])
lf = 1 / n_ag + lambd
mu = lambd
rec_corners = 2 * np.array([(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)])
rec_dims = 2 * np.array([(2.0, 2.0), (2.0, 2.0), (2.0, 2.0)])
A_ineq = np.block(
    [
        [np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
        [-np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
        [np.zeros((2, 2)), np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), -np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)],
        [np.zeros((2, 2)), np.zeros((2, 2)), -np.eye(2)],
    ]
)
b_ineq = np.array(
    [
        rec_corners[0][0] + rec_dims[0][0],
        rec_corners[0][1] + rec_dims[0][1],
        -rec_corners[0][0],
        -rec_corners[0][1],
        rec_corners[1][0] + rec_dims[1][0],
        rec_corners[1][1] + rec_dims[1][1],
        -rec_corners[1][0],
        -rec_corners[1][1],
        rec_corners[2][0] + rec_dims[2][0],
        rec_corners[2][1] + rec_dims[2][1],
        -rec_corners[2][0],
        -rec_corners[2][1],
    ]
)
n_ineq = b_ineq.size
# %% Define Parametric Lower-level Cost Matrix
# The objective of the followers is
# J_i = 1/2 \lambda \norm{y_i - \hat{y}_i}^2 + 1/2 \norm{1/N \sum{y_i} - x[0:2]}^2_E
# where \norm{}^2_E means the quadratic form weighted by E.
# Then, E(x) =  E_1 x[2] + E_2 x[3], where x[2] + x[3] = 1 and x[2], x[3] >= 0.
# The pseudo-gradient and its partial Jacobians are:s
# F_i(x, y) = \lambda (y_i - \hat{y}_i) + 1/N^2 E(x) \sum{y_i} - 1/N E(x) x[0:2]
# J_1 F_i(x, y) = [-1/N E(x)[:, 0], -1/N E(x)[:, 1],
#                  1/N^2 E_1 \sum{y_i} - 1/N E_1 x[0:2],
#                  1/N^2 E_2 \sum{y_i} - 1/N E_2 x[0:2] ]
# J_2 F_i(x, y) = 1/N^2 E(x) \ones_N^{\top} \kron \eye_2
E_1 = np.array([[1.0, 0.0], [0.0, 0.0]])
E_2 = np.array([[0.0, 0.0], [0.0, 1.0]])
E = lambda x: x[2] * E_1 + x[3] * E_2
# %% Define distributed solver parameters
dvix = x
dims_y = np.ones(n_ag, dtype=int) * dim
dvim = dim_x
dvigrad = []
dviJ1 = []
dviJ2 = []
A_eqdvi = []
b_eqdvi = []
H_eqdvi = []
A_ineqdvi = []
b_ineqdvi = []
G_ineqdvi = []
mytest = []
for ag in range(n_ag):
    start = dims_y[:ag].sum()
    end = dims_y[: ag + 1].sum()
    print(start)
    print(end)

    ppg = (
        lambda x, y, *, ag=ag: lambd
        * (y[ag] - y_hat[dims_y[:ag].sum() : dims_y[: ag + 1].sum()])
        + 1 / n_ag**2 * (E(x) @ np.sum(y, axis=0))
        - 1 / n_ag * (E(x) @ x[0:2])
    )
    dvigrad.append(ppg)

    # Jacobian x
    jacx = lambda x, y, *, ag=ag: np.vstack(
        [
            -1 / n_ag * E(x)[:, 0],
            -1 / n_ag * E(x)[:, 1],
            -1 / n_ag**2 * E_1 @ np.sum(y, axis=0) - 1 / n_ag * E_1 @ x[0:2],
            -1 / n_ag**2 * E_2 @ np.sum(y, axis=0) - 1 / n_ag * E_2 @ x[0:2],
        ]
    ).transpose()
    dviJ1.append(jacx)

    # Jacobian y
    jacy = lambda x, y, *, ag=ag: lambd * np.kron(
        np.eye(n_ag)[ag, :], np.eye(dim)
    ) + 1 / n_ag**2 * E(x) @ np.tile(np.eye(2), (1, n_ag))
    dviJ2.append(jacy)

    # Equality Constraints
    A_eqdvi.append(None)
    b_eqdvi.append(None)
    H_eqdvi.append(None)

    # Inequality Constraints
    A_ineqeach = np.block([[np.eye(2)], [-np.eye(2)]])
    A_ineqdvi.append(A_ineqeach)

    b_ineqeach = b_ineq[start * 2 : end * 2]
    b_ineqdvi.append(b_ineqeach)

    G_ineqdvi.append(None)

dvimu = mu
dvilf = lf
# %% Define distributed vi solver
vi = distributed_vi.dis_vi(
    x=dvix,
    dims_y=dims_y,
    m=dvim,
    grad=dvigrad,
    J1=dviJ1,
    J2=dviJ2,
    mu=dvimu,
    lf=dvilf,
    step=None,
    A_eq=A_eqdvi,
    b_eq=b_eqdvi,
    H_eq=H_eqdvi,
    A_ineq=A_ineqdvi,
    b_ineq=b_ineqdvi,
    G_ineq=G_ineqdvi,
)
# %% Define Upper-level Problem
x_rec_size = 4.0
x_rec_bot_left = np.array([-1.0, -1.0])
lrec_corner = x_rec_size * x_rec_bot_left
lrec_dims = 2 * np.array([x_rec_size, x_rec_size])
Ax_ineq = np.block(
    [
        [np.eye(2), np.zeros((2, 2))],
        [-np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), -np.eye(2)],
    ]
)
bx_ineq = np.concatenate(
    (
        np.array(
            [
                lrec_corner[0] + lrec_dims[0],
                lrec_corner[1] + lrec_dims[1],
                -lrec_corner[0],
                -lrec_corner[1],
            ]
        ),
        np.zeros((2,)),
        # - 0.5 * np.ones((2,)), # fix lower-lever cost to be uniform in x, y directions
    )
)
Ax_eq = np.block([[np.zeros((1, 2)), np.ones((1, 2))]])
bx_eq = np.array([2.0])
x_hat = np.array([1.0, 10.0])
mat_ones = 1 / n_ag * np.tile(np.eye(2), (1, n_ag))
# mat_choose = np.array([[1.0, 0.0], [0.0, 0.0]])
mat_choose = np.array([[0.0, 0.0], [0.0, 1.0]])
np_obj = (
    lambda x, y: np.linalg.norm(
        mat_choose @ (mat_ones @ y - np.expand_dims(x_hat, axis=1)), axis=0
    )
    ** 2
)
# %% Solver upper level with first-order method
J1 = lambda x, y: 0
# This definition of J2 works but it is a bit strange
# J2 = lambda x, y: (
#     mat_choose @ (mat_ones @ y - x_hat) @ (mat_choose @ mat_ones)
# ).transpose()
J2 = (
    lambda x, y: mat_ones.transpose()
    @ mat_choose.transpose()
    @ mat_choose
    @ (mat_ones @ y - x_hat)
)
gstep = lambda k: 5e0 / (k + 1) ** 0.51
rstep = lambda k: 1.0
upp_solv = upper.upper_opt(
    m=4,
    J1=J1,
    J2=J2,
    gstep=gstep,
    rstep=rstep,
    A_ineq=Ax_ineq,
    b_ineq=bx_ineq,
    A_eq=Ax_eq,
    b_eq=bx_eq,
)
upp_solv.x = np.array([0.0, 0.0, 0.5, 0.5])
# %% Run upper level
n_iters = 100
inner_iters = 1
vi.clear_log()
for k in tqdm(range(n_iters)):
    vi.x = upp_solv.x
    vi.run_proj_sens(n_iter=inner_iters, log_data=True)
    y_all = np.concatenate(vi.y, axis=0)
    s_all = np.vstack(vi.s)
    upp_solv.run_step(y=y_all, s=s_all)
# %% Do plotting
plt.figure(1)
ax = plt.gca()

plt.plot(
    upp_solv.x_log[0],
    upp_solv.x_log[1],
    marker="o",
    markersize=6,
    c="yellow",
    label="Leader",
)

plt.plot(
    upp_solv.x_log[0][-1],
    upp_solv.x_log[1][-1],
    marker="8",
    markersize=12,
    c="yellow",
    markeredgecolor="black",
)

plt.plot(
    np.average(vi.y_log[[0, 2, 4]], axis=0),
    np.average(vi.y_log[[1, 3, 5]], axis=0),
    c="cyan",
    label="Fol. Average",
    linestyle="--",
    linewidth=2,
)


plt.plot(
    np.average(vi.y_log[[0, 2, 4]], axis=0)[-1],
    np.average(vi.y_log[[1, 3, 5]], axis=0)[-1],
    c="cyan",
    marker="8",
    markersize=14,
    markeredgecolor="black",
)

plt.plot(
    x_hat[0], x_hat[1], marker="P", markersize=8, markeredgecolor="black", c="cyan"
)

lrect = Rectangle(
    lrec_corner,
    lrec_dims[0],
    lrec_dims[1],
    edgecolor="black",
    facecolor="yellow",
    alpha=0.5,
)
ax.add_patch(lrect)

colors = ["red", "green", "blue"]
for ag in range(n_ag):
    xs = vi.y_log[ag * dim, :]
    ys = vi.y_log[ag * dim + 1, :]
    # Plot line
    plt.plot(
        xs,
        ys,
        c=colors[ag],
        label="Follower " + str(ag + 1),
        marker="o",
        markersize=4,
        linewidth=2,
    )
    # Plot end
    plt.plot(
        xs[-1], ys[-1], marker="8", markersize=12, c=colors[ag], markeredgecolor="black"
    )
    # Plot "source"
    plt.plot(
        y_hat[ag * dim],
        y_hat[ag * dim + 1],
        marker="s",
        markersize=10,
        c=colors[ag],
        markeredgecolor="black",
    )

    rect = Rectangle(
        rec_corners[ag],
        rec_dims[ag][0],
        rec_dims[ag][1],
        edgecolor="black",
        facecolor=colors[ag],
        alpha=0.5,
    )
    ax.add_patch(rect)
plt.legend()
plt.show()
# %% Plot Objective Values
plt.figure(2)
upper_vals = np_obj(upp_solv.x_log, vi.y_log)
# opt_value = np.min((upper_vals_van[-1], prob.value))
plt.semilogy(
    upper_vals[inner_iters - 1 :: inner_iters]
    # label="Vanishing",
)
# plt.semilogy(upper_vals_con, label="Constant")
plt.grid()
# plt.legend()
plt.title("Leader's Convergence")
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.show()
plt.pause(0.1)

# %% Plot distance to VI solution
plt.figure(3)
y_diffs = np.linalg.norm(np.diff(vi.y_log, axis=1), axis=0)
plt.semilogy(
    y_diffs,
    label="Inner Loop",
    linewidth=1,
    color="blue",
    marker="o",
    markersize=4,
)
plt.semilogy(
    np.arange(0, n_iters * inner_iters)[inner_iters - 1 :: inner_iters],
    y_diffs[inner_iters - 1 :: inner_iters],
    label="Outer Loop",
    linewidth=4,
    color="red",
    marker="o",
    markersize=4,
)
# plt.semilogy(
#     1 / (np.arange(1, n_iters * inner_iters + 1)) ** 0.51,
#     label="Tolerance",
#     linewidth=2,
#     color="green",
# )
# # plt.loglog(y_diffs_con, label="Constant")
plt.grid()
plt.legend()
plt.title("Equilibrium Error")
plt.xlabel("Iterations")
plt.ylabel("2-Norm Error")
plt.show()
# %% Plot Distance of Sensitivity Derivatives
# plt.subplot(224)
plt.figure(4)
s_diffs = np.linalg.norm(np.diff(vi.s_log, axis=2), axis=(0, 1), ord=2)
plt.semilogy(
    s_diffs,
    label="Inner Loop",
    linewidth=1,
    color="blue",
    marker="o",
    markersize=4,
)
plt.semilogy(
    np.arange(0, n_iters * inner_iters)[inner_iters - 1 :: inner_iters],
    s_diffs[inner_iters - 1 :: inner_iters],
    label="Outer Loop",
    linewidth=4,
    color="red",
    marker="o",
    markersize=4,
)
plt.semilogy(
    1 / (np.arange(1, n_iters * inner_iters + 1)) ** 0.51,
    label="Tolerance",
    linewidth=2,
    color="green",
)
# # plt.loglog(s_diffs_con, label="Constant")
plt.grid()
plt.legend()
plt.title("Sensitivity Error")
plt.xlabel("Iterations")
plt.ylabel("Spectral Norm Error")
# Plot
plt.tight_layout()
plt.show()
# plt.savefig("toy_sim1.pdf", dpi=300
#    )
# %% Plot Parametric Cost Variables
plt.figure()
plt.plot(
    upp_solv.x_log[2, :],
    label="X Cost",
    linewidth=1,
    color="blue",
    marker="o",
    markersize=4,
)
plt.plot(
    upp_solv.x_log[3, :],
    label="Y Cost",
    linewidth=1,
    color="red",
    marker="o",
    markersize=4,
)
plt.grid()
plt.legend()
plt.title("Axis Cost Choice")
plt.xlabel("Iterations")
plt.ylabel("Axis Cost")
plt.tight_layout()
plt.show()

# %% Setup Distributed Solver Class
qp_inds = None
simp_proj = None
tol_y = lambda k: 1 / (k + 1) ** 0.51
tol_s = lambda k: 1 / (k + 1) ** 0.51

np_obj_dis = (
    lambda x, y: np.linalg.norm(mat_choose @ (mat_ones @ y - x_hat), axis=0) ** 2
)

genhg = general_hg.general_hypergradient(
    dims_y=dims_y,
    dim_x=dim_x,
    pgrad=dvigrad,
    pjacob1=dviJ1,
    pjacob2=dviJ2,
    A_ineq=A_ineqdvi,
    b_ineq=b_ineqdvi,
    G_ineq=G_ineqdvi,
    A_eq=A_eqdvi,
    b_eq=b_eqdvi,
    H_eq=H_eqdvi,
    mu=mu,
    lf=lf,
    upper_obj=np_obj_dis,
    Ax_ineq=Ax_ineq,
    bx_ineq=bx_ineq,
    Ax_eq=Ax_eq,
    bx_eq=bx_eq,
    qp_inds=qp_inds,
    simp_proj=simp_proj,
    phiJ1=J1,
    phiJ2=J2,
    gstep=gstep,
    rstep=rstep,
    tol_y=tol_y,
    tol_s=tol_s,
    up_offset=0,
    step_guess=None,
)
# %% Run Distributed Solver
genhg.run_fixed(inner_iters=1, n_iters=100, log_data=True)
# %% Plot Results
plt.figure(1)
genhg.plot_equilibrium_error()
plt.show()
plt.figure(2)
genhg.plot_sensitivity_error()
plt.show()
plt.figure(3)
genhg.plot_objective(every=10, inner=20)
plt.show()
plt.pause(0.1)
# %% Setup General Aggregative Hypergradient
# dvix = x
# dims_y = np.ones(n_ag, dtype=int) * dim
# dvim = dim_x
dvigrad_agg = []
dviJ1_agg = []
dviJ2_agg = []
dviJ3_agg = []
# A_eqdvi = []
# b_eqdvi = []
# H_eqdvi = []
# A_ineqdvi = []
# b_ineqdvi = []
# G_ineqdvi = []
for ag in range(n_ag):
    start = dims_y[:ag].sum()
    end = dims_y[: ag + 1].sum()
    print(start)
    print(end)
    # PPG
    ppg = (
        lambda x, yi, y_agg, *, ag=ag: lambd
        * (yi - y_hat[dims_y[:ag].sum() : dims_y[: ag + 1].sum()])
        + 1 / n_ag**2 * (E(x) @ y_agg)
        - 1 / n_ag * (E(x) @ x[0:2])
    )
    dvigrad_agg.append(ppg)

    # Jacobian x
    jacx = lambda x, yi, y_agg, *, ag=ag: np.vstack(
        [
            -1 / n_ag * E(x)[:, 0],
            -1 / n_ag * E(x)[:, 1],
            -1 / n_ag**2 * E_1 @ y_agg - 1 / n_ag * E_1 @ x[0:2],
            -1 / n_ag**2 * E_2 @ y_agg - 1 / n_ag * E_2 @ x[0:2],
        ]
    ).transpose()
    dviJ1_agg.append(jacx)

    # Jacobian y_i
    jacyi = lambda x, yi, y_agg, *, ag=ag: lambd * np.eye(dim)
    dviJ2_agg.append(jacyi)

    # Jacobian y_aggregate
    jacy_agg = lambda x, yi, y_agg, *, ag=ag: 1 / n_ag**2 * E(x)
    dviJ3_agg.append(jacy_agg)

    # Equality Constraints
    # A_eqdvi.append(None)
    # b_eqdvi.append(None)
    # H_eqdvi.append(None)

    # # Inequality Constraints
    # A_ineqeach = np.block([[np.eye(2)], [-np.eye(2)]])
    # A_ineqdvi.append(A_ineqeach)

    # b_ineqeach = b_ineq[start * 2 : end * 2]
    # b_ineqdvi.append(b_ineqeach)

    G_ineqdvi.append(None)

J1_agg = lambda x, y_agg: 0
J2_agg = lambda x, y_agg: mat_choose.transpose() @ mat_choose @ (y_agg - x_hat)
# dvimu = mu
# dvilf = lf
# %% Define general aggregative hypergradient
agg_lead = True
gen_agg_hg = general_agg_hg.general_agg_hypergradient(
    dims_y=dims_y,
    dim_x=dim_x,
    pgrad=dvigrad_agg,
    pjacob1=dviJ1_agg,
    pjacob2=dviJ2_agg,
    pjacob3=dviJ3_agg,
    A_ineq=A_ineqdvi,
    b_ineq=b_ineqdvi,
    G_ineq=G_ineqdvi,
    A_eq=A_eqdvi,
    b_eq=b_eqdvi,
    H_eq=H_eqdvi,
    mu=mu,
    lf=lf,
    upper_obj=np_obj_dis,
    Ax_ineq=Ax_ineq,
    bx_ineq=bx_ineq,
    Ax_eq=Ax_eq,
    bx_eq=bx_eq,
    qp_inds=qp_inds,
    simp_proj=simp_proj,
    phiJ1=J1_agg,
    phiJ2=J2_agg,
    gstep=gstep,
    rstep=rstep,
    tol_y=tol_y,
    tol_s=tol_s,
    up_offset=0,
    step_guess=None,
    agg_lead=agg_lead,
)
# %% Run Aggregative Solver
gen_agg_hg.run_fixed(inner_iters=1, n_iters=100, timing=True)
# %% Plot Results
plt.figure(1)
gen_agg_hg.plot_equilibrium_error()
plt.show()
plt.figure(2)
gen_agg_hg.plot_sensitivity_error()
plt.show()
plt.figure(3)
gen_agg_hg.plot_objective(every=1, inner=10)
plt.show()
plt.pause(0.1)
# %% Solve using GUROBI
gpm = gp.Model("general")
# Followers' Variable
yi = gpm.addMVar(shape=n_ag * dim, lb=-GRB.INFINITY, name="yi")
# Leader's Variable
xlead = gpm.addMVar(shape=dim_x, lb=-GRB.INFINITY, name="xlead")
# Dual Variables
# (Without the minus infinity lower bound constraint these are already assumed nonnegative)
lambda_i = gpm.addMVar(shape=n_ineq, name="lambda_i")
# Binary Variables/Interactive-1.interactive
bin_i = gpm.addMVar(shape=n_ineq, vtype=GRB.BINARY, name="bin_i")
bigM1 = 1e3
bigM2 = 1e3
# %% Constraints
# Stationarity Constraints
sum_yi = []
for ii in range(dim):
    sum_tmp = 0
    for ag in range(n_ag):
        sum_tmp += yi[ag * dim + ii]
    sum_yi.append(sum_tmp)
lead_pg = []
for ii in range(dim):
    lead_pg.append(xlead[ii] * xlead[ii + dim])

dual_pg = A_ineq.transpose() @ lambda_i
tot_pg = []
for ag in range(n_ag):
    for ii in range(dim):
        idx = ag * dim + ii
        tot_pg_tmp = (
            lambd * (yi[idx] - y_hat[idx])
            + 1 / (n_ag**2) * xlead[dim + ii] * sum_yi[ii]
            - 1 / n_ag * lead_pg[ii]
        )
        dual_pg_tmp = 0
        for jj in range(n_ineq):
            dual_pg_tmp += A_ineq.transpose()[idx, jj] * lambda_i[jj]

        tot_pg.append(tot_pg_tmp + dual_pg_tmp)

for ag in range(n_ag):
    for ii in range(dim):
        idx = ag * dim + ii
        gpm.addConstr(tot_pg[idx] == 0)

# %%
# Lower-level Local
gpm.addConstr(0 <= lambda_i)
gpm.addConstr(lambda_i <= bigM1 * bin_i)
gpm.addConstr(0 <= b_ineq - A_ineq @ yi)
gpm.addConstr(b_ineq - A_ineq @ yi <= bigM2 * (1 - bin_i))

# Upper level
gpm.addConstr(0 <= bx_ineq - Ax_ineq @ xlead)
gpm.addConstr(Ax_eq @ xlead == bx_eq)
# solx = np.array([0.0, 4.0, 1.0, 1.0])
# gpm.addConstr(xlead == solx)

# %% Define Objective
gpm.setObjective(
    yi @ (mat_ones.transpose() @ mat_choose.transpose() @ mat_choose @ mat_ones) @ yi
    - 2 * (x_hat @ mat_choose.transpose() @ mat_choose @ mat_ones) @ yi
    + x_hat.transpose() @ mat_choose.transpose() @ mat_choose @ x_hat
)
# gpm.setObjective(1.0)
# %% Set Parameters (same as in Matlab YALMIP)
gpm.update()
gpm.setParam(GRB.Param.MIPGap, 0.01)
gpm.setParam(GRB.Param.CrossoverBasis, 0)
gpm.setParam(GRB.Param.NodefileDir, "")
gpm.setParam(GRB.Param.NonConvex, 2)
gpm.setParam(GRB.Param.PreSOS2BigM, 0)
gpm.setParam(GRB.Param.TuneTrials, 3)
gpm.setParam(GRB.Param.NumericFocus, 3)
gpm.setParam(GRB.Param.TimeLimit, 100)
gpm.setParam(GRB.Param.Seed, 33)
# %%
gpm.optimize()
