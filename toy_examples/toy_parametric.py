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
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname("__file__"), "../hypergradient"))
)
# %% Imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import low_vi, upper, hypergrad
import cvxpy as cp
from matplotlib.patches import Rectangle
# %% Set problem parameters
dim = 2
n_ag = 3
n = dim * n_ag
lambd = 1.0
y_hat = np.concatenate((np.array([0.0, 0.0]), np.array([4.0, 0]), np.array([0, 2.0])))
dim_x = 3
x = np.array([0.2, 0.3, 0.1])
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
G_ineq = np.array(
    [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
)
G_ineq = np.hstack((np.zeros((n_ineq, 1)), np.zeros((n_ineq, 1)), G_ineq))
# %% Define gradient
C = -1 / n_ag * np.tile(np.hstack((np.eye(2), np.zeros((dim, 1)))), (n_ag, 1))
Q = 1 / n_ag**2 * np.kron(np.ones((n_ag, n_ag)), np.eye(2)) + lambd * np.eye(n)
grad = lambda x, y: np.matmul(Q, y) + np.matmul(C, x) - lambd * y_hat
# %% Get Jacobians
jacob1 = lambda x, y: C
jacob2 = lambda x, y: Q
# %%
vi = low_vi.lower_vi(
    x=x,
    n=n,
    m=dim_x,
    grad=grad,
    J1=jacob1,
    J2=jacob2,
    mu=mu,
    lf=lf,
    A_ineq=A_ineq,
    b_ineq=b_ineq,
    G_ineq=G_ineq,
    A_eq=None,
)
# %% Define upper level
x_hat = np.array([10.0, 0.0])
mat_ones = 1 / n_ag * np.tile(np.eye(2), (1, n_ag))
# mat_ones = 0.5 * np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
np_obj = (
    lambda x, y: np.linalg.norm(mat_ones @ y - np.expand_dims(x_hat, axis=1), axis=0)
    ** 2
)
x_rec_size = 30.0
x_rec_bot_left = np.array([-1.0, -1.0])
lrec_corner = x_rec_size * x_rec_bot_left
lrec_dims = 2 * np.array([x_rec_size, x_rec_size])
Ax_ineq = np.block(
    [
        [np.eye(2), np.zeros((2, 1))],
        [-np.eye(2), np.zeros((2, 1))],
        [np.zeros((2, 2)), np.array([[1.0], [-1.0]])],
    ]
)
bx_ineq = np.array(
    [
        lrec_corner[0] + lrec_dims[0],
        lrec_corner[1] + lrec_dims[1],
        -lrec_corner[0],
        -lrec_corner[1],
        10.0,
        0.0,
    ]
)
# %% Exact solution using MIQP
x_top = cp.Variable(dim_x)
y_low = cp.Variable(n)
lambdas = cp.Variable(n_ineq)
binaries = cp.Variable(n_ineq, boolean=True)
bigM = 4e2
cvx_objective = cp.Minimize(cp.sum_squares(mat_ones @ y_low - x_hat))
constraints = (
    [Ax_ineq @ x_top <= bx_ineq]
    + [Q @ y_low + C @ x_top - lambd * y_hat + A_ineq.transpose() @ lambdas == 0]
    + [0 <= b_ineq + G_ineq @ x_top - A_ineq @ y_low]
    + [b_ineq + G_ineq @ x_top - A_ineq @ y_low <= bigM * (1 - binaries)]
    + [0 <= lambdas]
    + [lambdas <= bigM * binaries]
)
prob = cp.Problem(cvx_objective, constraints)
# mparams = {mosek.dparam.mio_tol_feas: 1e-9, mosek.iparam.mio_seed: 312}
prob.solve(solver=cp.GUROBI, verbose=False)
# %% Solve upper level with first-order method
J1 = lambda x, y: 0
J2 = lambda x, y: np.matmul(np.transpose(np.matmul(mat_ones, y) - x_hat), mat_ones)
gstep = lambda k: 5 / (k + 1) ** 0.51
# gstep = lambda k: 3.0
rstep_van = lambda k: 1.0
rstep_con = lambda k: 1.0
# %% Solve problem using the hypergradient class
smallc = -lambd * y_hat
dims_y = np.repeat(dim, n_ag)
cvx_hgm_objective = lambda x, y: cp.Minimize(cp.sum_squares(mat_ones @ y - x_hat))
tol = lambda k: 5 / (k + 1) ** 0.51
hgm = hypergrad.hypergradient_method(
    Q=Q,
    C=C,
    c=smallc,
    dims_y=dims_y,
    dim_x=dim_x,
    A_ineq=A_ineq,
    b_ineq=b_ineq,
    A_eq=None,
    b_eq=None,
    G_ineq=G_ineq,
    H_eq=None,
    upper_obj=np_obj,
    Ax_ineq=Ax_ineq,
    bx_ineq=bx_ineq,
    Ax_eq=None,
    bx_eq=None,
    phiJ1=J1,
    phiJ2=J2,
    gstep=gstep,
    rstep=rstep_van,
    tol_y=tol,
    tol_s=tol,
)
# %%
hgm.run_fixed(inner_iters=1, n_iters=300)
# %%
plt.subplot(2, 3, (1, 2))
ax = plt.gca()

plt.plot(
    hgm.hypergrad.x_log[0],
    hgm.hypergrad.x_log[1],
    marker="o",
    markersize=6,
    c="yellow",
    label="Leader",
)

plt.plot(
    hgm.hypergrad.x_log[0][-1],
    hgm.hypergrad.x_log[1][-1],
    marker="8",
    markersize=12,
    c="yellow",
    markeredgecolor="black",
)

plt.plot(
    np.average(hgm.vi.y_log[[0, 2, 4]], axis=0),
    np.average(hgm.vi.y_log[[1, 3, 5]], axis=0),
    c="cyan",
    label="Fol. Average",
    linestyle="--",
    linewidth=2,
)


plt.plot(
    np.average(hgm.vi.y_log[[0, 2, 4]], axis=0)[-1],
    np.average(hgm.vi.y_log[[1, 3, 5]], axis=0)[-1],
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
    xs = hgm.vi.y_log[ag * dim, :]
    ys = hgm.vi.y_log[ag * dim + 1, :]
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

# plt.gca().set_aspect('equal', adjustable='box')
plt.plot(
    x_top.value[0],
    x_top.value[1],
    marker="*",
    markersize=10,
    markeredgecolor="black",
    c="yellow",
)
plt.plot(
    y_low.value[0],
    y_low.value[1],
    marker="*",
    markersize=10,
    markeredgecolor="black",
    c="red",
)
plt.plot(
    y_low.value[2],
    y_low.value[3],
    marker="*",
    markersize=10,
    markeredgecolor="black",
    c="green",
)
plt.plot(
    y_low.value[4],
    y_low.value[5],
    marker="*",
    markersize=10,
    markeredgecolor="black",
    c="blue",
)
plt.legend()
# %% Plot constraint variable
plt.subplot(2, 3, 3)
plt.plot(hgm.hypergrad.x_log[2])
plt.xlabel("Iterations")
plt.ylabel("Constraint Relaxation")
# %%
hgm.solve_exact(cvx_objective=cvx_hgm_objective)
plt.subplot(234)
hgm.plot_relative_suboptimality()
plt.subplot(235)
hgm.plot_equilibrium_error()
plt.subplot(236)
hgm.plot_sensitivity_error()
plt.tight_layout()
plt.show()
