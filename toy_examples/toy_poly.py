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
import numpy as np
import matplotlib.pyplot as plt
import low_vi, upper, hypergrad
import cvxpy as cp
from matplotlib.patches import Rectangle
from distributed_hg import distributed_hypergradient as dhg

# matplotlib.rcParams['figure.dpi'] = 200

# %% Set problem parameters
dim = 2
n_ag = 3
n = dim * n_ag
lambd = 1.0
y_hat = np.concatenate((np.array([0.0, 0.0]), np.array([4.0, 0]), np.array([0, 2.0])))
dim_x = 2
x = np.array([0.2, 0.3])
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
# %% Define gradient
C = -1 / n_ag * np.tile(np.eye(dim_x), (n_ag, 1))
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
    G_ineq=None,
    A_eq=None,
)
# %% Define upper level
x_hat = np.array([7.0, 0.3])
mat_ones = 1 / n_ag * np.tile(np.eye(2), (1, n_ag))
# mat_ones = 0.5 * np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
np_obj = (
    lambda x, y: np.linalg.norm(mat_ones @ y - np.expand_dims(x_hat, axis=1), axis=0)
    ** 2
)
x_rec_size = 4.0
x_rec_bot_left = np.array([-1.0, -1.0])
lrec_corner = x_rec_size * x_rec_bot_left
lrec_dims = 2 * np.array([x_rec_size, x_rec_size])
Ax_ineq = np.block([[np.eye(2)], [-np.eye(2)]])
bx_ineq = np.array(
    [
        lrec_corner[0] + lrec_dims[0],
        lrec_corner[1] + lrec_dims[1],
        -lrec_corner[0],
        -lrec_corner[1],
    ]
)
# %% Exact solution using MIQP
x_top = cp.Variable(dim_x)
y_low = cp.Variable(n)
lambdas = cp.Variable(n_ineq)
binaries = cp.Variable(n_ineq, boolean=True)
bigM = 4e0
cvx_objective = cp.Minimize(cp.sum_squares(mat_ones @ y_low - x_hat))
constraints = (
    [Ax_ineq @ x_top <= bx_ineq]
    + [Q @ y_low + C @ x_top - lambd * y_hat + A_ineq.transpose() @ lambdas == 0]
    + [0 <= b_ineq - A_ineq @ y_low]
    + [b_ineq - A_ineq @ y_low <= bigM * (1 - binaries)]
    + [0 <= lambdas]
    + [lambdas <= bigM * binaries]
)
prob = cp.Problem(cvx_objective, constraints)
prob.solve(solver=cp.GUROBI, verbose=False)
# %% Solve upper level with first-order method
J1 = lambda x, y: 0
J2 = lambda x, y: np.matmul(np.transpose(np.matmul(mat_ones, y) - x_hat), mat_ones)
gstep = lambda k: 5 / (k + 1) ** 0.51
# gstep = lambda k: 3.0
rstep_van = lambda k: 1.0
rstep_con = lambda k: 1.0
upp_van = upper.upper_opt(
    m=2, J1=J1, J2=J2, gstep=gstep, rstep=rstep_van, A_ineq=Ax_ineq, b_ineq=bx_ineq
)
upp_con = upper.upper_opt(
    m=2, J1=J1, J2=J2, gstep=gstep, rstep=rstep_con, A_ineq=Ax_ineq, b_ineq=bx_ineq
)
upp_van.x = np.array([0.0, 0.0])
upp_con.x = np.array([0.0, 0.0])
# %% Run upper level
n_iters = 300
inner_iters = 3
# Run vanishing steps
# upp_van.clear_states(2*np.random.randn(2))
upp_van.clear_states(x0=np.array([0.0, 0.0]))
vi.clear_states(y0=np.random.randn(6))
for k in range(n_iters):
    vi.x = upp_van.x
    vi.run_proj_sens(n_iter=inner_iters, log_data=True)
    upp_van.run_step(y=vi.y, s=vi.s)

upper_vals_van = np_obj(upp_van.x_log, vi.y_log)
y_diffs_van = np.linalg.norm(np.diff(vi.y_log, axis=1), axis=0)
s_diffs_van = np.linalg.norm(np.diff(vi.s_log, axis=2), axis=(0, 1), ord=2)
# %% Run constant steps
# vi.clear_states()
# n_iters = 50
# for k in range(n_iters):
#     vi.x = upp_con.x
#     vi.run_proj_sens(n_iter=inner_iters, log_data=True)
#     upp_con.run_step(y=vi.y, s=vi.s)
# upper_vals_con = np_obj(vi.y_log)
# y_diffs_con = np.linalg.norm(np.diff(vi.y_log, axis=1), axis=0)
# s_diffs_con = np.linalg.norm(np.diff(vi.s_log, axis=2), axis=(0, 1))
# %% Do plotting
plt.subplot(221)
ax = plt.gca()

plt.plot(
    upp_van.x_log[0],
    upp_van.x_log[1],
    marker="o",
    markersize=6,
    c="yellow",
    label="Leader",
)

plt.plot(
    upp_van.x_log[0][-1],
    upp_van.x_log[1][-1],
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
# %% Plot Objective Values
plt.subplot(222)
opt_value = np.min((upper_vals_van[-1], prob.value))
plt.semilogy(
    np.abs(upper_vals_van[inner_iters - 1 :: inner_iters] - opt_value)
    / (np.max((1e-1, opt_value))),
    # label="Vanishing",
)
# plt.semilogy(upper_vals_con, label="Constant")
plt.grid()
# plt.legend()
plt.title("Leader's Convergence")
plt.xlabel("Iterations")
plt.ylabel("Relative suboptimality")
# %% Plot distance to VI solution
plt.subplot(223)
plt.semilogy(
    y_diffs_van,
    label="Inner Loop",
    linewidth=1,
    color="blue",
    marker="o",
    markersize=4,
)
plt.semilogy(
    np.arange(0, n_iters * inner_iters)[inner_iters - 1 :: inner_iters],
    y_diffs_van[inner_iters - 1 :: inner_iters],
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
# # plt.loglog(y_diffs_con, label="Constant")
plt.grid()
plt.legend()
plt.title("Equilibrium Error")
plt.xlabel("Iterations")
plt.ylabel("2-Norm Error")
# %% Plot Distance of Sensitivity Derivatives
plt.subplot(224)
plt.semilogy(
    s_diffs_van,
    label="Inner Loop",
    linewidth=1,
    color="blue",
    marker="o",
    markersize=4,
)
plt.semilogy(
    np.arange(0, n_iters * inner_iters)[inner_iters - 1 :: inner_iters],
    s_diffs_van[inner_iters - 1 :: inner_iters],
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
# %% Solve problem using the hypergradient class
smallc = -lambd * y_hat
dims_y = np.repeat(dim, n_ag)
cvx_hgm_objective = lambda x, y: cp.Minimize(cp.sum_squares(mat_ones @ y - x_hat))
tol = lambda k: 1 / (k + 1) ** 0.51
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
    G_ineq=None,
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
hgm.run_fixed(inner_iters=1, n_iters=600)
hgm.plot_equilibrium_error()
plt.show()
hgm.plot_sensitivity_error()
plt.show()
# %% Define distributed solver parameters
dvix = hgm.hypergrad.x
dims_y = np.ones(n_ag, dtype=int) * dim
dvim = dim_x
Qs = []
Cs = []
cs = []
dvigrad = []
dviJ1 = []
dviJ2 = []
A_eqdvi = []
b_eqdvi = []
H_eqdvi = []
A_ineqdvi = []
b_ineqdvi = []
G_ineqdvi = []
for ag in range(n_ag):
    start = dims_y[:ag].sum()
    end = dims_y[: ag + 1].sum()
    print(start)
    print(end)
    # PPG
    Qs.append(Q[start:end, :])
    Cs.append(C[start:end, :])
    cs.append(-lambd * y_hat[start:end])
    ppg = (
        lambda x, y, *, ag=ag: np.matmul(Qs[ag], np.concatenate(y, axis=0))
        + np.matmul(Cs[ag], x)
        + cs[ag]
    )
    dvigrad.append(ppg)

    # Jacobian X
    jacx = lambda x, y, *, ag=ag: Cs[ag]
    dviJ1.append(jacx)

    # Jacobian Y
    jacy = lambda x, y, *, ag=ag: Qs[ag]
    dviJ2.append(jacy)

    # Equality constraints
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

qp_inds = None
simp_proj = None
box_proj = None


# %%
def dis_np_obj(x, y):
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
        x = np.expand_dims(x, axis=1)

    n_iters = y.shape[1]
    objs = np.zeros(n_iters)
    for ii in range(n_iters):
        objs[ii] = np.linalg.norm(mat_ones @ y[:, ii] - x_hat)
    return objs


dishg = dhg(
    Q=Qs,
    C=Cs,
    c=cs,
    dims_y=dims_y,
    dim_x=dim_x,
    A_ineq=A_ineqdvi,
    b_ineq=b_ineqdvi,
    G_ineq=G_ineqdvi,
    A_eq=A_eqdvi,
    b_eq=b_eqdvi,
    H_eq=H_eqdvi,
    upper_obj=dis_np_obj,
    Ax_ineq=Ax_ineq,
    bx_ineq=bx_ineq,
    Ax_eq=None,
    bx_eq=None,
    qp_inds=qp_inds,
    simp_proj=simp_proj,
    box_proj=box_proj,
    phiJ1=J1,
    phiJ2=J2,
    gstep=gstep,
    rstep=rstep_van,
    tol_y=tol,
    tol_s=tol,
)
dishg.exact_x = x_top.value
dishg.exact_y = y_low.value
dishg.exact_val = prob.value
# %%
dishg.run_fixed(inner_iters=1, n_iters=600)
# %%
plt.figure()
dishg.plot_equilibrium_error()
plt.show()
plt.figure()
dishg.plot_sensitivity_error()
plt.show()
plt.figure()
dishg.plot_objective(every=1, inner=5)
plt.show()
