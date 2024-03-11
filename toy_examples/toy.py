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

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../hypergradient"))
)
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import low_vi, upper
import cvxpy as cp
import distributed_hg

# %% Set problem parameters
dim = 2
n_ag = 3
n = dim * n_ag
lambd = 0.1
y_hat = np.concatenate((np.array([0, 0]), np.array([1, 0]), np.array([0, 1])))
dim_x = 2
x = np.array([0.2, 0.3])
lf = 1 / n_ag + lambd
mu = lambd
# %% Define gradient
C = -1 / n_ag * np.tile(np.eye(dim_x), (n_ag, 1))
Q = 1 / n_ag**2 * np.kron(np.ones((n_ag, n_ag)), np.eye(2)) + lambd * np.eye(n)
grad = lambda x, y: np.matmul(Q, y) + np.matmul(C, x) - lambd * y_hat
# %% Get Jacobians
jacob1 = lambda x, y: C
jacob2 = lambda x, y: Q
# %% Define VI problem
vi = low_vi.lower_vi(x=x, n=n, m=dim_x, grad=grad, J1=jacob1, J2=jacob2, mu=mu, lf=lf)
exact_y = np.linalg.solve(Q, -np.matmul(C, x) + lambd * y_hat)
exact_s = -np.matmul(np.linalg.inv(Q), C)

# %% Solve VI and compare to exact
# vi.run_projection(n_iter=20, log_data=True)
# print("VI solution distance {}".format(np.linalg.norm(vi.y - exact_y)))

# # %% Get Sensitivity Estimate and compare to exact
# vi.run_sensitivity(n_iter=50)
# print("VI sensitivity distance {}".format(np.linalg.norm(vi.s - exact_s)))
# # %% Run Both Simultaneously
# vi.run_proj_sens(n_iter=50,log_data=True)
# print("VI solution distance {}".format(np.linalg.norm(vi.y - exact_y)))
# print("VI sensitivity distance {}".format(np.linalg.norm(vi.s - exact_s)))

# %% Define upper-level
x_hat = np.array([0.3, 0.5])
mat_ones = 1 / n_ag * np.tile(np.eye(2), (1, n_ag))
# mat_ones = 0.5 * np.array(
#     [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
# )
A_mat = np.matmul(mat_ones, exact_s)
b_vec = np.matmul(mat_ones, np.matmul(np.linalg.inv(Q), lambd * y_hat)) - x_hat
np_obj = (
    lambda x: np.linalg.norm(A_mat @ x + np.expand_dims(b_vec, axis=1), axis=0) ** 2
)

x_top = cp.Variable(2)
obj = cp.sum_squares(A_mat @ x_top + b_vec)
prob = cp.Problem(cp.Minimize(obj))
prob.solve()
print("The optimal x is {0} with value {1}".format(x_top.value, prob.value))

# %% Solve upper-level with first-order
J1 = lambda x, y: 0
J2 = lambda x, y: np.matmul(np.transpose(np.matmul(mat_ones, y) - x_hat), mat_ones)
gstep = lambda k: 1 / np.linalg.norm(exact_s) ** 2
rstep_van = lambda k: 2 / (k + 1) ** 0.51
rstep_con = lambda k: 2
upp_van = upper.upper_opt(m=2, J1=J1, J2=J2, gstep=gstep, rstep=rstep_van)
upp_con = upper.upper_opt(m=2, J1=J1, J2=J2, gstep=gstep, rstep=rstep_con)
upp_van.x = np.array([0.0, 0.0])
upp_con.x = np.array([0.0, 0.0])
# %% Run upper-level
n_iters = 600
# Run vanishing steps
vi.clear_states()
for k in range(n_iters):
    vi.x = upp_van.x
    vi.run_proj_sens(n_iter=1, log_data=True)
    upp_van.run_step(y=vi.y, s=vi.s)
# Compute distance of VI solutions
exact_ys_van = np.matmul(exact_s, upp_van.x_log) + np.expand_dims(
    lambd * np.matmul(np.linalg.inv(Q), y_hat), axis=1
)
ys_dists_van = np.linalg.norm(vi.y_log - exact_ys_van, axis=0)
s_dists_van = np.linalg.norm(vi.s_log - np.expand_dims(exact_s, axis=2), axis=(0, 1))

# Run constant steps
vi.clear_states()
for k in range(n_iters):
    vi.x = upp_con.x
    vi.run_proj_sens(n_iter=1, log_data=True)
    upp_con.run_step(y=vi.y, s=vi.s)
exact_ys_con = np.matmul(exact_s, upp_con.x_log) + np.expand_dims(
    lambd * np.matmul(np.linalg.inv(Q), y_hat), axis=1
)
ys_dists_con = np.linalg.norm(vi.y_log - exact_ys_con, axis=0)
s_dists_con = np.linalg.norm(vi.s_log - np.expand_dims(exact_s, axis=2), axis=(0, 1))
# print("Objective is {}".format(np.linalg.norm(A_mat @ upp_van.x + b_vec)**2))

# %% Do plotting
plt.figure()
plt.subplot(221)
colors = ["red", "black", "blue"]
for ag in range(n_ag):
    xs = vi.y_log[ag * dim, :]
    ys = vi.y_log[ag * dim + 1, :]
    # Plot line
    plt.plot(xs, ys, c=colors[ag], label="Follower " + str(ag + 1))
    # Plot start and end
    plt.plot(xs[0], ys[0], marker="+", markersize=12, c=colors[ag])
    plt.plot(xs[-1], ys[-1], marker="o", markersize=12, c=colors[ag])
    # plt.plot(exact_y[ag*dim], exact_y[ag*dim + 1], marker="*", markersize=12, c=colors[ag])
    # Plot "source"
    plt.plot(
        y_hat[ag * dim], y_hat[ag * dim + 1], marker="s", markersize=12, c=colors[ag]
    )

plt.plot(
    upp_con.x_log[0],
    upp_con.x_log[1],
    marker="*",
    markersize=6,
    c="yellow",
    label="Leader",
)

plt.plot(
    np.average(vi.y_log[[0, 2, 4]], axis=0),
    np.average(vi.y_log[[1, 3, 5]], axis=0),
    c="cyan",
    label="Fol. Average",
)
plt.plot(x_hat[0], x_hat[1], marker="d", markersize=12)
plt.plot(x_top.value[0], x_top.value[1], marker="*", markersize=12)
# plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
#  Plot Objective Values
plt.subplot(222)
upper_vals_van = np_obj(upp_van.x_log)
upper_vals_con = np_obj(upp_con.x_log)
plt.semilogy(
    (upper_vals_van - prob.value) / np.max((1e-3, np.abs(prob.value))),
    label="Vanishing",
)
# plt.semilogy((upper_vals_con - prob.value) / np.max((1e-3, np.abs(prob.value))), label="Constant")
plt.grid()
plt.legend()
plt.title("Leader's objective")
plt.xlabel("Iterations")
plt.ylabel("Relative Suboptimality")
# plt.axhline(prob.value, c="yellow")
# Plot distance to VI solution
plt.subplot(223)
plt.loglog(ys_dists_van, label="Vanishing")
plt.loglog(ys_dists_con, label="Constant")
plt.grid()
plt.legend()
plt.title("Distance of VI Solution")
plt.xlabel("Iterations")
plt.ylabel("2-Norm")
# Plot Distance of Sensitivity Derivatives
plt.subplot(224)
plt.loglog(s_dists_van, label="Vanishing")
plt.loglog(s_dists_con, label="Constant")
plt.grid()
plt.legend()
plt.title("Distance of Sensitivity Derivatives")
plt.xlabel("Iterations")
plt.ylabel("Spectral Norm")
# Plot
plt.tight_layout()
plt.show()

# %% Setup Distributed Class
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
    A_ineqdvi.append(None)

    b_ineqdvi.append(None)

    G_ineqdvi.append(None)

dvimu = mu
dvilf = lf

qp_inds = None
simp_proj = None
box_proj = None


def dis_np_obj(x, y):
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
        x = np.expand_dims(x, axis=1)

    n_iters = y.shape[1]
    objs = np.zeros(n_iters)
    for ii in range(n_iters):
        objs[ii] = np.linalg.norm(mat_ones @ y[:, ii] - x_hat) ** 2
    return objs


# %% Instantiate Distributed Hypergradient
tol_y = lambda k: 1 / (k + 1) ** 0.51
tol_s = lambda k: 1 / (k + 1) ** 0.51
dishg = distributed_hg.distributed_hypergradient(
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
    Ax_ineq=None,
    bx_ineq=None,
    Ax_eq=None,
    bx_eq=None,
    qp_inds=qp_inds,
    simp_proj=simp_proj,
    box_proj=box_proj,
    phiJ1=J1,
    phiJ2=J2,
    gstep=gstep,
    rstep=rstep_van,
    tol_y=tol_y,
    tol_s=tol_s,
)
dishg.exact_x = x_top.value
y_low = A_mat @ x_top.value + np.matmul(
    mat_ones, np.matmul(np.linalg.inv(Q), lambd * y_hat)
)
dishg.exact_y = y_low
dishg.exact_val = prob.value
# %% Run Distributed Solver
dishg.run_fixed(inner_iters=1, n_iters=600, log_data=True)
# %% Plot Results
plt.figure()
dishg.plot_relative_suboptimality()
plt.show()
plt.figure()
dishg.plot_equilibrium_error()
plt.show()
plt.figure()
dishg.plot_sensitivity_error()
plt.show()
plt.figure()
dishg.plot_objective(every=1, inner=5)
plt.show()
