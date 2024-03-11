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

import numpy as np
import cvxpy as cp
from low_vi import lower_vi
from upper import upper_opt
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


class hypergradient_method:
    def __init__(
        self,
        Q,
        C,
        c,
        dims_y,
        dim_x,
        A_ineq,
        b_ineq,
        G_ineq,
        A_eq,
        b_eq,
        H_eq,
        upper_obj,
        Ax_ineq,
        bx_ineq,
        Ax_eq,
        bx_eq,
        phiJ1,
        phiJ2,
        gstep,
        rstep,
        tol_y,
        tol_s,
        up_offset=0,
    ):
        # Problem-defining matrices, i.e., the pseudo-gradient is
        # Q * y + C * x + c
        self.Q = Q
        self.C = C
        self.c = c

        # Variable dimensions
        # Array of y dimensions
        self.dims_y = dims_y
        # Number of agents
        self.n_ag = dims_y.size
        # Total followers' dimension
        self.dim_y = dims_y.sum()
        self.dim_x = dim_x

        # Lower-level constraints
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.A_eq = A_eq
        self.b_eq = b_eq

        # Lower-level parametric constraints
        self.G_ineq = G_ineq
        self.H_eq = H_eq

        # Pseudo-gradient properties
        # Lipschitz constant
        self.lf = np.linalg.norm(Q, ord=2)
        # Strong monotonicity
        eigs, _ = np.linalg.eig(self.Q)
        assert eigs.min() > 0, "The provided problem is not strongly monotone."
        self.mu = eigs.min()

        # Define Lower-level VI problem
        self.pgrad = lambda x, y: np.matmul(self.Q, y) + np.matmul(self.C, x) + self.c
        self.pjacob1 = lambda x, y: self.C
        self.pjacob2 = lambda x, y: self.Q
        x0 = np.zeros(self.dim_x)
        self.vi = lower_vi(
            x=x0,
            n=self.dim_y,
            m=self.dim_x,
            grad=self.pgrad,
            J1=self.pjacob1,
            J2=self.pjacob2,
            mu=self.mu,
            lf=self.lf,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            A_ineq=self.A_ineq,
            b_ineq=self.b_ineq,
            G_ineq=self.G_ineq,
            H_eq=self.H_eq,
        )

        # Setup Upper level
        # Upper objective here is a Numpy expression
        self.upper_obj = upper_obj
        self.Ax_ineq = Ax_ineq
        self.bx_ineq = bx_ineq
        self.Ax_eq = Ax_eq
        self.bx_eq = bx_eq
        self.phiJ1 = phiJ1
        self.phiJ2 = phiJ2
        self.gstep = gstep
        self.rstep = rstep

        # Instantiate upper level solver
        self.hypergrad = upper_opt(
            m=self.dim_x,
            J1=self.phiJ1,
            J2=self.phiJ2,
            gstep=self.gstep,
            rstep=self.rstep,
            A_ineq=self.Ax_ineq,
            b_ineq=self.bx_ineq,
            A_eq=self.Ax_eq,
            b_eq=self.bx_eq,
        )

        # Upper level iteration counter
        self.up_iter = 0
        self.low_iter = 0
        # Offsets for iteration counters (only for plotting and evaluating tolerances)
        self.up_off = up_offset
        self.hypergrad.iter = self.up_off
        # Array containing the cumulative lower-level iterations for each upper iteration
        self.low2up = np.array([0])

        # Tolerance sequence
        self.tol_y = tol_y
        self.tol_s = tol_s

        # Exact solution attributes
        self.exact_x = None
        self.exact_y = None
        self.exact_val = None

    def run_fixed(
        self,
        inner_iters,
        n_iters=1,
        timing=False,
    ):
        for k in tqdm(range(n_iters)):
            # Change VI parameter x
            self.vi.x = self.hypergrad.x.copy()

            # Run projection and sensitivity learning
            self.vi.run_proj_sens(n_iter=inner_iters, log_data=False, timing=timing)

            # Run Hypergradient step
            self.hypergrad.run_step(y=self.vi.y, s=self.vi.s, timing=timing)

            # Update counters
            self.up_iter += 1
            self.low_iter += inner_iters
            self.low2up = np.append(self.low2up, np.array([self.low_iter]), axis=0)

    def run_fixed_proj(
        self,
        inner_iters,
        n_iters=1,
    ):
        for k in tqdm(range(n_iters)):
            # Change VI parameter x
            self.vi.x = self.hypergrad.x.copy()

            # Run projection and sensitivity learning
            for iter in range(inner_iters):
                self.vi.run_projection_only(n_iter=1, log_data=False)
                self.vi.run_sensitivity(n_iter=1, log_data=False)

            # Run Hypergradient step
            self.hypergrad.run_step(y=self.vi.y, s=self.vi.s)

            # Update counters
            self.up_iter += 1
            self.low_iter += inner_iters
            self.low2up = np.append(self.low2up, np.array([self.low_iter]), axis=0)

    def run_with_tolerance(self, n_iters=1, timing=False):
        for k in tqdm(range(n_iters)):
            # Change the VI parameter x
            self.vi.x = self.hypergrad.x.copy()

            converged = False
            n_inner = 0
            # Repeat inner loop until tolerance is satisfied
            while not (converged):
                # Run inner iteration and update counter
                self.vi.run_proj_sens(n_iter=1, log_data=False, timing=timing)
                n_inner += 1

                # Equilibrium Error
                eq_error = self.vi.eqlog[-1].copy()
                # Normalize by the dimension
                eq_error /= self.dim_y

                # Sensitivity error
                sens_error = self.vi.senslog[-1].copy()
                # Normalize by the dimension
                sens_error /= self.dim_y

                # Inner iteration has "converged" when both errors are below tolerance
                converged = (eq_error < self.tol_y(self.up_iter + self.up_off)) and (
                    sens_error < self.tol_s(self.up_iter + self.up_off)
                )

            # Run Hypergradient step
            self.hypergrad.run_step(y=self.vi.y, s=self.vi.s, timing=timing)

            # Update counters
            self.up_iter += 1
            self.low_iter += n_inner
            self.low2up = np.append(self.low2up, np.array([self.low_iter]), axis=0)

    def solve_exact(self, cvx_objective):
        # The objective should a cvxpy expression
        x_top = cp.Variable(self.dim_x)
        y_low = cp.Variable(self.dim_y)
        lagrangian_low = self.Q @ y_low + self.C @ x_top + self.c

        # Define lower level problem via big M reformulation
        constraints = []
        if not (self.A_ineq is None):
            n_ineq = self.b_ineq.size
            lambdas = cp.Variable(n_ineq)
            bigM = 4e0
            binaries = cp.Variable(n_ineq, boolean=True)
            lagrangian_low += self.A_ineq.transpose() @ lambdas
            if self.G_ineq is None:
                constraints += (
                    [0 <= self.b_ineq - self.A_ineq @ y_low]
                    + [self.b_ineq - self.A_ineq @ y_low <= bigM * (1 - binaries)]
                    + [0 <= lambdas]
                    + [lambdas <= bigM * binaries]
                )
            else:
                constraints += (
                    [0 <= self.b_ineq + self.G_ineq @ x_top - self.A_ineq @ y_low]
                    + [
                        self.b_ineq + self.G_ineq @ x_top - self.A_ineq @ y_low
                        <= bigM * (1 - binaries)
                    ]
                    + [0 <= lambdas]
                    + [lambdas <= bigM * binaries]
                )

        if not (self.A_eq is None):
            n_eq = self.b_eq.size
            nus = cp.Variable(n_eq)
            lagrangian_low += self.A_eq.transpose() @ nus
            if self.H_eq is None:
                constraints += [self.A_eq @ nus - self.b_eq == 0]
            else:
                constraints += [self.A_eq @ nus - self.b_eq - self.H_eq @ x_top == 0]

        # Set low level lagrangian to be zero
        constraints += [lagrangian_low == 0]

        # Define upper level constraints
        if not (self.Ax_ineq is None):
            constraints += [self.Ax_ineq @ x_top <= self.bx_ineq]

        if not (self.Ax_eq is None):
            constraints += [self.Ax_eq @ x_top <= self.bx_eq]

        prob = cp.Problem(
            objective=cvx_objective(x_top, y_low), constraints=constraints
        )
        prob.solve(solver=cp.MOSEK, verbose=False)

        # Populate attributes with the exact solution
        self.exact_x = x_top.value
        self.exact_y = y_low.value
        self.exact_val = prob.value
        return self.exact_val

    def compute_objective(self, every=10, inner=5):
        # every: frequency of computing the objective value
        n_iters = self.hypergrad.x_log.shape[1]
        # Keep a save of the current y
        y_curr = self.vi.y.copy()
        # Number of objective computations
        n_comps = (np.floor(n_iters / every)).astype(int)
        objs = np.zeros(n_comps)
        for k in tqdm(range(n_comps)):
            ktmp = k * every
            xk = self.hypergrad.x_log[:, ktmp]
            self.vi.x = xk.copy()
            self.vi.run_projection_only(n_iter=inner, log_data=False)
            objs[k] = self.upper_obj(xk, self.vi.y)

        self.vi.y = y_curr
        self.vi.x = self.hypergrad.x.copy()

        return objs

    ################################################################################
    #  Plotting Functionalities
    ################################################################################
    def plot_relative_suboptimality(self):
        assert not (
            self.exact_val is None
        ), "Cannot plot relative suboptimality without exact solution"
        y_up = self.vi.y_log[:, self.low2up]
        upper_vals = self.upper_obj(self.hypergrad.x_log, y_up)
        opt_value = np.min((upper_vals[-1], self.exact_val))

        # Plot
        plt.semilogy(
            np.abs(upper_vals - opt_value) / (np.max((1e-3, np.abs(opt_value)))),
        )
        # Plot appearance
        plt.grid()
        plt.title("Leader's Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("Relative Suboptimality")

    def plot_equilibrium_error(self, logged=True):
        if logged:
            y_diffs = self.vi.eqlog.copy()
        else:
            y_diffs = np.linalg.norm(np.diff(self.vi.y_log, axis=1), axis=0)
        # Normalize by dimension
        y_diffs /= self.dim_y
        plt.semilogy(
            y_diffs,
            label="Inner Loop",
            linewidth=1,
            color="blue",
            marker="o",
            markersize=4,
        )
        plt.semilogy(
            self.low2up[1:] - 1,
            y_diffs[self.low2up[1:] - 1],
            label="Outer Loop",
            linestyle="None",
            linewidth=4,
            color="red",
            marker="o",
            markersize=10,
        )
        plt.semilogy(
            self.low2up,
            self.tol_y(np.arange(self.up_off, self.up_iter + 1 + self.up_off)),
            label="Tolerance",
            linewidth=2,
            color="green",
            linestyle="--",
            drawstyle="steps-post",
            marker="o",
            markersize=10,
        )
        plt.grid()
        plt.legend()
        plt.title("Normalized Equilibrium Error")
        plt.xlabel("Iterations")
        plt.ylabel("2-Norm Normalized Error")

    def plot_sensitivity_error(self, logged=True):
        if logged:
            s_diffs = self.vi.senslog.copy()
        else:
            s_diffs = np.linalg.norm(np.diff(self.vi.s_log, axis=2), axis=(0, 1), ord=2)
        # Normalize by dimension
        s_diffs /= self.dim_y
        plt.semilogy(
            s_diffs,
            label="Inner Loop",
            linewidth=1,
            color="blue",
            marker="o",
            markersize=4,
        )
        plt.semilogy(
            self.low2up[1:] - 1,
            s_diffs[self.low2up[1:] - 1],
            label="Outer Loop",
            linestyle="None",
            linewidth=4,
            color="red",
            marker="o",
            markersize=10,
        )
        plt.semilogy(
            self.low2up,
            self.tol_s(np.arange(self.up_off, self.up_iter + 1 + self.up_off)),
            label="Tolerance",
            linewidth=2,
            color="green",
            linestyle="--",
            drawstyle="steps-post",
            marker="o",
            markersize=10,
        )

        plt.grid()
        plt.legend()
        plt.title("Normalized Sensitivity Error")
        plt.xlabel("Iterations")
        plt.ylabel("Normalized Spectral Norm Error")

    def plot_objective(self, every=10, inner=5):
        objs = self.compute_objective(every=every, inner=inner)
        times = np.arange(self.up_off, objs.size * every + self.up_off, every)
        plt.plot(times, objs, linewidth=2, color="blue", marker="o", markersize=4)

        plt.grid()
        plt.title("Leader's Objective")
        plt.xlabel("Iterations")
        plt.ylabel("Objective Value")

    def plot_inner_times(self):
        plt.plot(
            self.vi.times_log, color="blue", linestyle="None", marker="o", markersize=4
        )
        plt.grid()
        plt.title("Inner Loop CPU time")
        plt.xlabel("Iterations")
        plt.ylabel("Time [s]")

    def plot_leader_times(self):
        plt.plot(
            self.hypergrad.times_log,
            color="red",
            linestyle="None",
            marker="o",
            markersize=4,
        )
        plt.grid()
        plt.title("CPU time of leader")
        plt.xlabel("Iterations")
        plt.ylabel("Time [s]")

    ################################################################################
    # Logging Functionalities
    ################################################################################
    def clear_log(self):
        self.up_off += self.up_iter
        self.up_iter = 0
        self.low_iter = 0
        self.low2up = np.array([0])
        self.vi.clear_log()
        self.hypergrad.clear_log()
        self.hypergrad.iter = self.up_off

    def save_to_json(self, path):
        data_dict = {
            "up_iter": self.up_iter,
            "low_iter": self.low_iter,
            "up_offset": self.up_off,
            "low2up": self.low2up.tolist(),
            "x": self.hypergrad.x.tolist(),
            "x_log": self.hypergrad.x_log.tolist(),
            "y": self.vi.y.tolist(),
            "s": np.real(self.vi.s).tolist(),
            "eqlog": self.vi.eqlog.tolist(),
            "senslog": self.vi.senslog.tolist(),
            "vitimes": self.vi.times_log.tolist(),
            "leadtimes": self.hypergrad.times_log.tolist(),
            # "y_log": self.vi.y_log.tolist(),
            # "s_log": self.vi.s_log.tolist(),
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data_dict, file, ensure_ascii=False, indent=4)

    def parse_json(self, path):
        with open(path) as file:
            data_dict = json.loads(file.read())

        return (
            data_dict["up_iter"],
            data_dict["low_iter"],
            data_dict["up_offset"],
            np.array(data_dict["low2up"]),
            np.array(data_dict["x"]),
            np.array(data_dict["x_log"]),
            np.array(data_dict["y"]),
            np.array(data_dict["s"]),
            np.array(data_dict["eqlog"]),
            np.array(data_dict["senslog"]),
            np.array(data_dict["vitimes"]),
            np.array(data_dict["leadtimes"]),
            # np.array(data_dict["y_log"]),
            # np.array(data_dict["s_log"]),
        )

    def load_json(self, paths):
        # Paths is an array of absolute paths containing data from a simulation
        # ordered appropriately.
        up_its = 0
        low_its = 0
        low2ups = np.empty((0))
        eqlogs = np.empty(0)
        senslogs = np.empty(0)
        x_logs = np.empty((self.hypergrad.m, 0))
        vitimes = np.empty(0)
        leadtimes = np.empty(0)
        # y_logs = np.empty((self.vi.n, 0))
        # s_logs = np.empty((self.vi.n, self.hypergrad.m, 0))
        for path in paths:
            (
                up_iter,
                low_iter,
                up_offset,
                low2up,
                x,
                x_log,
                y,
                s,
                eqlog,
                senslog,
                vi_time,
                lead_time,
                # y_log,
                # s_log,
            ) = self.parse_json(path)
            up_its += up_iter
            low2up += low_its
            low2ups = np.append(low2ups, low2up[:-1], axis=0)
            eqlogs = np.append(eqlogs, eqlog)
            senslogs = np.append(senslogs, senslog)
            x_logs = np.append(x_logs, x_log[:, :-1], axis=1)
            vitimes = np.append(vitimes, vi_time)
            leadtimes = np.append(leadtimes, lead_time)
            low_its += low_iter
            # y_logs = np.append(y_logs, y_log[:, :-1], axis=1)
            # s_logs = np.append(s_logs, s_log[:, :, :-1], axis=2)

        self.up_iter = up_its
        self.hypergrad.iter = up_its
        self.low_iter = low_its
        low2ups = np.append(low2ups, np.expand_dims(low2up[-1], axis=0), axis=0)
        self.low2up = low2ups.astype(int)
        x_logs = np.append(x_logs, np.expand_dims(x_log[:, -1], axis=1), axis=1)
        # y_logs = np.append(y_logs, np.expand_dims(y_log[:, -1], axis=1), axis=1)
        # s_logs = np.append(s_logs, np.expand_dims(s_log[:, :, -1], axis=2), axis=2)
        self.hypergrad.x_log = x_logs
        self.vi.times_log = vitimes
        self.vi.eqlog = eqlogs
        self.vi.senslog = senslogs
        # self.vi.y_log = y_logs
        # self.vi.s_log = s_logs
        self.hypergrad.x = x
        self.hypergrad.times_log = leadtimes
        # Set the VI states
        self.vi.y = y
        self.vi.s = s

    def continue_from_json(self, path):
        (
            up_iter,
            low_iter,
            up_offset,
            low2up,
            x,
            x_log,
            y,
            s,
            eqlog,
            senslog,
            vi_time,
            lead_time,
            # y_log,
            # s_log,
        ) = self.parse_json(path)
        self.up_iter = 0
        self.low_iter = 0
        self.up_off = up_iter
        self.hypergrad.iter = up_iter
        self.low2up = np.array([0])
        self.hypergrad.x = x.copy()
        self.vi.y = y.copy()
        self.vi.s = s.copy()
