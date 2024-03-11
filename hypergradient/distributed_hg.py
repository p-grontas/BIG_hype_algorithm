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
import scipy
from distributed_vi import dis_vi
from upper import upper_opt
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from scipy.sparse import csc_array


class distributed_hypergradient:
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
        qp_inds,
        simp_proj,
        box_proj,
        phiJ1,
        phiJ2,
        gstep,
        rstep,
        tol_y,
        tol_s,
        up_offset=0,
        step_guess=None,
    ):
        # Problem-defining matrices, i.e., the pseudo-gradient is
        # Q * y + C * x + c
        # The matrices are given as arrays for each agent!
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
        if step_guess is None:
            Q_all = np.vstack(self.Q)
            # self.lf = np.linalg.norm(Q_all, ord=2)
            # Faster computation
            self.lf = np.sqrt(
                scipy.linalg.eigh(
                    Q_all.transpose() @ Q_all,
                    eigvals_only=True,
                    subset_by_index=[self.dim_y - 1, self.dim_y - 1],
                )
            )
            # Strong monotonicity
            eigs = scipy.linalg.eigvals(Q_all)
            assert eigs.min() > 0, "The provided problem is not strongly monotone."
            self.mu = eigs.min()
        else:
            self.lf = 1.0
            self.mu = 1.0

        # Define Lower-level VI problem
        self.pgrad = []
        self.pjacob1 = []
        self.pjacob2 = []
        for ag in range(self.n_ag):
            ppg = (
                lambda x, y, *, ag=ag: np.matmul(self.Q[ag], np.concatenate(y, axis=0))
                + np.matmul(self.C[ag], x)
                + self.c[ag]
            )
            self.pgrad.append(ppg)
            jacx = lambda x, y, *, ag=ag: self.C[ag]
            self.pjacob1.append(jacx)
            # jacy = lambda x, y, *, ag=ag: self.Q[ag]
            jacy = lambda x, y, *, ag=ag: csc_array(self.Q[ag])
            self.pjacob2.append(jacy)
        x0 = np.zeros(self.dim_x)

        self.vi = dis_vi(
            x=x0,
            dims_y=dims_y,
            m=self.dim_x,
            grad=self.pgrad,
            J1=self.pjacob1,
            J2=self.pjacob2,
            mu=self.mu,
            lf=self.lf,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            H_eq=self.H_eq,
            A_ineq=self.A_ineq,
            b_ineq=self.b_ineq,
            G_ineq=self.G_ineq,
            step=step_guess,
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
        self.qp_inds = qp_inds
        self.simp_proj = simp_proj
        self.box_proj = box_proj

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
            qp_inds=self.qp_inds,
            simp_proj=self.simp_proj,
            box_proj=box_proj,
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
        log_data=False,
    ):
        for k in tqdm(range(n_iters)):
            # Change VI parameter x
            self.vi.x = self.hypergrad.x.copy()

            # Run projection and sensitivity learning
            self.vi.run_proj_sens(n_iter=inner_iters, log_data=log_data, timing=timing)

            # Aggregate all followers' decision
            y_all = np.concatenate(self.vi.y, axis=0)
            s_all = np.vstack(self.vi.s)

            # Run Hypergradient step
            self.hypergrad.run_step(y=y_all, s=s_all, timing=timing)

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
            y_all = np.concatenate(self.vi.y, axis=0)
            s_all = np.vstack(self.vi.s)
            self.hypergrad.run_step(y=y_all, s=s_all, timing=timing)

            # Update counters
            self.up_iter += 1
            self.low_iter += n_inner
            self.low2up = np.append(self.low2up, np.array([self.low_iter]), axis=0)

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
            y_all = np.concatenate(self.vi.y, axis=0)
            objs[k] = self.upper_obj(xk, y_all)

        # Restore y and x variables (in order to continue running HG)
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
        # plt.title("Normalized Equilibrium Error")
        plt.xlabel("Iteration $k$")
        plt.ylabel("Equilibrium Error")

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
        plt.title("Sensitivity Error")
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

    def plot_max_ag_times(self, pnd=False, sens=False, both=True):
        pnd_times = np.max(self.vi.times_pnd, axis=0)
        sens_times = np.max(self.vi.times_sens, axis=0)
        both_times = np.max(self.vi.times_pnd + self.vi.times_sens, axis=0)
        if both:
            plt.plot(
                both_times,
                color="blue",
                linestyle="None",
                marker="o",
                markersize=5,
                label="Total",
            )
        if pnd:
            plt.plot(
                pnd_times,
                color="red",
                linestyle="None",
                marker="+",
                markersize=5,
                label="Proj. & Diff.",
            )
        if sens:
            plt.plot(
                sens_times,
                color="green",
                linestyle="None",
                marker="*",
                markersize=5,
                label="Sensitivity",
            )
        plt.legend()
        plt.grid()
        plt.title("Maximum CPU time of agents")
        plt.xlabel("Iterations")
        plt.ylabel("Time [s]")

    def plot_all_ag_times(self):
        both_times = self.vi.times_pnd + self.vi.times_sens
        for ag in range(self.n_ag):
            plt.plot(
                both_times[ag, :],
                linestyle="None",
                marker="o",
                markersize=4,
                label="Agent " + str(ag + 1),
            )
        plt.legend()
        plt.grid()
        plt.title("CPU time of agents")
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

    def plot_rel_obj_convergence(self, every=10, inner=5, maw=0):
        objs = self.compute_objective(every=every, inner=inner)
        rel_diffs = np.abs(np.diff(objs) / objs[1:])

        times = np.arange(self.up_off, objs.size * every + self.up_off, every)
        plt.plot(
            times[1:],
            rel_diffs,
            linewidth=2,
            color="blue",
            marker="o",
            markersize=4,
            label="$|f(x^{k+1}) - f(x^{k})|/|f(x^{k+1})|$",
        )
        # Moving average window
        if maw > 0:
            rel_diffs_maw = np.convolve(rel_diffs, np.ones(maw), "same") / maw
            plt.plot(
                times[1:],
                rel_diffs_maw,
                linewidth=2,
                color="red",
                marker="o",
                markersize=2,
                label="Moving Average (window = " + str(maw) + ")",
            )

        plt.grid()
        plt.legend()
        plt.title("Leader's Objective")
        plt.xlabel("Iteration $k$")
        plt.ylabel("Normalized Objective Difference")
        plt.yscale("log")

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
            "y": np.concatenate(self.vi.y, axis=0).tolist(),
            "s": np.real(np.vstack(self.vi.s)).tolist(),
            "eqlog": self.vi.eqlog.tolist(),
            "senslog": self.vi.senslog.tolist(),
            "vitimespnd": self.vi.times_pnd.tolist(),
            "vitimessens": self.vi.times_sens.tolist(),
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
            np.array(data_dict["vitimespnd"]),
            np.array(data_dict["vitimessens"]),
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
        vitimespnd = np.empty((self.n_ag, 0))
        vitimessens = np.empty((self.n_ag, 0))
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
                vi_time_pnd,
                vi_time_sens,
                lead_time,
                # y_log,
                # s_log,
            ) = self.parse_json(path)
            # Set the upper offset to zero when loading data into an existing object
            self.up_off = 0
            up_its += up_iter
            low2up += low_its
            low2ups = np.append(low2ups, low2up[:-1], axis=0)
            eqlogs = np.append(eqlogs, eqlog)
            senslogs = np.append(senslogs, senslog)
            x_logs = np.append(x_logs, x_log[:, :-1], axis=1)
            vitimespnd = np.append(vitimespnd, vi_time_pnd, axis=1)
            vitimessens = np.append(vitimessens, vi_time_sens, axis=1)
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
        self.vi.times_pnd = vitimespnd
        self.vi.times_sens = vitimessens
        self.vi.eqlog = eqlogs
        self.vi.senslog = senslogs
        # self.vi.y_log = y_logs
        # self.vi.s_log = s_logs
        self.hypergrad.x = x
        self.hypergrad.times_log = leadtimes
        # Set the VI states
        end = np.zeros(self.n_ag)
        for ag in range(self.n_ag):
            end[ag] = self.dims_y[: ag + 1].sum()
        end = end.astype(int)
        self.vi.y = np.split(y.copy(), end[:-1])
        self.vi.s = np.split(s.copy(), end[:-1])

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
        self.hypergrad.x = x
        # Set the VI states
        end = np.zeros(self.n_ag)
        for ag in range(self.n_ag):
            end[ag] = self.dims_y[: ag + 1].sum()
        end = end.astype(int)
        self.vi.y = np.split(y.copy(), end[:-1])
        self.vi.s = np.split(s.copy(), end[:-1])

    def save_total_time(self, path, n_ag, tsteps, best_obj, distributed=True):
        # Followers' time
        if distributed:
            fol_time = np.sum(np.max(self.vi.times_pnd + self.vi.times_sens, axis=0))
        else:
            fol_time = np.sum(self.vi.times_pnd + self.vi.times_sens)
        # Leader's time
        lead_time = np.sum(self.hypergrad.times_log)
        # Total time
        tot_time = fol_time + lead_time

        # Data to be saved
        new_row = np.array([[n_ag, tsteps, tot_time, best_obj]])
        # Save data
        if os.path.isfile(path):
            mode = "r+"
        else:
            mode = "a+"
        with open(path, mode, encoding="utf-8") as file:
            if not (os.path.getsize(path) == 0):
                data_dict = json.loads(file.read())
                updated_array = np.append(np.array(data_dict["data"]), new_row, axis=0)
            else:
                updated_array = new_row

            data_dict = {"data": updated_array.tolist()}

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data_dict, file, ensure_ascii=False, indent=4)
