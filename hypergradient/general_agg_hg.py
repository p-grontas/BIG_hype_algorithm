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
import upper
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import distributed_hg
import general_agg_vi


class general_agg_hypergradient(distributed_hg.distributed_hypergradient):
    def __init__(
        self,
        dims_y,
        dim_x,
        pgrad,
        pjacob1,
        pjacob2,
        pjacob3,
        A_ineq,
        b_ineq,
        G_ineq,
        A_eq,
        b_eq,
        H_eq,
        mu,
        lf,
        upper_obj,
        Ax_ineq,
        bx_ineq,
        Ax_eq,
        bx_eq,
        qp_inds,
        simp_proj,
        phiJ1,
        phiJ2,
        gstep,
        rstep,
        tol_y,
        tol_s,
        up_offset=0,
        agg_lead=False,
        step_guess=None,
    ):
        # Lower-level problem functions
        # Pseudo-gradient for each agent with arguments: (x, y_i, \sum y_i)
        self.pgrad = pgrad
        # Partial Jacobian w.r.t. x
        self.pjacob1 = pjacob1
        # Partial Jacobian w.r.t. y_i
        self.pjacob2 = pjacob2
        # Partial Jacobian w.r.t. \sum y_i
        self.pjacob3 = pjacob3

        # Variable dimensions
        self.dims_y = dims_y
        # Number of Agents
        self.n_ag = dims_y.size
        # Total followers' dimenions
        self.dim_y = dims_y.sum()
        # Leader's dimension
        self.dim_x = dim_x

        # Lower-level Constraints
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.A_eq = A_eq
        self.b_eq = b_eq

        # Lower-level Parametric Constraints
        self.G_ineq = G_ineq
        self.H_eq = H_eq

        # Pseudo-gradient Properties (these have to be computed/approximated analytically)
        if step_guess is None:
            # Lipschitz Constant
            self.lf = lf
            # Strong Monotonicity Constant
            self.mu = mu
            # Contraction Constant assuming that step = mu/lf**2
            gamma = mu / np.square(lf)
            self.eta = np.sqrt(1 - gamma * (2 * mu - gamma * lf**2))
        else:
            # Placeholders for the constants
            self.lf = 1.0
            self.mu = 1.0
            self.eta = 0.95

        # Define Lower-level VI problem
        x0 = np.zeros(self.dim_x)
        self.vi = general_agg_vi.gen_agg_vi(
            x=x0,
            dims_y=self.dims_y,
            m=self.dim_x,
            grad=self.pgrad,
            J1=self.pjacob1,
            J2=self.pjacob2,
            J3=self.pjacob3,
            mu=self.mu,
            lf=self.lf,
            step=step_guess,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            H_eq=self.H_eq,
            A_ineq=self.A_ineq,
            b_ineq=self.b_ineq,
            G_ineq=self.G_ineq,
        )

        # Setup Upper level
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

        # Instantiate upper level solver
        self.hypergrad = upper.upper_opt(
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
        )

        # Upper level iteration counter
        self.up_iter = 0
        self.low_iter = 0
        # Offsets for iteration counters (only for plotting and evaluating tolerances)
        self.up_off = up_offset
        self.hypergrad.iter = self.up_off
        # Array containing the cumulative lower-level iterations for each upper iteration
        self.low2up = np.array([0])

        # Tolerance sequences
        self.tol_y = tol_y
        self.tol_s = tol_s

        # Exact solution attributes
        self.exact_x = None
        self.exact_y = None
        self.exact_val = None

        # Choose how the leader's gradient is computed
        self.agg_lead = agg_lead
        if self.agg_lead:
            self.run_fixed = self.run_fixed_agg_lead

    def run_fixed_agg_lead(self, inner_iters, n_iters=1, timing=False, log_data=False):
        for k in tqdm(range(n_iters)):
            # Change VI parameter x
            self.vi.x = self.hypergrad.x.copy()

            # Run projection and sensitivity learning
            self.vi.run_proj_sens(n_iter=inner_iters, log_data=False, timing=timing)

            # Aggregate all followers' decisions
            y_agg = self.vi.y_agg.copy()
            s_agg = self.vi.s_agg.copy()

            # Run Hypergradient step
            self.hypergrad.run_step(y=y_agg, s=s_agg, timing=timing)

            # Update counters
            self.up_iter += 1
            self.low_iter += inner_iters
            self.low2up = np.append(self.low2up, np.array([self.low_iter]), axis=0)

    def plot_leader_times(self, eq_agg=False, sens_agg=False, all=True):
        total_times = (
            self.hypergrad.times_log + self.vi.times_agg_eq + self.vi.times_agg_sens
        )
        # Plot direct leader time, i.e., computing gradient and projecting
        plt.plot(
            self.hypergrad.times_log,
            color="red",
            linestyle="None",
            marker="*",
            markersize=8,
            label="Grad. & Proj.",
        )
        # Plot time in aggregating equilibrium variables
        if eq_agg:
            plt.plot(
                self.vi.times_agg_eq,
                color="green",
                linestyle="None",
                marker="+",
                markersize=8,
                label="$ \sum y_i $",
            )
        # Plot time in aggregating sensitivity estimates
        if sens_agg:
            plt.plot(
                self.vi.times_agg_sens,
                color="blue",
                linestyle="None",
                marker="s",
                markersize=8,
                label="$ \sum s_i$",
            )
        # Plot total time of the leader
        if all:
            plt.plot(
                total_times,
                color="black",
                linestyle="None",
                marker="o",
                markersize=8,
                label="Total",
            )
        plt.grid()
        plt.legend()
        plt.title("CPU time of leader")
        plt.xlabel("Iterations")
        plt.ylabel("Time [s]")

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
            "leadaggeqt": self.vi.times_agg_eq.tolist(),
            "leadaggsenst": self.vi.times_agg_sens.tolist(),
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
            np.array(data_dict["leadaggeqt"]),
            np.array(data_dict["leadaggsenst"]),
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
        leadaggeqt = np.empty(0)
        leadaggsenst = np.empty(0)
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
                lead_eqt,
                lead_senst,
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
            leadaggeqt = np.append(leadaggeqt, lead_eqt)
            leadaggsenst = np.append(leadaggsenst, lead_senst)
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
        self.vi.times_agg_eq = leadaggeqt
        self.vi.times_agg_sens = leadaggsenst
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

    def save_total_time_agg(self, path, n_ag, tsteps):
        # Followers' time
        fol_time = np.sum(np.max(self.vi.times_pnd + self.vi.times_sens, axis=0))
        # Leader's time
        lead_time = np.sum(
            self.hypergrad.times_log + self.vi.times_agg_eq + self.vi.times_agg_sens
        )
        # Total time
        tot_time = fol_time + lead_time

        # Data to be saved
        new_row = np.array([[n_ag, tsteps, tot_time]])
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
