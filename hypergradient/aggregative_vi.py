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
import distributed_vi
import polyhedral_proj
import time
import cvxpy as cp
from scipy.sparse import csc_array


class agg_vi(distributed_vi.dis_vi):
    def __init__(
        self,
        x,
        dims_y,
        m,
        Q,
        Qreg,
        C,
        c,
        mu,
        lf,
        step=None,
        A_eq=None,
        b_eq=None,
        H_eq=None,
        A_ineq=None,
        b_ineq=None,
        G_ineq=None,
    ):
        # Initialize class attributes
        # Leader's decision
        self.x = x
        # Followers' dimensions (array)
        self.dims_y = dims_y
        # Total followers' dimension
        self.n = self.dims_y.sum()
        # Number of agents
        self.n_ag = dims_y.size
        # Leader's dimension
        self.m = m
        # Matrix of agent interactions
        self.Q = Q
        # Matrix of agent regularizatoin
        self.Qreg = Qreg
        # Matrix of leader's effect
        self.C = C
        # Vector of constant PG term
        self.c = c
        # Strong monotonicity constant
        self.mu = mu
        # Lipschitz continuity constant
        self.lf = lf
        # Constraints (lists of arrays)
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.H_eq = H_eq
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.G_ineq = G_ineq

        # Set step size if not provided
        if step is None:
            self.step = mu / np.square(lf)
        else:
            self.step = step

        self.projector = []
        self.jac_y = []
        self.jac_x = []
        # Define projection operators
        for ag in range(self.n_ag):
            if (not (self.A_eq[ag] is None) and not (self.b_eq[ag] is None)) or (
                not (self.A_ineq[ag] is None) and not (self.b_ineq[ag] is None)
            ):
                self.projector.append(
                    polyhedral_proj.poly_proj(
                        A_eq=self.A_eq[ag],
                        b_eq=self.b_eq[ag],
                        A_ineq=self.A_ineq[ag],
                        b_ineq=self.b_ineq[ag],
                        G_ineq=self.G_ineq[ag],
                        H_eq=self.H_eq[ag],
                    )
                )
                self.jac_y.append(np.zeros((self.dims_y[ag], self.dims_y[ag])))
                self.jac_x.append(None)

        # Initialize Variables
        self.s = []
        self.s_log = np.empty((self.n, self.m, 0))
        self.y = []
        self.y_log = np.empty((self.n, 0))
        for ag in range(self.n_ag):
            # Sensitivity
            self.s.append(np.zeros((self.dims_y[ag], self.m)))
            # Follower's decision
            self.y.append(np.zeros(self.dims_y[ag]))
        y_all = np.concatenate(self.y, axis=0)
        self.y_log = np.append(self.y_log, np.expand_dims(y_all, axis=1), axis=1)
        s_all = np.vstack(self.s)
        self.s_log = np.append(self.s_log, np.expand_dims(s_all, axis=2), axis=2)
        # Placeholder for the aggregate y and s
        self.y_agg = np.zeros(self.dims_y[0])
        self.s_agg = np.zeros((self.dims_y[0], self.m))
        # Error logging
        # Equilibrium Error Log
        self.eqlog = np.empty(0)
        # Sensitivity Error  Log
        self.senslog = np.empty(0)
        # Timing log of Projection and Differentiation (for each agent)
        self.times_pnd = np.empty((self.n_ag, 0))
        # Timing log of Sensitivity Learning (for each agent)
        self.times_sens = np.empty((self.n_ag, 0))
        # Timing of aggregating equilibrium
        self.times_agg_eq = np.empty(0)
        # Timing of aggregating sensitivies
        self.times_agg_sens = np.empty(0)

        # Save the identity matrix for fast computations
        # self.eye_n = np.eye(self.n)
        self.eye_n = csc_array(np.eye(self.n))

    def run_projection(self, n_iter=1, log_data=False, timing=False):
        y_new = self.y.copy()
        ag_times = np.zeros((self.n_ag, n_iter))
        lead_times = np.zeros(n_iter)

        for iter in range(n_iter):
            # Enable when timing
            if timing:
                start_tmp = time.time()
            # Compute equilibrium aggregate
            y_agg = np.sum(y_new, axis=0)
            if timing:
                end_tmp = time.time()
                lead_times[iter] = end_tmp - start_tmp

            for ag in range(self.n_ag):
                # PG Step
                if timing:
                    startt = time.time()
                y_new[ag] = y_new[ag] - self.step * (
                    self.Q @ (y_agg + y_new[ag])
                    + self.Qreg @ y_new[ag]
                    + self.c
                    + self.C @ self.x
                )

                if not (self.A_eq[ag] is None) or not (self.A_ineq[ag] is None):
                    y_new[ag], jac_y, jac_x = self.projector[ag].proj_and_diff(
                        y=y_new[ag], x=self.x, solver=cp.OSQP
                    )
                    self.jac_y[ag] = jac_y
                    self.jac_x[ag] = jac_x

                if timing:
                    endt = time.time()
                    ag_times[ag, iter] = endt - startt
            # Data Logging
            if log_data:
                self.y_log = np.append(
                    self.y_log,
                    np.expand_dims(np.concatenate(y_new, axis=0), axis=1),
                    axis=1,
                )
            # Error logging
            y_all = np.concatenate(self.y, axis=0)
            y_all_new = np.concatenate(y_new, axis=0)
            eqerr = np.linalg.norm(y_all - y_all_new, ord=2)
            self.eqlog = np.append(self.eqlog, eqerr)
            # Update variable
            self.y = y_new.copy()

        # Save aggregate y
        self.y_agg = y_agg.copy()
        # Times Logging
        self.times_pnd = np.append(self.times_pnd, ag_times, axis=1)
        self.times_agg_eq = np.append(self.times_agg_eq, lead_times, axis=0)

    def run_projection_only(self, n_iter=1, log_data=False):
        y_new = self.y.copy()
        for iter in range(n_iter):
            y_agg = np.sum(y_new, axis=0)
            for ag in range(self.n_ag):
                # PG Step
                y_new[ag] = y_new[ag] - self.step * (
                    self.Q @ (y_agg + y_new[ag])
                    + self.Qreg @ y_new[ag]
                    + self.c
                    + self.C @ self.x
                )

                if not (self.A_eq[ag] is None) or not (self.A_ineq[ag] is None):
                    y_new[ag] = self.projector[ag].project(y=y_new[ag], x=self.x)
            if log_data:
                self.y_log = np.append(
                    self.y_log,
                    np.expand_dims(np.concatenate(y_new, axis=0), axis=1),
                    axis=1,
                )
            self.y = y_new

    def run_sensitivity(self, n_iter=1, log_data=False, timing=False):
        s_new = self.s.copy()
        ag_times = np.zeros((self.n_ag, n_iter))
        lead_times = np.zeros(n_iter)

        for iter in range(n_iter):
            # Enable when timing
            if timing:
                start_tmp = time.time()
            # Compute sensitivity aggregate
            s_agg = np.sum(s_new, axis=0)
            qs_agg = self.step * self.Q @ s_agg
            if timing:
                end_tmp = time.time()
                lead_times[iter] = end_tmp - start_tmp

            for ag in range(self.n_ag):
                # Timing start
                if timing:
                    startt = time.time()

                # Eye minus nabla2h times s for each agent
                eyemnabla2hts = (
                    np.eye(self.dims_y[ag]) - self.step * self.Qreg - self.step * self.Q
                ) @ s_new[ag] - qs_agg
                # Jacobian of PG w.r.t. x
                nabla1h = -self.step * self.C
                if not (self.A_eq[ag] is None) or not (self.A_ineq[ag] is None):
                    # Jacobian of Projected PG (PPG) w.r.t. y
                    # nabla2h = np.matmul(self.jac_y[ag], nabla2h)
                    eyemnabla2hts = self.jac_y[ag] @ eyemnabla2hts
                    # Jacobian of Projected PG (PPG) w.r.t. x
                    nabla1h = np.matmul(self.jac_y[ag], nabla1h)
                    if not (self.jac_x[ag] is None):
                        nabla1h += self.jac_x[ag]
                # Sensitivity learning step
                s_new[ag] = eyemnabla2hts + nabla1h

                # Timing Complete
                if timing:
                    endt = time.time()
                    ag_times[ag, iter] = endt - startt
            if log_data:
                self.s_log = np.append(
                    self.s_log, np.expand_dims(np.vstack(s_new), axis=2), axis=2
                )

            s_all = np.vstack(self.s)
            s_all_new = np.vstack(s_new)
            senserr = np.linalg.norm(s_all - s_all_new, ord="fro")
            self.senslog = np.append(self.senslog, senserr)
            self.s = s_new.copy()

        # Save aggregate s
        self.s_agg = s_agg.copy()

        # Times logging
        self.times_sens = np.append(self.times_sens, ag_times, axis=1)
        self.times_agg_sens = np.append(self.times_agg_sens, lead_times, axis=0)

    def clear_log(self):
        self.y_log = np.empty((self.n, 0))
        y_all = np.concatenate(self.y, axis=0)
        self.y_log = np.append(self.y_log, np.expand_dims(y_all.copy(), axis=1), axis=1)
        self.s_log = np.empty((self.n, self.m, 0))
        s_all = np.vstack(self.s)
        self.s_log = np.append(self.s_log, np.expand_dims(s_all.copy(), axis=2), axis=2)
        self.eqlog = np.empty(0)
        self.senslog = np.empty(0)
        self.times_pnd = np.empty((self.n_ag, 0))
        self.times_sens = np.empty((self.n_ag, 0))
        self.times_agg_eq = np.empty(0)
        self.times_agg_sens = np.empty(0)
