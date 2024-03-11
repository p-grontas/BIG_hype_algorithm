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

import time
import cvxpy as cp
import distributed_vi
import numpy as np
import polyhedral_proj


class gen_agg_vi(distributed_vi.dis_vi):
    def __init__(
        self,
        x,
        dims_y,
        m,
        grad,
        J1,
        J2,
        J3,
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
        # Followers' dimension (array)
        self.dims_y = dims_y
        # Total followers' dimension
        self.n = self.dims_y.sum()
        # Number of agents
        self.n_ag = dims_y.size
        # Leader's dimension
        self.m = m
        # Pseudo-gradient (PG) array of functions
        # The PG is of the form F(x, y_i, \sum Q_i y_i)
        self.grad = grad
        # PG Jacobian w.r.t. x (array)
        self.J1 = J1
        # PG Jacobian w.r.t. y_i (array)
        self.J2 = J2
        # PG Jacobian w.r.t. the aggregate \sum y_i
        self.J3 = J3
        # Strong monotonicity constant
        self.mu = mu
        # Lipschitz continuity constant
        self.lf = lf
        # Constraints (list of arrays)
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
        # Jacobian of the projection w.r.t. 2nd and 1st argument, respectively.
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

    def run_projection(self, n_iter=1, log_data=False, timing=False):
        y_new = self.y.copy()
        ag_times = np.zeros((self.n_ag, n_iter))
        lead_times = np.zeros(n_iter)

        # Compute Equilibrium Aggregate
        y_agg = np.sum(y_new, axis=0)

        for iter in range(n_iter):
            # Compute Equilibrium Aggregate
            # Maybe I can use an aggregate with matrices using einsum
            # or 3d matrix-vector multiplication
            # y_agg = np.sum(y_new, axis=0)

            for ag in range(self.n_ag):
                # PG Step
                if timing:
                    startt = time.time()
                y_new[ag] = y_new[ag] - self.step * self.grad[ag](
                    self.x, self.y[ag], y_agg
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
            # Error Loggging
            y_all = np.concatenate(self.y, axis=0)
            y_all_new = np.concatenate(y_new, axis=0)
            eqerr = np.linalg.norm(y_all - y_all_new, ord=2)
            self.eqlog = np.append(self.eqlog, eqerr)
            # Enable when timing
            if timing:
                start_tmp = time.time()

            # Update equilibrium aggregate
            y_agg = np.sum(y_new, axis=0)
            if timing:
                end_tmp = time.time()
                lead_times[iter] = end_tmp - start_tmp
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
                y_new[ag] = y_new[ag] - self.step * self.grad[ag](
                    self.x, self.y[ag], y_agg
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
        # Compute sensitivity aggregate
        s_agg = np.sum(s_new, axis=0)

        for iter in range(n_iter):
            # Enable when timing
            if timing:
                start_tmp = time.time()
            # Compute sensitivity aggregate
            # s_agg = np.sum(s_new, axis=0)
            if timing:
                end_tmp = time.time()
                lead_times[iter] = end_tmp - start_tmp

            for ag in range(self.n_ag):
                # Timing start
                if timing:
                    startt = time.time()

                # Coordinates relevant to the current agent
                # start = self.dims_y[:ag].sum()
                # end = self.dims_y[: ag + 1].sum()

                # Jacobian w.r.t. x
                nabla1h = -self.step * self.J1[ag](self.x, self.y[ag], self.y_agg)
                # Jacobian w.r.t. y_i
                nabla2h = np.eye(self.dims_y[ag]) - self.step * self.J2[ag](
                    self.x, self.y[ag], self.y_agg
                )
                # Jacobian w.r.t. aggregate
                nabla3h = -self.step * self.J3[ag](self.x, self.y[ag], self.y_agg)
                # Jacobians w.r.t. the projection
                if not (self.A_eq[ag] is None) or not (self.A_ineq[ag] is None):
                    # Jacobian of Projected PG (PPG) w.r.t. x
                    nabla1h = self.jac_y[ag] @ nabla1h
                    if not (self.jac_x[ag] is None):
                        nabla1h += self.jac_x[ag]
                    # Jacobian of Projected PG (PPG) w.r.t. y_i
                    nabla2h = self.jac_y[ag] @ nabla2h
                    # Jacobian of Projected PG (PPG) w.r.t. aggregate
                    nabla3h = self.jac_y[ag] @ nabla3h

                # Sensitivity learning step
                s_new[ag] = nabla1h + nabla2h @ self.s[ag] + nabla3h @ s_agg

                # Timing complete
                if timing:
                    endt = time.time()
                    ag_times[ag, iter] = endt - startt
            if log_data:
                self.s_log = np.append(
                    self.s_log, np.expand_dims(np.vstack(s_new), axis=2), axis=2
                )

            # Update sensitivity aggregate
            s_agg = np.sum(s_new, axis=0)
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

    def clear_states(self, y0=None, s0=None):
        # Clear log and set y and s states
        if y0 is None:
            y0 = []
            for ag in range(self.n_ag):
                y0.append(np.zeros(self.dims_y[ag]))
        if s0 is None:
            s0 = []
            for ag in range(self.n_ag):
                # Sensitivity
                s0.append(np.zeros((self.dims_y[ag], self.m)))

        self.s = s0
        self.y = y0
        self.y_log = np.empty((self.n, 0))
        y_all = np.concatenate(self.y, axis=0)
        self.y_log = np.append(self.y_log, np.expand_dims(y_all, axis=1), axis=1)
        self.s_log = np.empty((self.n, self.m, 0))
        s_all = np.vstack(self.s)
        self.s_log = np.append(self.s_log, np.expand_dims(s_all, axis=2), axis=2)
        self.eqlog = np.empty(0)
        self.senslog = np.empty(0)
        self.times_pnd = np.empty((self.n_ag, 0))
        self.times_sens = np.empty((self.n_ag, 0))

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
