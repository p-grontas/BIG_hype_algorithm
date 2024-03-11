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
import polyhedral_proj
import time
import cvxpy as cp
from scipy.sparse import csc_array


class dis_vi:
    def __init__(
        self,
        x,
        dims_y,
        m,
        grad,
        J1,
        J2,
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
        # Pseudo-gradient (PG) array of functions
        self.grad = grad
        # PG Jacobian w.r.t. x (array)
        self.J1 = J1
        # PG Jacobian w.r.t. y (array)
        self.J2 = J2
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
        # Error logging
        # Equilibrium Error Log
        self.eqlog = np.empty(0)
        # Sensitivity Error  Log
        self.senslog = np.empty(0)
        # Timing log of Projection and Differentiation (for each agent)
        self.times_pnd = np.empty((self.n_ag, 0))
        # Timing log of Sensitivity Learning (for each agent)
        self.times_sens = np.empty((self.n_ag, 0))

        # Save the identity matrix for fast computations
        # self.eye_n = np.eye(self.n)
        self.eye_n = csc_array(np.eye(self.n))

    def run_projection(self, n_iter=1, log_data=False, timing=False):
        y_new = self.y.copy()
        ag_times = np.zeros((self.n_ag, n_iter))

        for iter in range(n_iter):
            for ag in range(self.n_ag):
                # PG Step
                if timing:
                    startt = time.time()
                y_new[ag] = y_new[ag] - self.step * self.grad[ag](self.x, self.y)
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

        # Times Logging
        self.times_pnd = np.append(self.times_pnd, ag_times, axis=1)

    def run_projection_only(self, n_iter=1, log_data=False):
        y_new = self.y.copy()
        for iter in range(n_iter):
            for ag in range(self.n_ag):
                # PG Step
                y_new[ag] = y_new[ag] - self.step * self.grad[ag](self.x, self.y)
                if not (self.A_eq[ag] is None) or not (self.A_ineq[ag] is None):
                    y_new[ag] = self.projector[ag].project(y=y_new[ag], x=self.x)
            if log_data:
                self.y_log = np.append(
                    self.y_log,
                    np.expand_dims(np.concatenate(y_new, axis=0), axis=1),
                    axis=1,
                )
            # Update variable
            self.y = y_new.copy()

    def run_sensitivity(self, n_iter=1, log_data=False, timing=False):
        s_new = self.s.copy()
        s_stack = np.vstack(self.s)
        ag_times = np.zeros((self.n_ag, n_iter))

        for iter in range(n_iter):
            for ag in range(self.n_ag):
                # Timing start
                if timing:
                    startt = time.time()

                # Coordinates relevant to the current agent
                start = self.dims_y[:ag].sum()
                end = self.dims_y[: ag + 1].sum()
                # Jacobian of PG w.r.t. y
                nabla2h = self.eye_n[start:end] - self.step * self.J2[ag](
                    self.x, self.y
                )
                # Jacobian of PG w.r.t. x
                nabla1h = -self.step * self.J1[ag](self.x, self.y)
                if not (self.A_eq[ag] is None) or not (self.A_ineq[ag] is None):
                    # Jacobian of Projected PG (PPG) w.r.t. y
                    # nabla2h = np.matmul(self.jac_y[ag], nabla2h)
                    nabla2h = self.jac_y[ag] @ nabla2h
                    # Jacobian of Projected PG (PPG) w.r.t. x
                    nabla1h = np.matmul(self.jac_y[ag], nabla1h)
                    if not (self.jac_x[ag] is None):
                        nabla1h += self.jac_x[ag]
                # Sensitivity learning step
                # s_new[ag] = np.matmul(nabla2h, np.vstack(self.s)) + nabla1h
                # s_new[ag] = np.matmul(nabla2h, s_stack) + nabla1h
                s_new[ag] = nabla2h @ s_stack + nabla1h

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
            s_stack = np.vstack(self.s)

        # Times logging
        self.times_sens = np.append(self.times_sens, ag_times, axis=1)

    def run_proj_sens(self, n_iter=1, log_data=False, timing=False):
        # Run both projection and sensitivity iterations
        for iter in range(n_iter):
            self.run_projection(log_data=log_data, timing=timing)
            self.run_sensitivity(log_data=log_data, timing=timing)

    def clear_states(self, y0=None, s0=None):
        # Clear log and set y and s states
        if y0 is None:
            y0 = []
            for ag in range(self.n_ag):
                # Equilibrium
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
