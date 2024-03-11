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

# import torch
import polyhedral_proj
import time


class lower_vi:
    def __init__(
        self,
        x,
        n,
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
        # Followers' dimension
        self.n = n
        # Leader's dimension
        self.m = m
        # Pseudo-gradient (PG) function
        self.grad = grad
        # PG Jacobian w.r.t. x
        self.J1 = J1
        # PG Jacobian w.r.t. y
        self.J2 = J2
        # Strong monotonicity constant
        self.mu = mu
        # Lipschitz continuity constant
        self.lf = lf
        # Constraints
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

        # Define projection operator
        if (not (self.A_eq is None) and not (self.b_eq is None)) or (
            not (self.A_ineq is None) and not (self.b_ineq is None)
        ):
            self.projector = polyhedral_proj.poly_proj(
                A_eq=self.A_eq,
                b_eq=self.b_eq,
                A_ineq=self.A_ineq,
                b_ineq=self.b_ineq,
                G_ineq=self.G_ineq,
                H_eq=self.H_eq,
            )
            self.jac_y = np.zeros((self.n, self.n))
            self.jac_x = None

        # Initialize variables
        # Sensitivity
        self.s = np.zeros((self.n, self.m))
        # Followers' decision
        self.y = np.zeros(self.n)
        self.y_log = np.empty((self.n, 0))
        self.y_log = np.append(self.y_log, np.expand_dims(self.y, axis=1), axis=1)
        self.s_log = np.empty((self.n, self.m, 0))
        self.s_log = np.append(self.s_log, np.expand_dims(self.s, axis=2), axis=2)
        # Error logging
        # Equilibrium Error Log
        self.eqlog = np.empty(0)
        # Sensitivity Error Log
        self.senslog = np.empty(0)
        # Timing Log (for all agents)
        self.times_log = np.empty(0)

    def run_projection(self, n_iter=1, log_data=False, timing=False):
        y_new = self.y.copy()
        times = np.zeros(n_iter)

        for iter in range(n_iter):
            # Start timer
            if timing:
                startt = time.time()
            # PG step
            y_new = y_new - self.step * self.grad(self.x, y_new)
            if not (self.A_eq is None) or not (self.A_ineq is None):
                y_new, jac_y, jac_x = self.projector.proj_and_diff(y=y_new, x=self.x)
                self.jac_y = jac_y
                self.jac_x = jac_x

            # Store time
            if timing:
                endt = time.time()
                times[iter] = endt - startt

            # Log y
            if log_data:
                self.y_log = np.append(
                    self.y_log, np.expand_dims(y_new, axis=1), axis=1
                )

            # Log error
            eqerr = np.linalg.norm(y_new - self.y, ord=2)
            self.eqlog = np.append(self.eqlog, eqerr)

            # Update variable
            self.y = y_new.copy()

        # Times Logging
        self.times_log = np.append(self.times_log, times)

    def run_projection_only(self, n_iter=1, log_data=False):
        y_new = self.y.copy()
        for iter in range(n_iter):
            # PG step
            y_new = y_new - self.step * self.grad(self.x, y_new)
            if not (self.A_eq is None) or not (self.A_ineq is None):
                y_new = self.projector.project(y=y_new, x=self.x)
            if log_data:
                self.y_log = np.append(
                    self.y_log, np.expand_dims(y_new, axis=1), axis=1
                )
        self.y = y_new.copy()

    def run_sensitivity(self, n_iter=1, log_data=False):
        s_new = self.s.copy()
        for iter in range(n_iter):
            # Jacobian of PG (PPG) w.r.t. y
            nabla2h = np.eye(self.n) - self.step * self.J2(self.x, self.y)
            # Jacobian of PG w.r.t. x
            nabla1h = -self.step * self.J1(self.x, self.y)
            if not (self.A_eq is None) or not (self.A_ineq is None):
                # Jacobian of Projected PG (PPG) w.r.t. y
                nabla2h = np.matmul(self.jac_y, nabla2h)
                # Jacobian of Projected PG (PPG) w.r.t. x
                nabla1h = np.matmul(self.jac_y, nabla1h)
                # If the constraints are parametric add the dependence of the projection on x
                if not (self.jac_x is None):
                    nabla1h += self.jac_x
            # Sensitivity learning setp
            s_new = np.matmul(nabla2h, s_new) + nabla1h
            if log_data:
                self.s_log = np.append(
                    self.s_log, np.expand_dims(s_new, axis=2), axis=2
                )

            # Error logging
            senserr = np.linalg.norm(s_new - self.s, ord=2)
            self.senslog = np.append(self.senslog, senserr)

            # Update variable
            self.s = s_new.copy()

    def run_proj_sens(self, n_iter=1, log_data=False, timing=False):
        # Run both projection and sensitivity iterations
        for iter in range(n_iter):
            self.run_projection(log_data=log_data, timing=timing)
            self.run_sensitivity(log_data=log_data)

    def clear_states(self, y0=None, s0=None):
        # Clear log and set y and s states
        if y0 is None:
            y0 = np.zeros(self.n)
        if s0 is None:
            s0 = np.zeros((self.n, self.m))

        self.s = s0
        self.y = y0
        self.y_log = np.empty((self.n, 0))
        self.y_log = np.append(self.y_log, np.expand_dims(self.y, axis=1), axis=1)
        self.s_log = np.empty((self.n, self.m, 0))
        self.s_log = np.append(self.s_log, np.expand_dims(self.s, axis=2), axis=2)
        self.eqlog = np.empty(0)
        self.senslog = np.empty(0)
        self.times_log = np.empty(0)

    def clear_log(self):
        self.y_log = np.empty((self.n, 0))
        self.y_log = np.append(self.y_log, np.expand_dims(self.y, axis=1), axis=1)
        self.s_log = np.empty((self.n, self.m, 0))
        self.s_log = np.append(self.s_log, np.expand_dims(self.s, axis=2), axis=2)
        self.eqlog = np.empty(0)
        self.senslog = np.empty(0)
        self.times_log = np.empty(0)
