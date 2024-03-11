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


class upper_opt:
    def __init__(
        self,
        m,
        J1,
        J2,
        gstep,
        rstep,
        A_eq=None,
        b_eq=None,
        A_ineq=None,
        b_ineq=None,
        qp_inds=None,
        simp_proj=None,
        box_proj=None,
    ):
        # Leader's Dimension
        self.m = m
        # Gradient of objective w.r.t. x
        self.J1 = J1
        # Gradient of objective w.r.t. y
        self.J2 = J2
        # Function of gradient steps, e.g., constant or vanishing
        self.gstep = gstep
        # Function of relaxation steps
        self.rstep = rstep
        # Constraints
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq

        # Define projection operator
        if (not (self.A_eq is None) and not (self.b_eq is None)) or (
            not (self.A_ineq is None) and not (self.b_ineq is None)
        ):
            # G_ineq and H_eq are always None: No parametric constraints for the leader
            self.projector = polyhedral_proj.poly_proj(
                A_eq=self.A_eq,
                b_eq=self.b_eq,
                A_ineq=self.A_ineq,
                b_ineq=self.b_ineq,
                G_ineq=None,
                H_eq=None,
            )

        # Simplex projection attributes
        # Indices of variables project via numerical QP solution
        self.qp_inds = qp_inds
        # Objects that perform projection onto simplices for the appropriate variables
        self.simp_proj = simp_proj
        # Objects that perform projection onto boxes for the appropriate variables
        self.box_proj = box_proj

        # Leader's Decision
        self.x = np.zeros(self.m)
        # Iteration counter, e.g., for step sizes
        self.iter = 0
        self.x_log = np.empty((self.m, 0))
        self.x_log = np.append(self.x_log, np.expand_dims(self.x, axis=1), axis=1)
        # Timing log
        self.times_log = np.empty(0)

    def run_step(self, y, s, timing=False):
        x_curr = self.x.copy()

        # Start timing
        if timing:
            startt = time.time()

        # Gradient Step
        x_gstep = x_curr - np.multiply(
            self.gstep(self.iter), self.J1(x_curr, y) + np.matmul(self.J2(x_curr, y), s)
        )
        # Projection Step
        x_proj = np.zeros(self.m)
        if (
            not (self.A_eq is None)
            or not (self.A_ineq is None)
            or not (self.simp_proj is None)
            or not (self.box_proj is None)
        ):
            if not (self.A_eq is None) or not (self.A_ineq is None):
                if self.qp_inds is None:
                    x_proj = self.projector.project(x_gstep, solver=cp.OSQP)
                else:
                    x_proj[self.qp_inds] = self.projector.project(
                        x_gstep[self.qp_inds], solver=cp.OSQP
                    )

            # Perform simplex projections
            if not (self.simp_proj is None):
                for smp in self.simp_proj:
                    x_proj[smp.var_inds] = smp.project(x_gstep)

            # Perform box projections
            if not (self.box_proj is None):
                for box in self.box_proj:
                    x_proj[box.var_inds] = box.project(x_gstep)
        else:
            x_proj = x_gstep

        # Relaxation Step
        x_new = (1.0 - self.rstep(self.iter)) * x_curr + self.rstep(self.iter) * x_proj
        self.iter += 1
        self.x = x_new

        # End timing
        if timing:
            endt = time.time()
            lead_time = endt - startt
            self.times_log = np.append(self.times_log, lead_time)

        self.x_log = np.append(self.x_log, np.expand_dims(x_new, axis=1), axis=1)

    def clear_states(self, x0=None):
        # Clear leader's decision and log
        if x0 is None:
            x0 = np.zeros(self.m)
        self.x = x0
        self.iter = 0
        self.x_log = np.empty((self.m, 0))
        self.x_log = np.append(self.x_log, np.expand_dims(self.x, axis=1), axis=1)

    def clear_log(self):
        self.iter = 0
        self.x_log = np.empty((self.m, 0))
        self.x_log = np.append(
            self.x_log, np.expand_dims(self.x.copy(), axis=1), axis=1
        )
        self.times_log = np.empty(0)
