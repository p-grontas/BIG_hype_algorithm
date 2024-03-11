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

import distributed_hg
import numpy as np
import distributed_vi
import upper
from tqdm import tqdm
import matplotlib.pyplot as plt


class general_hypergradient(distributed_hg.distributed_hypergradient):
    def __init__(
        self,
        dims_y,
        dim_x,
        pgrad,
        pjacob1,
        pjacob2,
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
        step_guess=None,
    ):
        # Lower-level problem functions
        # Pseudo-gradient for each agent
        self.pgrad = pgrad
        # Partial Jacobian w.r.t. x
        self.pjacob1 = pjacob1
        # Partial Jacobian w.r.t. y
        self.pjacob2 = pjacob2

        # Variable Dimensions
        self.dims_y = dims_y
        # Number of Agents
        self.n_ag = dims_y.size
        # Total followers' dimension
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

        # Define Lower-level VI solver
        x0 = np.zeros(self.dim_x)
        self.vi = distributed_vi.dis_vi(
            x=x0,
            dims_y=self.dims_y,
            m=self.dim_x,
            grad=self.pgrad,
            J1=self.pjacob1,
            J2=self.pjacob2,
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

        # Tolerance sequence
        self.tol_y = tol_y
        self.tol_s = tol_s

        # General Setup Sensitivity Error Log (this is computed differently)
        self.gen_senslog = np.empty(0)

        # Exact solution attributes
        self.exact_x = None
        self.exact_y = None
        self.exact_val = None

    def run_general(
        self,
        n_iters=1,
        timing=False,
        log_data=False,
    ):
        for k in tqdm(range(n_iters)):
            # Change the VI parameter x
            self.vi.x = self.hypergrad.x.copy()

            converged = False
            # Number of inner iterations
            n_inner = 0
            # Number of sensitivity iterations
            n_sens = 0
            # Sensitivity Error Perturbed
            sens_err_pert = 0
            # Repeat inner loop until tolerance is satisfied
            while not (converged):
                # Run equilibrium learning and update counter
                self.vi.run_projection(n_iter=1, log_data=log_data, timing=timing)
                n_inner += 1

                # Get Equilibrium Error
                eq_error = self.vi.eqlog[-1].copy()
                # Original Equilibrium Error
                eq_error_orig = eq_error.copy()
                # Normalize by the dimension
                eq_error /= self.dim_y

                # Check if the equilibrium is close enough
                if eq_error < self.tol_y(self.up_iter + self.up_off):
                    self.vi.run_sensitivity(n_iter=1, log_data=log_data, timing=timing)
                    n_sens += 1

                    # Update the perturbed error
                    sens_err_pert = sens_err_pert * self.eta + eq_error_orig
                    # Check the errors
                    converged = (
                        sens_err_pert < self.tol_s(self.up_iter + self.up_off)
                    ) and (self.eta**n_sens < self.tol_s(self.up_iter + self.up_off))

                # Log the perturbed sensitivity errors
                if not (sens_err_pert == 0):
                    self.gen_senslog = np.append(
                        self.gen_senslog, np.maximum(sens_err_pert, self.eta**n_sens)
                    )
                else:
                    self.gen_senslog = np.append(self.gen_senslog, np.NaN)

            # Run Hypergradient step
            y_all = np.concatenate(self.vi.y, axis=0)
            s_all = np.vstack(self.vi.s)
            self.hypergrad.run_step(y=y_all, s=s_all, timing=timing)

            # Update counters
            self.up_iter += 1
            self.low_iter += n_inner
            self.low2up = np.append(self.low2up, np.array([self.low_iter]), axis=0)

    def plot_perturbed_sensitivity_error(self):
        plt.semilogy(
            self.gen_senslog,
            label="Inner Loop",
            linewidth=1,
            color="blue",
            marker="o",
            markersize=4,
        )
        plt.semilogy(
            self.low2up[1:] - 1,
            self.gen_senslog[self.low2up[1:] - 1],
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
        plt.ylabel("Approximate Sensitivity Error")
