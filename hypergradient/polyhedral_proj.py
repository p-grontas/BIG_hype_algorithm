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

# from scipy import sparse
import scipy


class poly_proj:
    def __init__(self, A_eq, b_eq, A_ineq, b_ineq, H_eq, G_ineq):
        # Initialize Matrices for Projection
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.H_eq = H_eq
        self.G_ineq = G_ineq

        # Dimension of projected
        if not (self.A_eq is None):
            self.n = self.A_eq.shape[1]
        else:
            self.n = self.A_ineq.shape[1]

        # Projection Variable
        self.z = cp.Variable(self.n)
        # Projected Point
        self.y = cp.Parameter(self.n)
        # If relevant define parametric constraints
        if not (self.H_eq is None):
            # Dimension of leader's variable
            self.m = self.H_eq.shape[1]
            # Leader's variable
            self.x = cp.Parameter(self.m)
        elif not (self.G_ineq is None):
            self.m = self.G_ineq.shape[1]
            self.x = cp.Parameter(self.m)
        else:
            self.m = None

        # Define Projection Solver
        self.objective = cp.Minimize(cp.sum_squares(self.z - self.y))
        self.constraints = []
        if not (self.A_ineq is None):
            if self.G_ineq is None:
                self.constraints += [self.A_ineq @ self.z <= self.b_ineq]
            else:
                self.constraints += [
                    self.A_ineq @ self.z <= self.b_ineq + self.G_ineq @ self.x
                ]

        if not (self.A_eq is None):
            if self.H_eq is None:
                self.constraints += [self.A_eq @ self.z == self.b_eq]
            else:
                self.constraints += [
                    self.A_eq @ self.z == self.b_eq + self.H_eq @ self.x
                ]

        self.prob = cp.Problem(self.objective, self.constraints)
        assert self.prob.is_dpp()

        # Use dual variables to check if the same constraints are active.
        # In that case, we do not need to compute the Jacobian of the projection:
        # it is the same as the previous iteration.
        if not (self.A_ineq is None):
            # Dual Variables
            self.duals = None
            # Tolerance for considering a constraint as active
            self.dual_tol = 1e-1
        # Jacobian w.r.t. y
        self.jac_y = None
        # Jacobian w.r.t. x
        self.jac_x = None

        # Implementation of sparse projection. Did not work nicely.
        # if sparse.issparse(self.A_eq) or sparse.issparse(self.A_ineq):
        #     self.compute_jacobian = self.compute_jacobian_sparse
        # else:
        #     self.compute_jacobian = self.compute_jacobian_dense

    def project(self, y, x=None, solver=cp.OSQP):
        # Set the value of the parameter
        self.y.value = y
        # Check if there are parametric constraints and x is provided
        if not (x is None) and not (self.m is None):
            self.x.value = x
        # Solve projection
        self.prob.solve(
            solver=solver, warm_start=True, polish=True, eps_abs=1e-4, eps_rel=1e-4
        )

        return self.z.value

    def proj_and_diff(self, y, x=None, solver=cp.OSQP):
        # Set the value of the parameter
        self.y.value = y
        # Check if there are parametric constraints and x is provided
        if not (x is None) and not (self.m is None):
            self.x.value = x
        # Solve projection
        self.prob.solve(
            solver=solver, warm_start=True, polish=True, eps_abs=1e-4, eps_rel=1e-4
        )

        if not (self.A_ineq is None):
            # Current active (inequality) constraints
            duals_curr = self.constraints[0].dual_value > self.dual_tol

            # If the active constraints changed recompute the Jacobian
            if not ((self.duals == duals_curr).all()):
                # print("Computing jac")
                (jac_x, jac_y) = self.compute_jacobian()
                self.jac_y = jac_y
                if not (self.m is None):
                    self.jac_x = jac_x
            # Update dual variables
            self.duals = duals_curr
        else:
            # If we have only equality constraints we only
            # need to compute the Jacobian once
            if self.jac_y is None:
                # print("Computing jac")
                (jac_x, jac_y) = self.compute_jacobian()
                self.jac_y = jac_y
                if not (self.m is None):
                    self.jac_x = jac_x

        return (self.z.value, self.jac_y, self.jac_x)

    def compute_jacobian(self):
        # Build KKT matrix and RHS
        if self.A_ineq is None:
            n_eq = self.A_eq.shape[0]
            kkt_mat = np.block(
                [
                    [np.eye(self.n), self.A_eq.transpose()],
                    [self.A_eq, np.zeros((n_eq, n_eq))],
                ]
            )
            rhs = np.block([[np.eye(self.n)], [np.zeros((n_eq, self.n))]])
            if not (self.H_eq is None):
                rhs_tmp = np.block([[np.zeros((self.n, self.m))], [self.H_eq]])
                rhs = np.block([rhs, rhs_tmp])

        elif self.A_eq is None:
            active = self.constraints[0].dual_value > self.dual_tol
            n_active = active.sum()
            kkt_mat = np.block(
                [
                    [np.eye(self.n), self.A_ineq[active].transpose()],
                    [self.A_ineq[active], np.zeros((n_active, n_active))],
                ]
            )
            rhs = np.block([[np.eye(self.n)], [np.zeros((n_active, self.n))]])
            if not (self.G_ineq is None):
                rhs_tmp = np.block(
                    [[np.zeros((self.n, self.m))], [self.G_ineq[active]]]
                )
                rhs = np.block([rhs, rhs_tmp])
        else:
            n_eq = self.A_eq.shape[0]
            active = self.constraints[0].dual_value > self.dual_tol
            n_active = active.sum()
            kkt_mat = np.block(
                [
                    [
                        np.eye(self.n),
                        self.A_ineq[active].transpose(),
                        self.A_eq.transpose(),
                    ],
                    [self.A_ineq[active], np.zeros((n_active, n_active + n_eq))],
                    [self.A_eq, np.zeros((n_eq, n_active + n_eq))],
                ]
            )
            rhs = np.block(
                [
                    [np.eye(self.n)],
                    [np.zeros((n_active, self.n))],
                    [np.zeros((n_eq, self.n))],
                ]
            )
            if (not (self.G_ineq is None)) or (not (self.H_eq is None)):
                rhs_tmp = np.zeros((self.n + n_active + n_eq, self.m))
                if not (self.H_eq is None):
                    rhs_tmp += np.block(
                        [
                            [np.zeros((self.n, self.m))],
                            [np.zeros((n_active, self.m))],
                            [self.H_eq],
                        ]
                    )
                if not (self.G_ineq is None):
                    rhs_tmp += np.block(
                        [
                            [np.zeros((self.n, self.m))],
                            [self.G_ineq[active]],
                            [np.zeros((n_eq, self.m))],
                        ]
                    )

                rhs = np.block([rhs, rhs_tmp])
        # Compute Jacobian
        try:
            # Compute using pseudo-inverse
            jac = [scipy.linalg.pinv(kkt_mat) @ rhs]
        except:
            # If the pseudo-inverse fails (happened for sporadically for large problems)
            # use least-squares method
            jac = scipy.linalg.lstsq(kkt_mat, rhs)
        jac_y = jac[0][: self.n, : self.n]
        if (not (self.G_ineq is None)) or (not (self.H_eq is None)):
            jac_x = jac[0][: self.n, self.n :]
        else:
            jac_x = None
        return (jac_x, jac_y)

    # def compute_jacobian_sparse(self):
    #     if self.A_ineq is None:
    #         n_eq = self.A_eq.shape[0]
    #         kkt_mat = sparse.bmat(
    #             [
    #                 [sparse.eye(self.n), self.A_eq.transpose()],
    #                 [self.A_eq, np.zeros((n_eq, n_eq))],
    #             ],
    #             format="csr",
    #         )
    #         rhs = sparse.bmat(
    #             [[sparse.eye(self.n)], [np.zeros((n_eq, self.n))]], format="csr"
    #         )
    #         if not (self.H_eq is None):
    #             rhs_tmp = sparse.bmat(
    #                 [[np.zeros((self.n, self.m))], [self.H_eq]], format="csr"
    #             )
    #             rhs = sparse.bmat([rhs, rhs_tmp], format="csr")

    #     elif self.A_eq is None:
    #         active = self.constraints[0].dual_value > self.dual_tol
    #         n_active = active.sum()
    #         kkt_mat = sparse.bmat(
    #             [
    #                 [sparse.eye(self.n), self.A_ineq[active].transpose()],
    #                 [self.A_ineq[active], np.zeros((n_active, n_active))],
    #             ],
    #             format="csr",
    #         )
    #         rhs = sparse.bmat(
    #             [[sparse.eye(self.n)], [np.zeros((n_active, self.n))]], format="csr"
    #         )
    #         if not (self.G_ineq is None):
    #             rhs_tmp = sparse.bmat(
    #                 [[np.zeros((self.n, self.m))], [self.G_ineq[active]]], format="csr"
    #             )
    #             rhs = sparse.bmat([rhs, rhs_tmp], format="csr")

    #     else:
    #         n_eq = self.A_eq.shape[0]
    #         active = self.constraints[0].dual_value > self.dual_tol
    #         n_active = active.sum()
    #         kkt_mat = sparse.bmat(
    #             [
    #                 [
    #                     sparse.eye(self.n),
    #                     self.A_ineq[active].transpose(),
    #                     self.A_eq.transpose(),
    #                 ],
    #                 [
    #                     self.A_ineq[active],
    #                     np.zeros((n_active, n_active)),
    #                     np.zeros((n_active, n_eq)),
    #                 ],
    #                 [self.A_eq, np.zeros((n_eq, n_active)), np.zeros((n_eq, n_eq))],
    #             ],
    #             format="csr",
    #         )
    #         rhs = sparse.bmat(
    #             [
    #                 [sparse.eye(self.n)],
    #                 [np.zeros((n_active, self.n))],
    #                 [np.zeros((n_eq, self.n))],
    #             ],
    #             format="csr",
    #         )
    #         if (not (self.G_ineq is None)) or (not (self.H_eq is None)):
    #             rhs_tmp = np.zeros((self.n + n_active + n_eq, self.m))
    #             if not (self.H_eq is None):
    #                 rhs_tmp += sparse.bmat(
    #                     [
    #                         [np.zeros((self.n, self.m))],
    #                         [np.zeros((n_active, self.m))],
    #                         [self.H_eq],
    #                     ],
    #                     format="csr",
    #                 )
    #             if not (self.G_ineq is None):
    #                 rhs_tmp += sparse.bmat(
    #                     [
    #                         [np.zeros((self.n, self.m))],
    #                         [self.G_ineq[active]],
    #                         [np.zeros((n_eq, self.m))],
    #                     ],
    #                     format="csr",
    #                 )

    #             rhs = sparse.bmat([rhs, rhs_tmp], format="csr")
    #     # Compute Jacobian
    #     rhs = rhs.todense()
    #     jac_y = np.zeros((self.n, self.n))
    #     for ii in range(self.n):
    #         new_var = sparse.linalg.gmres(kkt_mat, rhs[:, ii])
    #         jac_y[:, ii] = new_var[0][: self.n]
    #     # jac_y = jac[: self.n, : self.n]
    #     if (not (self.G_ineq is None)) or (not (self.H_eq is None)):
    #         jac_x = np.zeros((self.n, self.m))
    #         for ii in range(self.m):
    #             jac_x[:, ii] = sparse.linalg.gmres(kkt_mat, rhs[:, self.n + ii])
    #     else:
    #         jac_x = None
    #     return (jac_x, jac_y)


class simplex_proj:
    def __init__(self, var_inds, lb=0, sum_to=1, method="active"):
        # Lower bound on simplex variables
        self.lb = lb
        # The number the simplex variables sum to
        self.sum_to = sum_to
        # The method used for projection
        self.method = method
        # The variables of the provided vector to be projected
        self.var_inds = var_inds
        # Dimension of simplex
        self.dim = var_inds.sum()
        # New sum_to when using change of variables for lower bound
        self.sum_to_ch = sum_to - self.lb * self.dim

        if self.method == "qp":
            ones = np.ones(self.dim)
            self.cp_proj = cp.Variable(self.dim)
            self.y_proj = cp.Parameter(self.dim)
            self.objective = cp.Minimize(cp.sum_squares(self.cp_proj - self.y_proj))
            self.constraints = [
                self.cp_proj >= self.lb,
                ones @ self.cp_proj == self.sum_to,
            ]
            self.prob = cp.Problem(
                objective=self.objective, constraints=self.constraints
            )
            self.project = self.qp_project
        elif self.method == "sort":
            self.project = self.sort_project
        elif self.method == "active":
            self.project = self.active_project
        elif self.method == "laurent":
            self.project = self.laurent_project
        else:
            raise Exception("Invalid simplex projection method.")

    def qp_project(self, y):
        self.y_proj.value = y[self.var_inds].copy()
        self.prob.solve(
            solver=cp.OSQP, warm_start=True, polish=True, eps_abs=1e-7, eps_rel=1e-7
        )

        return self.cp_proj.value

    # Projection methods based on:
    # Fast Projection onto the Simplex and the l_1 Ball
    # by Lauren Condat
    def sort_project(self, y):
        to_proj = y[self.var_inds].copy() - self.lb
        sorted = -np.sort(-to_proj, kind="quicksort")
        tmp = (np.cumsum(sorted) - self.sum_to_ch) / (np.arange(self.dim) + 1)
        K = np.max(np.where(tmp < sorted))
        tau_sort = tmp[K]
        sort_proj = np.maximum(to_proj - tau_sort, 0)

        return sort_proj + self.lb

    def active_project(self, y):
        to_proj = y[self.var_inds].copy() - self.lb
        v = to_proj.copy()
        rho = (v.sum() - self.sum_to_ch) / v.size
        converged = False
        while not (converged):
            v_new = v[v > rho]
            rho = (v_new.sum() - self.sum_to_ch) / v_new.size
            if v.size == v_new.size:
                converged = True
            else:
                v = v_new.copy()

        tau_active = rho
        active_proj = np.maximum(to_proj - tau_active, 0)

        return active_proj + self.lb

    def laurent_project(self, y):
        to_proj = y[self.var_inds].copy() - self.lb
        v = to_proj[0]
        v_til = np.empty(0)
        rho = to_proj[0] - self.sum_to_ch
        for ii in range(1, self.dim):
            if to_proj[ii] > rho:
                rho += (to_proj[ii] - rho) / (v.size + 1)
                if rho > to_proj[ii] - self.sum_to_ch:
                    v = np.append(v, to_proj[ii])
                else:
                    v_til = np.append(v_til, v.copy())
                    v = to_proj[ii]
                    rho = to_proj[ii] - self.sum_to_ch
        if v_til.size > 0:
            for y in v_til:
                if y > rho:
                    v = np.append(v, y)
                    rho += (y - rho) / v.size

        converged = False
        n_v = v.size
        while not (converged):
            n_v_new = n_v
            idx_del = []
            for idx, y in enumerate(v):
                if y <= rho:
                    idx_del.append(idx)
                    n_v_new -= 1
                    rho += (rho - y) / n_v_new
            v = np.delete(v, idx_del)
            if n_v_new == n_v:
                converged = True
            else:
                n_v = n_v_new

        tau_laur = rho
        laurent_proj = np.maximum(to_proj - tau_laur, 0)

        return laurent_proj + self.lb


class box_project:
    def __init__(self, var_inds, lb=-np.inf, ub=np.inf, init_qp=False):
        # The components of the provided vector to be projected
        self.var_inds = var_inds
        # Lower bounds on the box
        # If it is minus infinity then we have no lower bound
        self.lb = lb
        # Upper bounds on the box
        self.ub = ub
        # Dimension of the box
        self.dim = var_inds.sum()

        if init_qp:
            self.cp_proj = cp.Variable(self.dim)
            self.y_proj = cp.Parameter(self.dim)
            self.objective = cp.Minimize(cp.sum_squares(self.cp_proj - self.y_proj))
            A = np.block([[np.eye(self.dim)], [-np.eye(self.dim)]])
            b = np.concatenate((self.ub, self.lb))
            self.constraints = [A @ self.cp_proj <= b]
            self.prob = cp.Problem(
                objective=self.objective, constraints=self.constraints
            )

    def qp_project(self, y):
        self.y_proj.value = y[self.var_inds].copy()
        self.prob.solve(
            solver=cp.OSQP, warm_start=True, polish=True, eps_abs=1e-7, eps_rel=1e-7
        )

        return self.cp_proj.value

    def project(self, y):
        y_proj = np.maximum(self.lb, np.minimum(self.ub, y[self.var_inds]))
        return y_proj
