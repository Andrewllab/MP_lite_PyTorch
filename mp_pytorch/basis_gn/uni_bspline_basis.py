
import torch

from .basis_generator import BasisGenerator
from ..phase_gn import LinearPhaseGenerator


class UniBSplineBasis(BasisGenerator):

    def __init__(self,
                 phase_generator: LinearPhaseGenerator,
                 num_basis: int = 10,
                 degree_p: int = 3,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):
        """
        Clamped Uniform B-spline Basis on the [0,1] interval.
        Basis is precomputed, apply w*b to ge trajectory.
        To impose the B-spline starting from given initial condition, initialize
        the basis with "init_condition_order = 1" for initial position, "=2" for
        initial position and velocity.
        To impose the B-spline end at given end condition, use the argument
        "end_condition_order" similar to above.

        :param phase_generator: linear phase generator, scaling time to [0, 1]
        :param num_basis: number of basis
        :param degree_p: the degree of the polynomial of B-spline, and also the
        indicate the smoothness order of the B-spline at knots
        :param dtype:
        :param device:
        :param kwargs:
        """

        super().__init__(phase_generator, num_basis, dtype, device)
        self.degree_p = degree_p

        self.init_cond_order = kwargs.get("init_condition_order", 0)
        self.end_cond_order = kwargs.get("end_condition_order", 0)
        self.num_ctrlp = num_basis + self.init_cond_order + abs(self.end_cond_order)
        # number of knots needed, with respect to B-sp degree and number of
        # control points ( num_basis + init_cond_order+end_cond_order)
        num_knots = self.degree_p + 1 + self.num_ctrlp
        num_knots_non_rep_inside_1 = num_knots - 2*self.degree_p
        # uniform knots vector
        self.knots_vec = torch.linspace(0, 1, num_knots_non_rep_inside_1,
                                        dtype=dtype, device=device)
        knots_prev = torch.zeros(self.degree_p, dtype=dtype, device=device)
        knots_pro = torch.ones(self.degree_p, dtype=dtype, device=device)
        self.knots_vec = torch.cat([knots_prev, self.knots_vec, knots_pro])

    def basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        compute evaluated b-spline basis at given time points
        :param times:
        :return:
        """

        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of basis:
        # [*add_dim, num_times, num_ctrlp]

        phase = self.phase_generator.phase(times)

        basis = [self._basis_function(i, self.degree_p, self.knots_vec, phase)
                 for i in range(self.num_ctrlp)]
        basis = torch.stack(basis, dim=-1)

        return basis

    def _basis_function(self, i, k, knots, u, **kwargs):
        """
        recursive construct of B-spline basis using de Boor's algorithm

        :param i: basis index
        :param k: degree
        :param u: evaluate time point
        :param knots: knots vector
        :return: vector of shape [num_eval_points]
        """

        # adding some assertion to tell whether i is a feasible choice

        if k == 0:
            num_ctrlp = kwargs.get("num_ctrlp", self.num_ctrlp)
            if i == num_ctrlp-1:
                # with regard to original definition, each span is defined as \
                # left closed and right open interval [v_i, v_i+1), which makes\
                # the value at right end always 0. It is undesired,so that we \
                # need to handle the last basis specially
                b0 = torch.where((u >= knots[i]) & (u <= knots[i+1]), 1, 0)
            else:
                b0 = torch.where((u >= knots[i]) & (u < knots[i+1]), 1, 0)
            return torch.as_tensor(b0, dtype=self.dtype, device=self.device)
        else:
            denom1 = knots[i + k] - knots[i]
            term1 = 0.0 if denom1 == 0 else (u - knots[i]) / denom1 * \
                self._basis_function(i, k - 1, knots, u, **kwargs)
            denom2 = knots[i + k + 1] - knots[i + 1]
            term2 = 0.0 if denom2 == 0 else (knots[i + k + 1] - u) / denom2 * \
                self._basis_function(i + 1, k - 1, knots, u, **kwargs)
            return term1 + term2

    def velocity_control_points(self, ctrl_pts: torch.Tensor):
        """
        given the position control points (parameter), return the velocity control
        points for vel B-spline as linear combination of position control points.

        :param ctrl_pts: vector of position control points
        :return: velocity control points
        """
        # todo 想做constraint的话，应该写成矩阵乘法
        # diff shape: [*add_dim, num_dof, num_ctrlp-1]
        diff = ctrl_pts[..., 1:] - ctrl_pts[..., :-1]
        # shape: [num_basis-1]
        delta = self.knots_vec[1+self.degree_p: self.num_ctrlp+self.degree_p] -\
                self.knots_vec[1: self.num_ctrlp]
        #一般情况，注意公式是0-n共n+1个ctr_points
        diff = diff * (1/delta)
        return diff * self.degree_p

    def acceleration_control_points(self, ctrl_pts: torch.Tensor):
        """
        given the position control points (parameter), return the acceleration
        control points for acc B-spline as linear combination of position
        control points.

        :param ctrl_pts: vector of position control points
        :return: velocity control points
        """
        # shape: [*add_dim, num_dof, num_ctrlp-1]
        vel_ctrl_pts = self.velocity_control_points(ctrl_pts)
        # shape: [*add_dim, num_dof, num_ctrlp-2]
        diff = vel_ctrl_pts[..., 1:] - vel_ctrl_pts[..., :-1]
        # shape: [num_ctrlp-2]
        # delta = self.knots_vec[2+self.degree_p: self.num_ctrlp+self.degree_p-1]\
        #     - self.knots_vec[2: self.num_ctrlp-1]
        delta = self.knots_vec[
                2 + self.degree_p: self.num_ctrlp + self.degree_p] \
                - self.knots_vec[2: self.num_ctrlp]
        diff = diff * (1/delta)
        return diff * (self.degree_p-1)

    def vel_basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Directly get the basis of velocity B-spline
        :param times:
        :return:
        """

        phase = self.phase_generator.phase(times)

        # for clamped uni B-spline
        vel_nots_vec = self.knots_vec[1:-1]
        basis = \
            [self._basis_function(i, self.degree_p-1, vel_nots_vec, phase, num_ctrlp=self.num_ctrlp-1)
             for i in range(self.num_ctrlp-1)]
        basis = torch.stack(basis, dim=-1)
        return basis

    def acc_basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Directly get the basis of acceleration B-spline
        :param times:
        :return:
        """

        phase = self.phase_generator.phase(times)
        acc_knots_vec = self.knots_vec[2: -2]

        basis = [
            self._basis_function(i, self.degree_p - 2, acc_knots_vec, phase, num_ctrlp=self.num_ctrlp-2)
            for i in range(self.num_ctrlp - 2)]
        basis = torch.stack(basis, dim=-1)

        return basis

    def compute_init_params(self, init_pos, init_vel, **kwargs):
        """
        Given initial condition, compute corresponding the first control points
        :param init_pos:
        :param init_vel:
        :param kwargs:
        :return:
        """

        # Shape of init_pos:
        # [*add_dim, num_dof]
        #
        # Shape of init_vel:
        # [*add_dim, num_dof]
        #
        # return shape:
        # [*add_dim, num_dof, init_cond_order]

        if self.init_cond_order == 0:
            return None

        para_init_p = init_pos
        para_init = para_init_p[..., None]

        if self.init_cond_order == 2:
            para_init_v = \
                torch.einsum("...i,...->...i", init_vel, self.phase_generator.tau) * \
                (self.knots_vec[1 + self.degree_p] - self.knots_vec[1]) \
                / self.degree_p + para_init_p
            para_init = torch.cat([para_init, para_init_v[..., None]], dim=-1)

        return para_init

    def compute_end_params(self, end_pos, end_vel, **kwargs):
        """
        Given end condition, compute corresponding the last control points
        :param end_pos:
        :param end_vel:
        :param kwargs:
        :return:
        """
        # Shape of end_pos:
        # [*add_dim, num_dof]
        #
        # Shape of end_vel:
        # [*add_dim, num_dof]
        #
        # return shape:
        # [*add_dim, num_dof, init_cond_order]

        if self.end_cond_order == 0:
            return None

        if self.end_cond_order == -1:
            para_end = torch.einsum("...i,...->...i", end_vel,
                                    self.phase_generator.tau) * \
                       (self.knots_vec[self.num_ctrlp - 1 + self.degree_p] -
                        self.knots_vec[self.num_ctrlp - 1]) * self.degree_p
            return para_end[..., None]

        para_end_p = end_pos
        para_end = para_end_p[..., None]

        if self.end_cond_order == 2:
            para_end_v = para_end_p - \
                         torch.einsum("...i,...->...i", end_vel,
                                      self.phase_generator.tau) * \
                         (self.knots_vec[self.num_ctrlp - 1 + self.degree_p] -
                          self.knots_vec[self.num_ctrlp - 1]) * self.degree_p
            # para_end_v = para_end_p - (end_vel * self.phase_generator.tau) * \
            #               (self.knots_vec[self.num_ctrlp - 1 + self.degree_p] -
            #                self.knots_vec[self.num_ctrlp-1]) * self.degree_p
            para_end = torch.cat([para_end_v[..., None], para_end], dim=-1)

        return para_end

    def basis_multi_dofs(self,
                         times: torch.Tensor,
                         num_dof: int) -> torch.Tensor:
        """
        Index the initial condition and end condition irrelevant basis, and extend
        the dimensions for covariance calculation
        :param times:
        :param num_dof:
        :return:
        """

        # Shape of time
        # [*add_dim, num_times]
        #
        # Shape of basis_multi_dofs
        # [*add_dim, num_dof * num_times, num_dof * num_basis]

        # Extract additional dimensions
        add_dim = list(times.shape[:-1])
        num_times = times.shape[-1]
        # Get single basis, shape: [*add_dim, num_times, num_ctrlp]
        basis_single_dof = self.basis(times)
        # Get the not boundary condition relevant basis,
        # shape: [*add_dim, num_times, num_basis]
        if self.end_cond_order == -1:
            basis_end_pos = (basis_single_dof[..., -1] + basis_single_dof[..., -2])[..., None]
            basis_single_dof_ = torch.cat([basis_single_dof[..., self.init_cond_order: self.num_ctrlp-2],
                                          basis_end_pos], dim=-1)
        else:
            basis_single_dof_ = basis_single_dof[..., self.init_cond_order:
                                            self.num_ctrlp-self.end_cond_order]

        # Multiple Dofs, shape:
        # [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dofs = torch.zeros(*add_dim, num_dof * num_times,
                                       num_dof * self.num_basis, dtype=self.dtype,
                                       device=self.device)
        # Assemble
        for i in range(num_dof):
            row_indices = slice(i * num_times, (i + 1) * num_times)
            col_indices = slice(i * self.num_basis, (i + 1) * self.num_basis)
            basis_multi_dofs[..., row_indices, col_indices] = basis_single_dof_

        # Return
        return basis_multi_dofs



