from typing import Iterable
from typing import Tuple
from typing import Union
import logging

import numpy as np
import torch
import logging
from torch.distributions import MultivariateNormal

import mp_pytorch.util
from mp_pytorch.basis_gn import ProDMPPBasisGenerator
from .prodmp import ProDMP


class ProDMPP(ProDMP):

    def __init__(self,
                 basis_gn: ProDMPPBasisGenerator,
                 num_dof: int,
                 order: int = 2,
                 weights_scale: float = 1,
                 goal_scale: float = 1,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):

        # todo
        super().__init__(basis_gn, num_dof, weights_scale, goal_scale,
                         dtype, device, **kwargs)
        self.order = order

        if self.order == 3:
            self.y3 = None
            self.dy3 = None
            # the following atrs are not used
            # self.ddy1 = None
            # self.ddy2 = None
            # self.ddy3 = None

            self.y3_init = None
            self.dy3_init = None
            self.ddy1_init = None
            self.ddy2_init = None
            self.ddy3_init = None

            self.init_acc = None
            # self.vel_H_single = None  # check whether needed

    def set_times(self, times):
        if self.order == 2:
            super().set_times(times)
        else:
            super(ProDMP, self).set_times(times)
            # self.y1, self.y2, self.y3, self.dy1, self.dy2, self.dy3, self.ddy1, \
            #     self.ddy2, self.ddy3 = self.basis_gn.general_solution_values(times)
            self.y1, self.y2, self.y3, self.dy1, self.dy2, self.dy3, _, \
                _, _ = self.basis_gn.general_solution_values(times)

    def set_initial_conditions(self, init_time: Union[torch.Tensor, np.ndarray],
                               init_pos: Union[torch.Tensor, np.ndarray],
                               init_vel: Union[torch.Tensor, np.ndarray],
                               **kwargs):
        # Shape of init_time:
        # [*add_dim]
        #
        # Shape of init_pos:
        # [*add_dim, num_dof]

        if self.order == 2:
            super().set_initial_conditions(init_time, init_pos, init_vel)
        else:
            self.init_time = torch.as_tensor(init_time, dtype=self.dtype,
                                             device=self.device)
            # possibly do assertion
            basis_init = self.basis_gn.general_solution_values(init_time[..., None])
            self.y1_init = basis_init[0].squeeze(-1)
            self.y2_init = basis_init[1].squeeze(-1)
            self.y3_init = basis_init[2].squeeze(-1)
            self.dy1_init = basis_init[3].squeeze(-1)
            self.dy2_init = basis_init[4].squeeze(-1)
            self.dy3_init = basis_init[5].squeeze(-1)
            self.ddy1_init = basis_init[6].squeeze(-1)
            self.ddy2_init = basis_init[7].squeeze(-1)
            self.ddy3_init = basis_init[8].squeeze(-1)

            super(ProDMP, self).set_initial_conditions(init_time, init_pos, init_vel)
            init_acc = kwargs.get("init_acc")
            if init_acc is not None:
                self.init_acc = torch.as_tensor(init_acc, dtype=self.dtype,
                                                device=self.device)
            else:
                logging.warning("no initial acceleration is given, 0 is set")
                self.init_acc = torch.zeros(init_pos.shape, dtype=self.dtype,
                            device=self.device)

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg: float = 1e-9, **kwargs) -> dict:

        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        times = torch.as_tensor(times, dtype=self.dtype, device=self.device)
        trajs = torch.as_tensor(trajs, dtype=self.dtype, device=self.device)

        # Get initial conditions

        if all([key in kwargs.keys()
                for key in ["init_time", "init_pos", "init_vel"]]):
            logging.warning("ProDMP+ uses the given initial conditions")
            init_time = kwargs["init_time"]
            init_pos = kwargs["init_pos"]
            init_vel = kwargs["init_vel"]
            init_acc = kwargs.get("init_acc")
        else:
            init_time = times[..., 0]
            init_pos = trajs[..., 0, :]
            dt = (times[..., 1] - times[..., 0])
            init_vel = torch.einsum("...i,...->...i",
                                    torch.diff(trajs, dim=-2)[..., 0, :],
                                    dt)
            init_acc = None
            if self.order == 3:
                init_acc = torch.einsum("...i,...->...i",
                                        torch.diff(trajs, n=2, dim=-2)[..., 0, :],
                                        dt)

        # Setup stuff
        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_times(times)
        self.set_initial_conditions(init_time, init_pos, init_vel, init_acc=init_acc)

        self.compute_intermediate_terms_single()
        self.compute_intermediate_terms_multi_dof()

        weights_goal_scale = self.weights_goal_scale.repeat(self.num_dof)
        pos_H_multi = self.pos_H_multi * weights_goal_scale

        # Solve this: Aw = B -> w = A^{-1} B
        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        A = torch.einsum('...ki,...kj->...ij', pos_H_multi, pos_H_multi)

        A += torch.eye(self.num_dof * self.num_basis_g,
                       dtype=self.dtype,
                       device=self.device) * reg

        # Swap axis and reshape: [*add_dim, num_times, num_dof]
        #                     -> [*add_dim, num_dof, num_times]
        trajs = torch.einsum("...ij->...ji", trajs)

        # Reshape [*add_dim, num_dof, num_times]
        #      -> [*add_dim, num_dof * num_times]
        trajs = trajs.reshape([*self.add_dim, -1])

        # Position minus initial condition terms,
        pos_wg = trajs - self.pos_init

        if self.relative_goal:
            # Einsum shape: [*add_dim, num_times],
            #               [*add_dim, num_dof]
            #            -> [*add_dim, num_dof, num_times]
            # Reshape to -> [*add_dim, num_dof * num_times]
            pos_goal = \
                torch.einsum('...j,...i->...ij', self.pos_H_single[..., -1],
                             self.init_pos)
            pos_goal = torch.reshape(pos_goal, [*self.add_dim, -1])
            pos_wg -= pos_goal

        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times]
        #            -> [*add_dim, num_dof * num_basis_g]
        B = torch.einsum('...ki,...k->...i', pos_H_multi, pos_wg)

        if self.disable_goal:
            basis_idx = [i for i in range(self.num_dof * self.num_basis_g)
                         if i % self.num_basis_g != self.num_basis_g - 1]
            A = mp_pytorch.util.get_sub_tensor(A, [-1, -2],
                                               [basis_idx, basis_idx])
            B = mp_pytorch.util.get_sub_tensor(B, [-1], [basis_idx])
        # todo disable weights

        # Shape of weights: [*add_dim, num_dof * num_basis_g]
        params = torch.linalg.solve(A, B)

        # Check if parameters basis or phase generator exist
        if self.basis_gn.num_params > 0:
            # todo param super should be extended, add added dim. and reset later?
            # todo also for prodmp
            params_super = self.basis_gn.get_params()
            params = torch.cat([params_super, params], dim=-1)

        self.set_params(params)
        self.set_mp_params_variances(None)

        return {"params": params,
                "init_time": init_time,
                "init_pos": init_pos,
                "init_vel": init_vel,
                "init_acc": init_acc}

    def compute_intermediate_terms_single(self):
        if self.order == 2:
            super().compute_intermediate_terms_single()
        else:
            det = self.y1_init * self.dy2_init * self.ddy3_init + \
                self.y2_init * self.dy3_init * self.ddy1_init + \
                self.y3_init * self.ddy2_init * self.dy1_init - \
                self.ddy1_init * self.dy2_init * self.y3_init - \
                self.dy1_init * self.y2_init * self.ddy3_init - \
                self.y1_init * self.dy3_init * self.ddy2_init

            # init_pos basis
            xi_1 = torch.einsum("...,...i->...i", (self.dy2_init*self.ddy3_init
                                - self.dy3_init*self.ddy2_init)/det, self.y1) + \
                torch.einsum("...,...i->...i", (self.dy3_init*self.ddy1_init
                             - self.dy1_init*self.ddy3_init)/det, self.y2) + \
                torch.einsum("...,...i->...i", (self.dy1_init*self.ddy2_init
                             - self.dy2_init*self.ddy1_init)/det, self.y3)
            # init_vel basis
            xi_2 = torch.einsum("...,...i->...i", (self.y3_init*self.ddy2_init
                                - self.y2_init*self.ddy3_init)/det, self.y1) + \
                torch.einsum("...,...i->...i", (self.y1_init*self.ddy3_init
                             - self.y3_init*self.ddy1_init)/det, self.y2) + \
                torch.einsum("...,...i->...i", (self.y2_init*self.ddy1_init
                             - self.y1_init*self.ddy2_init)/det, self.y3)
            # init_acc basis
            xi_3 = torch.einsum("...,...i->...i", (self.y2_init*self.dy3_init
                                - self.y3_init*self.dy2_init)/det, self.y1) + \
                torch.einsum("...,...i->...i", (self.y3_init*self.dy1_init
                             - self.y1_init*self.dy3_init)/det, self.y2) + \
                torch.einsum("...,...i->...i", (self.y1_init*self.dy2_init
                             - self.y2_init*self.dy1_init)/det, self.y3)

            dxi_1 = torch.einsum("...,...i->...i", (self.dy2_init*self.ddy3_init
                                - self.dy3_init*self.ddy2_init)/det, self.dy1) + \
                torch.einsum("...,...i->...i", (self.dy3_init*self.ddy1_init
                             - self.dy1_init*self.ddy3_init)/det, self.dy2) + \
                torch.einsum("...,...i->...i", (self.dy1_init*self.ddy2_init
                             - self.dy2_init*self.ddy1_init)/det, self.dy3)

            dxi_2 = torch.einsum("...,...i->...i", (self.y3_init*self.ddy2_init
                                - self.y2_init*self.ddy3_init)/det, self.dy1) + \
                torch.einsum("...,...i->...i", (self.y1_init*self.ddy3_init
                             - self.y3_init*self.ddy1_init)/det, self.dy2) + \
                torch.einsum("...,...i->...i", (self.y2_init*self.ddy1_init
                             - self.y1_init*self.ddy2_init)/det, self.dy3)

            dxi_3 = torch.einsum("...,...i->...i", (self.y2_init*self.dy3_init
                                - self.y3_init*self.dy2_init)/det, self.dy1) + \
                torch.einsum("...,...i->...i", (self.y3_init*self.dy1_init
                             - self.y1_init*self.dy3_init)/det, self.dy2) + \
                torch.einsum("...,...i->...i", (self.y1_init*self.dy2_init
                             - self.y2_init*self.dy1_init)/det, self.dy3)

            pos_basis_init = self.basis_gn.basis(self.init_time[..., None]).squeeze(-2)
            vel_basis_init = self.basis_gn.vel_basis(self.init_time[..., None]).squeeze(-2)
            acc_basis_init = self.basis_gn.acc_basis(self.init_time[..., None]).squeeze(-2)

            # check whether neede to scale the vel and acc
            init_vel = self.init_vel * self.phase_gn.tau[..., None]
            init_acc = self.init_acc * self.phase_gn.tau[..., None]  # check, accroding to difference method, it;s correct

            pos_det = torch.einsum("...j, ...i->...ij", xi_1, self.init_pos)\
                        + torch.einsum("...j, ...i->...ij", xi_2, init_vel)\
                        + torch.einsum("...j, ...i->...ij", xi_3, init_acc)
            vel_det = torch.einsum("...j, ...i->...ij", dxi_1, self.init_pos)\
                        + torch.einsum("...j, ...i->...ij", dxi_2, init_vel)\
                        + torch.einsum("...j, ...i->...ij", dxi_3, init_acc)

            self.pos_init = torch.reshape(pos_det, [*self.add_dim, -1])
            self.vel_init = torch.reshape(vel_det, [*self.add_dim, -1])

            self.pos_H_single =\
                torch.einsum("...i,...j->...ij", -xi_1, pos_basis_init) \
                + torch.einsum("...i,...j->...ij", -xi_2, vel_basis_init) \
                + torch.einsum("...i,...j->...ij", -xi_3, acc_basis_init) \
                + self.basis_gn.basis(self.times)

            self.vel_H_single = \
                torch.einsum("...i,...j->...ij", -dxi_1, pos_basis_init) \
                + torch.einsum("...i,...j->...ij", -dxi_2, vel_basis_init) \
                + torch.einsum("...i,...j->...ij", -dxi_3, acc_basis_init) \
                + self.basis_gn.vel_basis(self.times)

    def compute_intermediate_terms_multi_dof(self):
        if self.order == 2:
            super().compute_intermediate_terms_multi_dof()
        else:
            det = self.y1_init * self.dy2_init * self.ddy3_init + \
                  self.y2_init * self.dy3_init * self.ddy1_init + \
                  self.y3_init * self.ddy2_init * self.dy1_init - \
                  self.ddy1_init * self.dy2_init * self.y3_init - \
                  self.dy1_init * self.y2_init * self.ddy3_init - \
                  self.y1_init * self.dy3_init * self.ddy2_init

            # init_pos basis
            xi_1 = torch.einsum("...,...i->...i",
                                (self.dy2_init * self.ddy3_init
                                 - self.dy3_init * self.ddy2_init) / det,
                                self.y1) + \
                   torch.einsum("...,...i->...i",
                                (self.dy3_init * self.ddy1_init
                                 - self.dy1_init * self.ddy3_init) / det,
                                self.y2) + \
                   torch.einsum("...,...i->...i",
                                (self.dy1_init * self.ddy2_init
                                 - self.dy2_init * self.ddy1_init) / det,
                                self.y3)
            # init_vel basis
            xi_2 = torch.einsum("...,...i->...i", (self.y3_init * self.ddy2_init
                                                   - self.y2_init * self.ddy3_init) / det,
                                self.y1) + \
                   torch.einsum("...,...i->...i", (self.y1_init * self.ddy3_init
                                                   - self.y3_init * self.ddy1_init) / det,
                                self.y2) + \
                   torch.einsum("...,...i->...i", (self.y2_init * self.ddy1_init
                                                   - self.y1_init * self.ddy2_init) / det,
                                self.y3)
            # init_acc basis
            xi_3 = torch.einsum("...,...i->...i", (self.y2_init * self.dy3_init
                                                   - self.y3_init * self.dy2_init) / det,
                                self.y1) + \
                   torch.einsum("...,...i->...i", (self.y3_init * self.dy1_init
                                                   - self.y1_init * self.dy3_init) / det,
                                self.y2) + \
                   torch.einsum("...,...i->...i", (self.y1_init * self.dy2_init
                                                   - self.y2_init * self.dy1_init) / det,
                                self.y3)

            dxi_1 = torch.einsum("...,...i->...i",
                                 (self.dy2_init * self.ddy3_init
                                  - self.dy3_init * self.ddy2_init) / det,
                                 self.dy1) + \
                    torch.einsum("...,...i->...i",
                                 (self.dy3_init * self.ddy1_init
                                  - self.dy1_init * self.ddy3_init) / det,
                                 self.dy2) + \
                    torch.einsum("...,...i->...i",
                                 (self.dy1_init * self.ddy2_init
                                  - self.dy2_init * self.ddy1_init) / det,
                                 self.dy3)

            dxi_2 = torch.einsum("...,...i->...i",
                                 (self.y3_init * self.ddy2_init
                                  - self.y2_init * self.ddy3_init) / det,
                                 self.dy1) + \
                    torch.einsum("...,...i->...i",
                                 (self.y1_init * self.ddy3_init
                                  - self.y3_init * self.ddy1_init) / det,
                                 self.dy2) + \
                    torch.einsum("...,...i->...i",
                                 (self.y2_init * self.ddy1_init
                                  - self.y1_init * self.ddy2_init) / det,
                                 self.dy3)

            dxi_3 = torch.einsum("...,...i->...i", (self.y2_init * self.dy3_init
                                                    - self.y3_init * self.dy2_init) / det,
                                 self.dy1) + \
                    torch.einsum("...,...i->...i", (self.y3_init * self.dy1_init
                                                    - self.y1_init * self.dy3_init) / det,
                                 self.dy2) + \
                    torch.einsum("...,...i->...i", (self.y1_init * self.dy2_init
                                                    - self.y2_init * self.dy1_init) / det,
                                 self.dy3)

            pos_basis_init_multi_dofs = self.basis_gn.basis_multi_dofs(
                self.init_time[..., None], self.num_dof)
            vel_basis_init_multi_dofs = self.basis_gn.vel_basis_multi_dofs(
                self.init_time[..., None], self.num_dof)
            acc_basis_init_multi_dofs = self.basis_gn.acc_basis_multi_dofs(
                self.init_time[..., None], self.num_dof)

            pos_H_ = torch.einsum('...j,...ik->...ijk',
                                  -xi_1, pos_basis_init_multi_dofs) + \
                torch.einsum('...j,...ik->...ijk',
                             -xi_2, vel_basis_init_multi_dofs) + \
                torch.einsum('...j,...ik->...ijk',
                             -xi_3, acc_basis_init_multi_dofs)

            vel_H_ = torch.einsum('...j,...ik->...ijk',
                                  -dxi_1, pos_basis_init_multi_dofs) + \
                torch.einsum('...j,...ik->...ijk',
                             -dxi_2, vel_basis_init_multi_dofs) + \
                torch.einsum('...j,...ik->...ijk',
                             -dxi_3, acc_basis_init_multi_dofs)

            pos_H_ = torch.reshape(pos_H_, [*self.add_dim, -1,
                                            self.num_dof*self.num_basis_g])
            vel_H_ = torch.reshape(vel_H_, [*self.add_dim, -1,
                                           self.num_dof * self.num_basis_g])

            self.pos_H_multi = \
                pos_H_ + self.basis_gn.basis_multi_dofs(self.times,
                                                        self.num_dof)
            self.vel_H_multi = \
                vel_H_ + self.basis_gn.vel_basis_multi_dofs(self.times,
                                                            self.num_dof)

    def _show_scaled_basis(self, plot=False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        if self.order == 2:
            super()._show_scaled_basis(self, plot=plot)
        else:
            raise NotImplementedError

    def sample_trajectories(self, times=None, params=None, params_L=None,
                            init_time=None, init_pos = None, init_vel=None,
                            num_smp=1, flat_shape=False, **kwargs):
        """
        Sample trajectories from MP

        Args:
            times: time points
            params: learnable parameters
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            num_smp: num of trajectories to be sampled
            flat_shape: if flatten the dimensions of Dof and time

        Returns:
            sampled trajectories
        """

        # Shape of pos_smp
        # [*add_dim, num_smp, num_times, num_dof]
        # or [*add_dim, num_smp, num_dof * num_times]
        if self.order==2:
            super().sample_trajectories(times, params, params_L,
                            init_time, init_pos, init_vel,
                            num_smp, flat_shape, **kwargs)

        init_acc = kwargs.get("init_acc")
        if all([data is None for data in {times, params, params_L, init_time,
                                          init_pos, init_vel, init_acc}]):
            times = self.times
            params = self.params
            params_L = self.params_L
            init_time = self.init_time
            init_pos = self.init_pos
            init_vel = self.init_vel
            init_acc =self.init_acc

        num_add_dim = params.ndim - 1

        # Add additional sample axis to time
        # Shape [*add_dim, num_smp, num_times]
        times_smp = mp_pytorch.util.add_expand_dim(times, [num_add_dim], [num_smp])

        # Sample parameters, shape [num_smp, *add_dim, num_mp_params]
        params_smp = MultivariateNormal(loc=params,
                                        scale_tril=params_L,
                                        validate_args=False).rsample([num_smp])

        # Switch axes to [*add_dim, num_smp, num_mp_params]
        params_smp = torch.einsum('i...j->...ij', params_smp)

        params_super = self.basis_gn.get_params()
        if params_super.nelement() != 0:
            params_super_smp = mp_pytorch.util.add_expand_dim(params_super, [-2],
                                                   [num_smp])
            params_smp = torch.cat([params_super_smp, params_smp], dim=-1)

        # Add additional sample axis to initial condition
        if init_time is not None:
            init_time_smp = mp_pytorch.util.add_expand_dim(init_time, [num_add_dim], [num_smp])
            init_pos_smp = mp_pytorch.util.add_expand_dim(init_pos, [num_add_dim], [num_smp])
            init_vel_smp = mp_pytorch.util.add_expand_dim(init_vel, [num_add_dim], [num_smp])
            init_acc_smp = mp_pytorch.util.add_expand_dim(init_acc, [num_add_dim], [num_smp])
        else:
            init_time_smp = None
            init_pos_smp = None
            init_vel_smp = None
            init_acc_smp = None

        # Update inputs
        self.reset()
        self.update_inputs(times_smp, params_smp, None,
                           init_time_smp, init_pos_smp, init_vel_smp, init_acc=init_acc_smp)

        # Get sample trajectories
        pos_smp = self.get_traj_pos(flat_shape=flat_shape)
        vel_smp = self.get_traj_vel(flat_shape=flat_shape)

        # Recover old inputs
        if params_super.nelement() != 0:
            params = torch.cat([params_super, params], dim=-1)
        self.reset()
        self.update_inputs(times, params, None, init_time, init_pos,
                           init_vel, init_acc=init_acc)

        return pos_smp, vel_smp


