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

    def set_initial_conditions(self, init_time,
                               init_conds):
        # Shape of init_time:
        # [*add_dim]
        #
        # init_conds: list of init_pos, init_vel and possibly init_acc
        # Shape of init_pos:
        # [*add_dim, num_dof]

        if self.order == 2:
            super().set_initial_conditions(init_time, *init_conds)
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

            self.init_pos = torch.as_tensor(init_conds[0], dtype=self.dtype, device=self.device)
            self.init_vel = torch.as_tensor(init_conds[1], dtype=self.dtype, device=self.device)
            self.init_acc = torch.as_tensor(init_conds[2], dtype=self.dtype, device=self.device)

            self.clear_computation_result()

    def update_inputs(self, times=None, params=None, params_L=None,
               init_time=None, init_conds=None, **kwargs):
        if params is not None:
            self.set_params(params)
        if times is not None:
            self.set_times(times)
        if init_time is not None and init_conds is not None:
            self.set_initial_conditions(init_time, init_conds)
        if params_L is not None:
            self.set_mp_params_variances(params_L)

    def get_traj_pos(self, times=None, params=None,
                     init_time=None, init_conds=None,
                     flat_shape=False):
        # Shape of pos
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, params, None, init_time, init_conds)

        if self.pos is not None:
            pos = self.pos
        else:
            # Recompute otherwise
            self.compute_intermediate_terms_single()

            # Reshape (and pad) params to [*add_dim, num_dof, num_basis_g]
            params = self.params.reshape([*self.add_dim, self.num_dof, -1])
            params = self.padding(params)

            # Scale basis functions
            pos_H_single = self.pos_H_single * self.weights_goal_scale

            # Position and velocity variant (part 3)
            # Einsum shape: [*add_dim, num_times, num_basis_g],
            #               [*add_dim, num_dof, num_basis_g]
            #            -> [*add_dim, num_dof, num_times]
            # Reshape to -> [*add_dim, num_dof * num_times]
            pos_linear = \
                torch.einsum('...jk,...ik->...ij', pos_H_single, params)
            pos_linear = torch.reshape(pos_linear, [*self.add_dim, -1])
            pos = self.pos_init + pos_linear

            if self.relative_goal:
                # Einsum shape: [*add_dim, num_times],
                #               [*add_dim, num_dof]
                #            -> [*add_dim, num_dof, num_times]
                # Reshape to -> [*add_dim, num_dof * num_times]
                pos_goal = \
                    torch.einsum('...j,...i->...ij', self.pos_H_single[..., -1],
                                 self.init_pos)
                pos_goal = torch.reshape(pos_goal, [*self.add_dim, -1])
                pos += pos_goal

            self.pos = pos

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            pos = pos.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            pos = torch.einsum('...ji->...ij', pos)

        return pos

    def get_traj_pos_cov(self, times=None, params_L=None, init_time=None,
                         init_conds=None, reg: float = 1e-4):

        # Shape of pos_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_conds)

        # Reuse result if existing
        if self.pos_cov is not None:
            return self.pos_cov

        # Recompute otherwise
        if self.params_L is None:
            return None

        # Get multi dof basis
        self.compute_intermediate_terms_multi_dof()

        # Scale basis functions
        weights_goal_scale = self.weights_goal_scale.repeat(self.num_dof)
        pos_H_multi = self.pos_H_multi * weights_goal_scale

        # Uncertainty of position
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g],
        #               [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        pos_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               pos_H_multi, self.params_cov, pos_H_multi)

        # Determine regularization term to make traj_cov positive definite
        traj_cov_reg = reg
        reg_term_pos = torch.max(torch.einsum('...ii->...i',
                                              pos_cov)).item() * traj_cov_reg

        # Add regularization term for numerical stability
        self.pos_cov = pos_cov + torch.eye(pos_cov.shape[-1],
                                           dtype=self.dtype,
                                           device=self.device) * reg_term_pos

        return self.pos_cov

    def get_traj_pos_std(self, times=None, params_L=None, init_time=None,
                         init_conds=None, flat_shape=False, reg: float = 1e-4):

        # Shape of pos_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_conds)

        # Reuse result if existing
        if self.pos_std is not None:
            pos_std = self.pos_std

        else:
            # Recompute otherwise
            pos_cov = self.get_traj_pos_cov(reg=reg)
            if pos_cov is None:
                pos_std = None
            else:
                # Shape [*add_dim, num_dof * num_times]
                pos_std = torch.sqrt(torch.einsum('...ii->...i', pos_cov))

            self.pos_std = pos_std

        if pos_std is not None and not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            pos_std = pos_std.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            pos_std = torch.einsum('...ji->...ij', pos_std)

        return pos_std

    def get_traj_vel(self, times=None, params=None,
                     init_time=None, init_conds=None,
                     flat_shape=False):

        # Shape of vel
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, params, None, init_time, init_conds)

        # Reuse result if existing
        if self.vel is not None:
            vel = self.vel
        else:
            # Recompute otherwise
            self.compute_intermediate_terms_single()

            # Reshape (and pad) params to [*add_dim, num_dof, num_basis_g]
            params = self.params.reshape([*self.add_dim, self.num_dof, -1])
            params = self.padding(params)

            # Scale basis functions
            vel_H_single = self.vel_H_single * self.weights_goal_scale

            # Position and velocity variant (part 3)
            # Einsum shape: [*add_dim, num_times, num_basis_g],
            #               [*add_dim, num_dof, num_basis_g]
            #            -> [*add_dim, num_dof, num_times]
            # Reshape to -> [*add_dim, num_dof * num_times]
            vel_linear = \
                torch.einsum('...jk,...ik->...ij', vel_H_single, params)
            vel_linear = torch.reshape(vel_linear, [*self.add_dim, -1])
            vel = self.vel_init + vel_linear

            if self.relative_goal:
                # Einsum shape: [*add_dim, num_times],
                #               [*add_dim, num_dof]
                #            -> [*add_dim, num_dof, num_times]
                # Reshape to -> [*add_dim, num_dof * num_times]
                vel_goal = \
                    torch.einsum('...j,...i->...ij', self.vel_H_single[..., -1],
                                 self.init_pos)
                vel_goal = torch.reshape(vel_goal, [*self.add_dim, -1])
                vel += vel_goal

            # Unscale velocity to original time scale space
            vel = vel / self.phase_gn.tau[..., None]
            self.vel = vel

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            vel = vel.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            vel = torch.einsum('...ji->...ij', vel)

        return vel

    def get_traj_vel_cov(self, times=None, params_L=None, init_time=None,
                         init_conds=None, reg: float = 1e-4):

        # Shape of vel_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_conds)

        # Reuse result if existing
        if self.vel_cov is not None:
            return self.vel_cov

        # Recompute otherwise
        if self.params_L is None:
            return None

        # Get multi dof basis
        self.compute_intermediate_terms_multi_dof()

        # Scale basis functions
        weights_goal_scale = self.weights_goal_scale.repeat(self.num_dof)
        vel_H_multi = self.vel_H_multi * weights_goal_scale

        # Uncertainty of velocity
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g],
        #               [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        vel_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               vel_H_multi, self.params_cov,
                               vel_H_multi)

        # Determine regularization term to make traj_cov positive definite
        traj_cov_reg = reg
        reg_term_vel = torch.max(torch.einsum('...ii->...i',
                                              vel_cov)).item() * traj_cov_reg

        # Add regularization term for numerical stability
        vel_cov = vel_cov + torch.eye(vel_cov.shape[-1],
                                      dtype=self.dtype,
                                      device=self.device) * reg_term_vel

        # Unscale velocity to original time scale space
        self.vel_cov = vel_cov / self.phase_gn.tau[..., None, None] ** 2

        return self.vel_cov

    def get_traj_vel_std(self, times=None, params_L=None, init_time=None,
                         init_conds=None, flat_shape=False, reg: float = 1e-4):

        # Shape of vel_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_conds)

        # Reuse result if existing
        if self.vel_std is not None:
            vel_std = self.vel_std
        else:
            # Recompute otherwise
            vel_cov = self.get_traj_vel_cov(reg=reg)
            if vel_cov is None:
                vel_std = None
            else:
                # Shape [*add_dim, num_dof * num_times]
                vel_std = torch.sqrt(torch.einsum('...ii->...i', vel_cov))
            self.vel_std = vel_std

        if vel_std is not None and not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            vel_std = vel_std.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            vel_std = torch.einsum('...ji->...ij', vel_std)

        return vel_std

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg: float = 1e-9, **kwargs) -> dict:

        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        times = torch.as_tensor(times, dtype=self.dtype, device=self.device)
        trajs = torch.as_tensor(trajs, dtype=self.dtype, device=self.device)

        # Get initial conditions

        if all([key in kwargs.keys()
                for key in ["init_time", "init_conds"]]):
            logging.warning("ProDMP+ uses the given initial conditions")
            init_time = kwargs["init_time"]
            init_conds = kwargs["init_conds"]
        else:
            init_time = times[..., 0]
            init_pos = trajs[..., 0, :]
            dt = (times[..., 1] - times[..., 0])
            init_vel_inter = torch.diff(trajs, dim=-2)
            init_vel = torch.einsum("...i,...->...i", init_vel_inter[..., 0, :],
                                    dt)
            init_conds = [init_pos, init_vel]
            if self.order == 3:
                init_acc_inter = torch.diff(init_vel_inter, dim=-2)
                init_acc = torch.einsum("...i,...->...i", init_acc_inter[..., 0, :],
                                        dt)
                init_conds.append(init_acc)

        # Setup stuff
        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_times(times)
        self.set_initial_conditions(init_time, init_conds)

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
            params_super = self.basis_gn.get_params()
            params = torch.cat([params_super, params], dim=-1)

        self.set_params(params)
        self.set_mp_params_variances(None)

        return {"params": params,
                "init_time": init_time,
                "init_conds": init_conds}

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

            # todo check whether neede to scale the vel and acc
            init_vel = self.init_vel * self.phase_gn.tau[..., None]
            init_acc = self.init_acc * self.phase_gn.tau[..., None]  # check

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
                + self.basis_gn.basis(self.times)

    def compute_initial_terms_multi_dof(self):
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
        tau = self.phase_gn.tau
        delay = self.phase_gn.delay
        assert tau.ndim == 0 and delay.ndim == 0
        times = torch.linspace(delay, delay + tau, steps=10000,
                               device=self.device, dtype=self.dtype)
        self.set_add_dim([])
        self.set_times(times)
        self.set_initial_conditions(
            init_time=torch.zeros([], device=self.device,
                                  dtype=self.dtype) + delay,
            init_conds=[torch.zeros([self.num_dof], device=self.device,
                                 dtype=self.dtype),
            torch.zeros([self.num_dof], device=self.device,
                                 dtype=self.dtype)],
        )

        self.compute_intermediate_terms_single()

        weights_goal_scale = self.weights_goal_scale

        dummy_params = torch.ones([self._num_local_params], device=self.device,
                                  dtype=self.dtype).reshape(self.num_dof, -1)
        # Shape: [num_basis_g]
        dummy_params_pad = self.padding(dummy_params)[0]

        # Get basis
        # Shape: [*add_dim, num_times, num_basis]
        basis_values = self.pos_H_single * weights_goal_scale * dummy_params_pad
        vel_basis_values =\
            self.vel_H_single * weights_goal_scale * dummy_params_pad

        # Unscale velocity back to original time-scale space
        vel_basis_values = vel_basis_values / self.phase_gn.tau[..., None]

        # Enforce all variables to numpy

        times, basis_values, vel_basis_values, delay, tau = \
            mp_pytorch.util.to_nps(times, basis_values, vel_basis_values,
                                   delay, tau)

        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, sharex=True, squeeze=False)
            for i in range(basis_values.shape[-1] - 1):
                axes[0, 0].plot(times, basis_values[:, i], label=f"w_basis_{i}")
                axes[1, 0].plot(times, vel_basis_values[:, i],
                                label=f"w_basis_{i}")
            axes[0, 0].grid()
            axes[0, 0].legend()
            axes[0, 0].axvline(x=delay, linestyle='--', color='k', alpha=0.3)
            axes[0, 0].axvline(x=delay + tau, linestyle='--', color='k',
                               alpha=0.3)

            axes[0, 1].plot(times, basis_values[:, -1], label=f"goal_basis")
            axes[0, 1].grid()
            axes[0, 1].legend()
            axes[0, 1].axvline(x=delay, linestyle='--', color='k', alpha=0.3)
            axes[0, 1].axvline(x=delay + tau, linestyle='--', color='k',
                               alpha=0.3)

            axes[1, 0].grid()
            axes[1, 0].legend()
            axes[1, 0].axvline(x=delay, linestyle='--', color='k', alpha=0.3)
            axes[1, 0].axvline(x=delay + tau, linestyle='--', color='k',
                               alpha=0.3)

            axes[1, 1].plot(times, vel_basis_values[:, -1], label=f"goal_basis")
            axes[1, 1].grid()
            axes[1, 1].legend()
            axes[1, 1].axvline(x=delay, linestyle='--', color='k', alpha=0.3)
            axes[1, 1].axvline(x=delay + tau, linestyle='--', color='k',
                               alpha=0.3)

            plt.show()
        return times, basis_values

    def sample_trajectories(self, times=None, params=None, params_L=None,
                            init_time=None, init_conds = None,
                            num_smp=1, flat_shape=False):
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

        if all([data is None for data in {times, params, params_L, init_time,
                                          init_conds}]):
            times = self.times
            params = self.params
            params_L = self.params_L
            init_time = self.init_time
            init_pos = self.init_pos
            init_vel = self.init_vel
            if self.order == 3:
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
            if self.order == 3:
                init_acc_smp = mp_pytorch.util.add_expand_dim(init_acc, [num_add_dim], [num_smp])
        else:
            init_time_smp = None
            init_pos_smp = None
            init_vel_smp = None
            if self.order == 3:
                init_acc_smp = None

        init_conds = [init_pos_smp, init_vel_smp]
        if self.order == 3:
            init_conds.append(init_acc_smp)

        # Update inputs
        self.reset()
        self.update_inputs(times_smp, params_smp, None,
                           init_time_smp, init_conds)

        # Get sample trajectories
        pos_smp = self.get_traj_pos(flat_shape=flat_shape)
        vel_smp = self.get_traj_vel(flat_shape=flat_shape)

        # Recover old inputs
        if params_super.nelement() != 0:
            params = torch.cat([params_super, params], dim=-1)
        self.reset()
        init_conds = [init_pos, init_vel]
        if self.order == 3:
            init_conds.append(init_acc)
        self.update_inputs(times, params, None, init_time, init_conds)

        return pos_smp, vel_smp


