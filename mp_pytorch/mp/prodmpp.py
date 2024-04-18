from typing import Iterable
from typing import Tuple
from typing import Union
import logging

import numpy as np
import torch
import logging

import mp_pytorch.util
from mp_pytorch.basis_gn import ProDMPPBasisGenerator
from .prodmp import ProDMP


class ProdMPP(ProDMP):

    def __init__(self,
                 basis_gn: ProDMPPBasisGenerator,
                 num_dof: int,
                 order: int,
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
            self.ddy1 = None
            self.ddy2 = None
            self.ddy3 = None

            self.y3_init = None
            self.dy3_init = None
            self.ddy1_init = None
            self.ddy2_init = None
            self.ddy3_init = None

            self.acc_init = None
            # self.vel_H_single = None  # check whether needed

    def set_times(self, times: torch.Tensor):
        if self.order == 2:
            super(ProdMPP).set_times(times)
        self.y1, self.y2, self.y3, self.dy1, self.dy2, self.dy3, self.ddy1, \
            self.ddy2, self.ddy3 = self.basis_gn.free_basis(times)
        # todo check how to call 父父方法

    def set_initial_conditions(self, init_time,
                               init_conds):

        pass

    def get_traj_pos(self, times=None, params=None,
                     init_time=None, init_conds=None,
                     flat_shape=False):
        pass

    def get_traj_vel(self):
        pass

    def learn_mp_params_from_trajs(self):
        pass

    def compute_intermediate_terms_single(self):
        if self.order == 2:
            super(ProdMPP).compute_intermediate_terms_single()
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

            pos_basis_init = self.basis_gn.basis(self.init_time[..., None]).squeeze(-2)
            vel_basis_init = self.basis_gn.vel_basis(self.init_time[..., None]).squeeze(-2)
            acc_basis_init = self.basis_gn.acc_basis(self.init_time[..., None]).squeeze(-2)

            # todo check whether neede to scale the vel and acc
            init_vel = self.init_vel * self.phase_gn.tau[..., None]
            init_acc = self.init_acc * self.phase_gn.tau[..., None]  # check

            pos_det = torch.einsum("...j, ...i->...ij", xi_1, self.init_pos)\
                        + torch.einsum("...j, ...i->...ij", xi_2, init_vel)\
                        + torch.einsum("...j, ...i->...ij", xi_3, init_acc)
            # vel_det

            self.pos_init = torch.reshape(pos_det, [*self.add_dim, -1])
            # self.vel_init = torch.reshape(vel_det, [*self.add_dim, -1])

            self.pos_H_single =\
                torch.einsum("...i,...j->...ij", -xi_1, pos_basis_init) \
                + torch.einsum("...i,...j->...ij", -xi_2, vel_basis_init) \
                + torch.einsum("...i,...j->...ij", -xi_3, acc_basis_init) \
                + self.basis_gn.basis(self.times)

            # self.vel_H_single = \
    def compute_initial_terms_multi_dof(self):
        if self.order == 2:
            super(ProdMPP).compute_intermediate_terms_multi_dof()
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

            # vel_H_

            pos_H_ = torch.reshape(pos_H_, [*self.add_dim, -1,
                                            self.num_dof*self.num_basis_g])
            # vel_H_

            self.pos_H_multi = \
                pos_H_ + self.basis_gn.basis_multi_dofs(self.times, self.num_dof)
            # self.vel_H_multi = \


