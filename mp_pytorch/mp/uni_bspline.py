
import logging
from typing import Iterable
from typing import Union
from typing import Tuple

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from ..basis_gn import UniBSplineBasis
from .mp_interfaces import ProbabilisticMPInterface
from .. import util

class UniformBSpline(ProbabilisticMPInterface):

    def __init__(self,
                 basis_gn: UniBSplineBasis,
                 num_dof: int,
                 weights_scale: float = 1.,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs,
                 ):
        super().__init__(basis_gn, num_dof, weights_scale, dtype, device, **kwargs)

        self.end_pos = None
        self.end_vel = None

        self.params_init = None
        self.params_end = None

    def update_inputs(self, times=None, params=None, params_L=None,
                      init_time=None, init_pos=None, init_vel=None, **kwargs):
        super().update_inputs(times, params, params_L, init_time, init_pos, init_vel)
        end_pos = kwargs.get('end_pos', None)
        end_vel = kwargs.get('end_vel', None)
        if all([cond is not None for cond in [end_pos, end_vel]]):
            self.set_end_condtions(end_pos, end_vel)

    def set_initial_conditions(self, init_time: Union[torch.Tensor, np.ndarray],
                               init_pos: Union[torch.Tensor, np.ndarray],
                               init_vel: Union[torch.Tensor, np.ndarray]):

        if not torch.all((init_time == self.phase_gn.delay)):
            logging.warning("the initial condition only applies at the 0+delay time point")
        super().set_initial_conditions(init_time, init_pos, init_vel)
        self.params_init = self.basis_gn.compute_init_params(self.init_pos, self.init_vel)
        if self.params_init is not None:
            self.params_init /= self.weights_scale

    def set_end_condtions(self, end_pos: Union[torch.Tensor, np.ndarray],
                          end_vel: Union[torch.Tensor, np.ndarray], **kwargs):
        self.end_pos = \
            torch.as_tensor(end_pos, device=self.device, dtype=self.dtype) \
                if end_pos is not None else None
        self.end_vel = \
            torch.as_tensor(end_vel, device=self.device, dtype=self.dtype) \
                if end_vel is not None else None

        self.params_end = self.basis_gn.compute_end_params(self.end_pos, self.end_vel)
        if self.params_end is not None:
            self.params_end /= self.weights_scale

    def set_mp_params_variances(self, params_L: Union[torch.Tensor, None]):
        """
        Set variance of MP params
        Args:
            params_L: cholesky of covariance matrix of the MP parameters

        Returns: None

        """
        # Shape of params_L:
        # [*add_dim, num_dof * num_basis, num_dof * num_basis]

        if params_L is not None:
            assert list(params_L.shape) == [*self.add_dim,
                                            self._num_local_params,
                                            self._num_local_params]
        super().set_mp_params_variances(params_L)

    def get_traj_pos(self, times=None, params=None, init_time=None,
                     init_pos=None, init_vel=None, flat_shape=False, **kwargs):

        self.update_inputs(times, params, init_time, init_pos, init_vel, **kwargs)

        if self.pos is not None:
            pos = self.pos
        else:
            assert self.params is not None

            # Reshape params
            # [*add_dim, num_dof * num_basis] -> [*add_dim, num_dof, num_basis]
            params = self.params.reshape(*self.add_dim, self.num_dof, -1)
            # extend params with possible init and end conditions
            # shape: [*add_dim, num_dof, num_ctrlp]
            if self.params_init is not None:
                params = torch.cat((self.params_init, params), dim=-1)
            if self.params_end is not None:
                params = torch.cat((params, self.params_end), dim=-1)

            # Get basis
            # Shape: [*add_dim, num_times, num_ctrlp]
            basis_single_dof = \
                self.basis_gn.basis(self.times) * self.weights_scale

            # Einsum shape: [*add_dim, num_times, num_ctrlp],
            #               [*add_dim, num_dof, num_ctrlp]
            #            -> [*add_dim, num_times, num_dof]
            pos = torch.einsum('...ik,...jk->...ij', basis_single_dof, params)

            self.pos = pos

        if flat_shape:
            # Switch axes to [*add_dim, num_dof, num_times]
            pos = torch.einsum('...ji->...ij', pos)

            # Reshape to [*add_dim, num_dof * num_times]
            pos = pos.reshape(*self.add_dim, -1)

        return pos

    def get_traj_pos_cov(self, times=None, params_L=None, init_time=None,
                         init_pos=None, init_vel=None, reg: float = 1e-4, **kwargs):

        # Shape of pos_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_pos, init_vel, **kwargs)

        # Reuse result if existing
        if self.pos_cov is not None:
            return self.pos_cov

        # Otherwise recompute result
        if self.params_L is None:
            return None

        # Get basis of all Dofs
        # Shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dofs = self.basis_gn.basis_multi_dofs(
            self.times, self.num_dof) * self.weights_scale

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        #               [*add_dim, num_dof * num_basis, num_dof * num_basis]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        pos_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               basis_multi_dofs,
                               self.params_cov,
                               basis_multi_dofs)

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
                         init_pos=None, init_vel=None, flat_shape=False,
                         reg: float = 1e-4, **kwargs):

        # Shape of pos_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_pos, init_vel, **kwargs)

        # Reuse result if existing
        if self.pos_std is not None:
            pos_std = self.pos_std

        else:
            # Otherwise recompute
            if self.pos_cov is not None:
                pos_cov = self.pos_cov
            else:
                pos_cov = self.get_traj_pos_cov()

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

    # def get_traj_vel(self, times=None, params=None, init_time=None,
    #                  init_pos=None, init_vel=None, flat_shape=False, **kwargs):
    #     # only for test
    #     # Shape of vel
    #     # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]
    #
    #     # Update inputs
    #     self.update_inputs(times, params, None, init_time, init_pos, init_vel, **kwargs)
    #
    #     # Reuse result if existing
    #     if self.vel is not None:
    #         vel = self.vel
    #
    #     else:
    #         # Recompute otherwise
    #         pos = self.get_traj_pos()
    #
    #         vel = torch.zeros_like(pos, dtype=self.dtype, device=self.device)
    #         vel[..., :-1, :] = torch.diff(pos, dim=-2) \
    #                            / torch.diff(self.times)[..., None]
    #         vel[..., -1, :] = vel[..., -2, :]
    #
    #         self.vel = vel
    #
    #     if flat_shape:
    #         # Switch axes to [*add_dim, num_dof, num_times]
    #         vel = torch.einsum('...ji->...ij', vel)
    #
    #         # Reshape to [*add_dim, num_dof * num_times]
    #         vel = vel.reshape(*self.add_dim, -1)
    #
    #     return vel

    def get_traj_vel(self, times=None, params=None, init_time=None,
                     init_pos=None, init_vel=None, flat_shape=False, **kwargs):

        self.update_inputs(times, params, init_time, init_pos, init_vel,
                           **kwargs)

        if self.vel is not None:
            vel = self.vel
        else:
            assert self.params is not None

            # Reshape params
            # [*add_dim, num_dof * num_basis] -> [*add_dim, num_dof, num_basis]
            params = self.params.reshape(*self.add_dim, self.num_dof, -1)
            # extend params with possible init and end conditions
            # shape: [*add_dim, num_dof, num_ctrlp]
            if self.params_init is not None:
                params = torch.cat((self.params_init, params), dim=-1)
            if self.params_end is not None:
                params = torch.cat((params, self.params_end), dim=-1)

            # velocity control points shape: [*add_dim, num_dof, num_ctrlp-1]
            vel_ctrlp = self.basis_gn.velocity_control_points(params)

            # vel_basis shape: [*add_dim, num_times, num_ctrlp-1]
            vel_basis = self.basis_gn.vel_basis(self.times) * self.weights_scale

            # Einsum shape: [*add_dim, num_times, num_ctrlp-1],
            #               [*add_dim, num_dof, num_ctrlp-1]
            #            -> [*add_dim, num_times, num_dof]
            vel = torch.einsum('...ik,...jk->...ij', vel_basis, vel_ctrlp)

            self.vel = vel

        if flat_shape:
            # Switch axes to [*add_dim, num_dof, num_times]
            vel = torch.einsum('...ji->...ij', vel)

            # Reshape to [*add_dim, num_dof * num_times]
            vel = vel.reshape(*self.add_dim, -1)

        return vel

    def get_traj_vel_cov(self, times=None, params_L=None, init_time=None,
                         init_pos=None, init_vel=None, reg: float = 1e-4):
        return None

    def get_traj_vel_std(self, times=None, params_L=None, init_time=None,
                         init_pos=None, init_vel=None, flat_shape=False,
                         reg: float = 1e-4):
        return None

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor, reg=1e-9, **kwargs):

        # only works for learn_tau=False, learn_delay=False. And delay=0 (or you
        # need to give the initial condition by yourself)

        # Shape of times
        # [*add_dim, num_times]
        #
        # Shape of trajs:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of params:
        # [*add_dim, num_dof * num_basis]

        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        times = torch.as_tensor(times, dtype=self.dtype, device=self.device)
        trajs = torch.as_tensor(trajs, dtype=self.dtype, device=self.device)

        # Setup stuff
        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_times(times)
        dummy_params = torch.zeros(*self.add_dim, self.num_dof, self.num_basis,
                                   device=self.device, dtype=self.dtype)

        # Get initial conditions
        if self.basis_gn.init_cond_order != 0:
            if all([key in kwargs.keys()
                    for key in ["init_time", "init_pos", "init_vel"]]):
                logging.warning("uses the given initial conditions")
                init_time = kwargs["init_time"]
                init_pos = kwargs["init_pos"]
                init_vel = kwargs["init_vel"]
            else:
                init_time = times[..., 0]
                init_pos = trajs[..., 0, :]
                dt = (times[..., 1] - times[..., 0])
                init_vel = torch.einsum("...i,...->...i",
                                        torch.diff(trajs, dim=-2)[..., 0, :],
                                        dt)
            self.set_initial_conditions(init_time, init_pos, init_vel)
            if self.params_init is not None:
                dummy_params = torch.cat([self.params_init, dummy_params],
                                         dim=-1)

        if self.basis_gn.end_cond_order != 0:
            if all([key in kwargs.keys()
                    for key in ["end_pos", "end_vel"]]):
                logging.warning("uses the given end conditions")
                end_pos = kwargs["end_pos"]
                end_vel = kwargs["end_vel"]
            else:
                end_pos = trajs[..., -1, :]
                dt = (times[..., 1] - times[..., 0])
                end_vel = torch.einsum("...i,...->...i",
                                        torch.diff(trajs, dim=-2)[..., -1, :], dt)
            self.set_end_condtions(end_pos, end_vel)
            if self.params_end is not None:
                dummy_params = torch.cat([dummy_params, self.params_end],
                                         dim=-1)

        basis_single_dof = self.basis_gn.basis(times) * self.weights_scale
        # shape: [*add_dim, num_time, num_ctrlp]
        #        [*add_dim, num_dof, num_ctrlp]
        #        [*add_dim, num_times, num_dof]
        pos_det = torch.einsum('...ik,...jk->...ij', basis_single_dof, dummy_params)
        # swtich axes to [*add_dim, num_dof, num_times]
        pos_det = torch.einsum('...ij->...ji', pos_det)
        pos_det = pos_det.reshape(*self.add_dim, -1)

        basis_multi_dofs = self.basis_gn.basis_multi_dofs(self.times, self.num_dof) * self.weights_scale
        # Solve this: Aw = B -> w = A^{-1} B
        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis]
        #            -> [*add_dim, num_dof * num_basis, num_dof * num_basis]
        A = torch.einsum('...ki,...kj->...ij', basis_multi_dofs,
                         basis_multi_dofs)
        A += torch.eye(self._num_local_params,
                       dtype=self.dtype,
                       device=self.device) * reg

        # Swap axis and reshape: [*add_dim, num_times, num_dof]
        #                     -> [*add_dim, num_dof, num_times]
        trajs = torch.einsum("...ij->...ji", trajs)
        # Reshape [*add_dim, num_dof, num_times]
        #      -> [*add_dim, num_dof * num_times]
        trajs = trajs.reshape([*self.add_dim, -1])

        # Position minus initial condition terms,
        pos_w = trajs - pos_det

        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        #               [*add_dim, num_dof * num_times]
        #            -> [*add_dim, num_dof * num_basis]
        B = torch.einsum('...ki,...k->...i', basis_multi_dofs, pos_w)

        # Shape of weights: [*add_dim, num_dof * num_basis]
        params = torch.linalg.solve(A, B)

        # Check if parameters basis or phase generator exist
        if self.basis_gn.num_params > 0:
            # todo param super should be extended, add added dim. and reset later?
            params_super = self.basis_gn.get_params()
            params = torch.cat([params_super, params], dim=-1)

        self.set_params(params)
        self.set_mp_params_variances(None)

        return {"params": params,
                "init_pos": self.init_pos,
                "init_vel": self.init_vel,
                "end_pos": self.end_pos,
                "end_vel": self.end_vel,
                }

    def _show_scaled_basis(self, *args, **kwargs):
        pass

    def sample_trajectories(self, times=None, params=None, params_L=None,
                            init_time=None, init_pos=None, init_vel=None,
                            num_smp=1, flat_shape=False, **kwargs):

        end_pos = kwargs.get("end_pos")
        end_vel = kwargs.get("end_vel")
        if all([data is None for data in {times, params, params_L, init_time,
                                          init_pos, init_vel, *kwargs}]):
            times = self.times
            params = self.params
            params_L = self.params_L
            init_time = self.init_time
            init_pos = self.init_pos
            init_vel = self.init_vel
            end_pos = self.end_pos
            end_vel = self.end_vel

        num_add_dim = params.ndim - 1

        # Add additional sample axis to time
        # Shape [*add_dim, num_smp, num_times]
        times_smp = util.add_expand_dim(times, [num_add_dim], [num_smp])

        # Sample parameters, shape [num_smp, *add_dim, num_mp_params]
        params_smp = MultivariateNormal(loc=params, scale_tril=params_L,
                                        validate_args=False).rsample([num_smp])
        # reshape to [*add_dim, num_smp, num_mp_params]
        params_smp = torch.einsum("i...j->...ij", params_smp)

        params_super = self.basis_gn.get_params()
        if params_super.nelement() != 0:
            params_super_smp = util.add_expand_dim(params_super, [-2],
                                                              [num_smp])
            params_smp = torch.cat([params_super_smp, params_smp], dim=-1)

        # add additional sample axis to possible initial condition and end condtion
        if self.basis_gn.init_cond_order > 0:
            init_time_smp = util.add_expand_dim(init_time, [num_add_dim],
                                                [num_smp])
            init_pos_smp = util.add_expand_dim(init_pos, [num_add_dim],
                                               [num_smp])
            init_vel_smp = util.add_expand_dim(init_vel, [num_add_dim],
                                               [num_smp])
        else:
            init_time_smp = None
            init_pos_smp = None
            init_vel_smp = None
        if self.basis_gn.end_cond_order > 0:
            end_pos_smp = util.add_expand_dim(end_pos, [num_add_dim],
                                              [num_smp])
            end_vel_smp = util.add_expand_dim(end_vel, [num_add_dim],
                                              [num_smp])
        else:
            end_pos_smp = None
            end_vel_smp = None

        self.reset()
        self.update_inputs(times_smp, params_smp, None, init_time_smp,
                           init_pos_smp, init_vel_smp, end_pos=end_pos_smp,
                           end_vel=end_vel_smp)

        pos_smp = self.get_traj_pos(flat_shape=flat_shape)
        vel_smp = self.get_traj_vel(flat_shape=flat_shape)

        if params_super.nelement() != 0:
            params = torch.cat([params_super, params], dim=-1)
        self.reset()
        self.update_inputs(times, params, None, init_time, init_pos,
                           init_vel, end_pos=end_pos, end_vel=end_vel)

        return pos_smp, vel_smp

