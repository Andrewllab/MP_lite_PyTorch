
import torch

from mp_pytorch import util
from mp_pytorch.phase_gn import LinearPhaseGenerator
from .norm_rbf_basis import NormalizedRBFBasisGenerator


class ProDMPPBasisGenerator(NormalizedRBFBasisGenerator):

    def __init__(self,
                 phase_generator: LinearPhaseGenerator,
                 order: int = 2,
                 num_basis: int = 10,
                 basis_bandwidth_factor: float = 2.0,
                 alpha: float = 25,
                 num_basis_outside: int = 0,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):
        super(ProDMPPBasisGenerator, self).__init__(phase_generator,
                                                    num_basis,
                                                    basis_bandwidth_factor,
                                                    num_basis_outside,
                                                    dtype, device)
        self.alpha = alpha
        self.order = order
        self.window_func = get_func(order)
        self.goal_func = get_func(order, self.alpha)

        self.num_basis_g = self.num_basis + 1

    def basis(self, times: torch.Tensor):

        # shape of times:
        # [*add_dim, num_tims]
        #
        # shape of basis:
        # [*add_dim, num_times, num_basis_g]

        # convert times to phases
        times = self.phase_generator.unbound_phase(times)
        nrbf_basis = super(ProDMPPBasisGenerator, self).basis(times)
        window = self.window_func(times)
        # shape: [*add_dim, num_tims], [*add_dim, num_times, num_basis]
        #        -> [*add_dim, num_tims, num_basis]
        f_basis = torch.einsum('...,...i->...i', window, nrbf_basis)
        g_basis = self.goal_func(times)
        basis = torch.cat([f_basis, g_basis[..., None]], dim=-1)
        return basis

    # def vel_basis(self, times: torch.Tensor):
    #
    #     # Shape of times:
    #     # [*add_dim, num_times]
    #     #
    #     # Shape of vel_basis:
    #     # [*add_dim, num_times, num_basis_g]
    #     # 一阶差分
    #     dt = (times[..., -1] - times[..., -2])[..., None]
    #     times = torch.cat([times, times[..., -1]+dt], dim=-1)
    #     pos_basis = self.basis(times)
    #     vel_basis = (pos_basis[..., 1:, :] - pos_basis[..., :-1, :]) / \
    #                 (dt/self.phase_generator.tau)
    #     return vel_basis

    def vel_basis(self, times: torch.Tensor):

        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of vel_basis:
        # [*add_dim, num_times, num_basis_g]
        # 二阶差分
        dt = (times[..., -1] - times[..., -2])[..., None]
        times = torch.cat([times[..., 0]-dt, times, times[..., -1]+dt], dim=-1)
        pos_basis = self.basis(times)
        vel_basis = (pos_basis[..., 2:, :] - pos_basis[..., :-2, :]) / \
                    (2*dt/self.phase_generator.tau)
        return vel_basis

    def acc_basis(self, times: torch.Tensor):

        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of vel_basis:
        # [*add_dim, num_times, num_basis_g]
        # 一阶差分
        dt = (times[..., -1] - times[..., -2])[..., None]
        times = torch.cat([times, times[..., -1]+dt], dim=-1)
        vel_basis = self.vel_basis(times)
        acc_basis = (vel_basis[..., 1:, :] - vel_basis[..., :-1, :]) / \
                    (dt/self.phase_generator.tau)
        return acc_basis

    def general_solution_values(self, times: torch.Tensor):

        # Shape of times
        # [*add_dim, num_times]
        #
        # Shape of each return
        # [*add_dim, num_times]

        # convert times to phases
        times = self.phase_generator.unbound_phase(times)
        free_basis = []
        derivative = []
        for i in range(self.order):
            free_basis_i = times ** i * torch.exp(-self.alpha * times)
            free_basis.append(free_basis_i)
            der = (i * times**max(i-1, 0) - self.alpha * times**i) * \
                  torch.exp(-self.alpha * times)
            derivative.append(der)
        if self.order == 3:
            for i in range(self.order):
                der = (i * (i-1) * times**max(i-2, 0) -
                       2 * self.alpha * i * times**max(i-1, 0) +
                       self.alpha**2 * times**i) * torch.exp(-self.alpha*times)
                derivative.append(der)

        return free_basis+derivative

    def vel_basis_multi_dofs(self, times: torch.Tensor, num_dof: int):

        # Shape of time
        # [*add_dim, num_times]
        #
        # Shape of vel_basis_multi_dofs
        # [*add_dim, num_dof * num_times, num_dof * num_basis_g]

        # Extract additional dimensions
        add_dim = list(times.shape[:-1])

        # Get single basis, shape: [*add_dim, num_times, num_basis_g]
        vel_basis_single_dof = self.vel_basis(times)
        num_times = vel_basis_single_dof.shape[-2]

        # Multiple Dofs, shape:
        # [*add_dim, num_dof * num_times, num_dof * num_basis]
        vel_basis_multi_dofs = torch.zeros(*add_dim,
                                           num_dof * num_times,
                                           num_dof * self.num_basis_g,
                                           dtype=self.dtype, device=self.device)
        # Assemble
        for i in range(num_dof):
            row_indices = slice(i * num_times,
                                (i + 1) * num_times)
            col_indices = slice(i * self.num_basis_g,
                                (i + 1) * self.num_basis_g)
            vel_basis_multi_dofs[..., row_indices, col_indices] = \
                vel_basis_single_dof

        # Return
        return vel_basis_multi_dofs

    def acc_basis_multi_dofs(self, times: torch.Tensor, num_dof: int):

        # Shape of time
        # [*add_dim, num_times]
        #
        # Shape of acc_basis_multi_dofs
        # [*add_dim, num_dof * num_times, num_dof * num_basis_g]

        # Extract additional dimensions
        add_dim = list(times.shape[:-1])

        # Get single basis, shape: [*add_dim, num_times, num_basis_g]
        acc_basis_single_dof = self.acc_basis(times)
        num_times = acc_basis_single_dof.shape[-2]

        # Multiple Dofs, shape:
        # [*add_dim, num_dof * num_times, num_dof * num_basis]
        acc_basis_multi_dofs = torch.zeros(*add_dim,
                                           num_dof * num_times,
                                           num_dof * self.num_basis_g,
                                           dtype=self.dtype, device=self.device)
        # Assemble
        for i in range(num_dof):
            row_indices = slice(i * num_times,
                                (i + 1) * num_times)
            col_indices = slice(i * self.num_basis_g,
                                (i + 1) * self.num_basis_g)
            acc_basis_multi_dofs[..., row_indices, col_indices] = \
                acc_basis_single_dof

        # Return
        return acc_basis_multi_dofs

    def show_basis(self, plot=False):
        """
        Compute basis function values for debug usage
        The times are in the range of [delay - tau, delay + 2 * tau]

        Returns: basis function values

        """
        tau = self.phase_generator.tau
        delay = self.phase_generator.delay
        assert tau.ndim == 0 and delay.ndim == 0
        times = torch.linspace(delay, delay + tau, steps=1000)
        basis_values = self.basis(times)
        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, sharex=True, squeeze=False)
            for i in range(basis_values.shape[-1] - 1):
                axes[0, 0].plot(times, basis_values[:, i], label=f"w_basis_{i}")
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

            plt.show()
        return times, basis_values


def _2ord(times: torch.Tensor, alpha: float = 50):
    return -alpha * times * torch.exp(- alpha * times) \
            - torch.exp(- alpha * times) + 1


def _3ord(times: torch.Tensor, alpha: float = 50):
    return - 0.5 * alpha**2 * times**2 * torch.exp(-alpha * times) \
        - alpha * times * torch.exp(- alpha * times) - torch.exp(-alpha * times)\
        + 1


ord2func = dict([(2, _2ord), (3, _3ord)])


def get_func(order: int, alpha: float = 36):
    func = ord2func[order]

    def func_ready(times: torch.Tensor):
        return func(times, alpha)

    return func_ready

