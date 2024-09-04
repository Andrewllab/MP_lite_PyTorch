import torch
import matplotlib.pyplot as plt

from mp_pytorch.basis_gn import LinearPhaseGenerator
from mp_pytorch.basis_gn import UniBSplineBasis
from mp_pytorch.mp import UniformBSpline
from mp_pytorch import util


def get_pos_vel_multidof():
    ph_gn = LinearPhaseGenerator(tau=3)
    # degree_p for degree of polynomial, which also means the smoothness order at
    # knots and effect the bandwidth
    #
    # init_condition_order = 2 means imposing initial pos+vel.
    # end_condition_order for imposing end condition
    # b_basis = UniBSplineBasis(ph_gn, degree_p=4,
    #                           init_condition_order=2, )
    # b_basis = UniBSplineBasis(ph_gn, num_basis=6, degree_p=4,
    #                           init_condition_order=2, end_condition_order=-1, goal_basis=True)
    b_basis = UniBSplineBasis(ph_gn, num_basis=6, degree_p=5,
                              init_condition_order=2, end_condition_order=2, goal_basis=True)
    mp = UniformBSpline(b_basis, num_dof=2)

    num_traj = 3
    # params = torch.tensor([100, 200, 300, -100, -200, -300,
    #                        100, 200, 300, -2] * 2)
    # params = torch.Tensor([1, 2, 2, 2, 2, 2,
    #                        2, 2, 2, 0] * 2)
    # params = torch.Tensor([0, 0, 0, 0, 0, 0,
    #                        0, 0, 0, 0] * 2)
    params = torch.Tensor([0, 0, 0, 0, 0, 0] * 2)
    params = util.add_expand_dim(params, [0], [num_traj])

    # diag = torch.Tensor([10, 20, 30, 10, 20, 30,
    #                      10, 20, 30, 4] * 2)
    # off_diag = torch.linspace(-9.5, 9.4, 190)
    # params_L = util.build_lower_matrix(diag, off_diag)
    # params_L = util.add_expand_dim(params_L, [0], [num_traj])
    params_L = None

    times = util.tensor_linspace(0, 3, 1000).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])
    init_time = times[:, 0]
    init_pos = 2 * torch.ones([num_traj, 2])
    init_vel = torch.zeros_like(init_pos)
    init_vel = torch.ones_like(init_pos) * -1
    end_pos = torch.zeros_like(init_pos)
    end_vel = torch.zeros_like(init_pos)
    end_vel = torch.ones_like(init_pos) *0.3

    # if imposing end conditon, give kwarg end_pos=blabla, end_vel=blabla
    # mp.update_inputs(times, params, params_L, init_time, init_pos, init_vel)
    mp.update_inputs(times, params, params_L, init_time, init_pos, init_vel, end_pos=end_pos, end_vel=end_vel)
    pos = mp.get_traj_pos()
    vel = mp.get_traj_vel()
    acc = mp.get_traj_acc()
    util.debug_plot(x=None, y=[pos[0, :, 0], pos[1,:,0], pos[2,:,0]], labels=['y0', 'y1', 'y2'], title="pos")
    util.debug_plot(x=None, y=[vel[0, :, 0], vel[1, :, 0], vel[2, :,0]], labels=['v0', 'v1', 'v2'], title="vel")
    util.debug_plot(x=None, y=[acc[0, :, 0]], labels=["acc"], title="acc")

    # pos_mean = mp.get_traj_pos(flat_shape=True)
    # pos_cov = mp.get_traj_pos_cov()
    # mvn = torch.distributions.MultivariateNormal(loc=pos_mean,
    #                                              covariance_matrix=pos_cov,
    #                                              validate_args=False)


def single_dof_with_std_plot():
    # 2 inital conditon + 2 end condition

    ph_gn = LinearPhaseGenerator()
    # degree_p for degree of polynomial, which also means the smoothness order at
    # knots and effect the bandwidth
    #
    # init_condition_order = 2 means imposing initial pos+vel.
    # end_condition_order for imposing end condition
    # b_basis = UniBSplineBasis(ph_gn, degree_p=4,
    #                           init_condition_order=2, end_condition_order=2)
    b_basis = UniBSplineBasis(ph_gn, degree_p=4,
                              init_condition_order=2, end_condition_order=-1)
    mp = UniformBSpline(b_basis, num_dof=1)

    num_traj = 3  # add_dim, the batch
    params = torch.tensor([100, 200, 300, -100, -200, -300,
                           100, 200, 300, -2])
    params = util.add_expand_dim(params, [0], [num_traj])

    diag = torch.Tensor([10, 20, 30, 10, 20, 30,
                         10, 20, 30, 4])
    off_diag = torch.linspace(-9.5, 9.4, 45)
    params_L = util.build_lower_matrix(diag, off_diag)
    params_L = util.add_expand_dim(params_L, [0], [num_traj])

    times = util.tensor_linspace(0, 1, 1000).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])

    init_time = times[:, 0]
    init_pos = 5 * torch.ones([num_traj, 1])
    init_vel = torch.zeros_like(init_pos)
    end_pos = -5 * torch.ones([num_traj, 1])
    end_vel = torch.zeros_like(end_pos)

    mp.update_inputs(times, params, params_L, init_time, init_pos, init_vel,
                     end_pos=end_pos, end_vel=end_vel)
    pos = mp.get_traj_pos()
    std = mp.get_traj_pos_std()

    y_smp = mp.sample_trajectories()
    y_ = y_smp[0][0, 0, :, 0]

    ax = plt.subplot()
    util.fill_between(times[0, :], pos[0, :, 0], std[0, :, 0], draw_mean=True,
                      axis=ax)
    ax.plot(times[0, :].numpy(), y_.numpy(), label="sample")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    get_pos_vel_multidof()
    single_dof_with_std_plot()

    # see also bsp_mp_test.py

