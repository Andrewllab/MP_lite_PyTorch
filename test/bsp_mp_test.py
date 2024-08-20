import torch
from addict import Dict

from mp_pytorch.mp import MPFactory
from mp_pytorch import util


def get_mp_config():


    cfg = Dict()
    cfg.mp_type = "uni_bspline"
    cfg.num_dof = 2
    cfg.tau = 3
    cfg.learn_tau = True
    cfg.learn_delay = True
    cfg.mp_args.num_basis = 10
    cfg.mp_args.degree_p = 4
    cfg.mp_args.init_condition_order = 2
    cfg.mp_args.end_condition_order = -1
    cfg.mp_args.weights_scale = 0.9

    # assume we have 3 trajectories in a batch
    num_traj = 3

    # Get trajectory scaling
    tau, delay = 4, 1
    scale_delay = torch.Tensor([tau, delay])
    scale_delay = util.add_expand_dim(scale_delay, [0], [num_traj])

    # Get params
    params = torch.Tensor([100, 200, 300, -100, -200, -300,
                           100, 200, 300, 5] * cfg.num_dof)
    params = util.add_expand_dim(params, [0], [num_traj])
    params = torch.cat([scale_delay, params], dim=-1)

    diag = torch.Tensor([10, 20, 30, 10, 20, 30,
                         10, 20, 30, 10]* cfg.num_dof)
    off_diag = torch.linspace(-9.5, 9.4, 190)
    params_L = util.build_lower_matrix(diag, off_diag)
    params_L = util.add_expand_dim(params_L, [0], [num_traj])

    # Get times
    times = util.tensor_linspace(0, tau+delay, 2000).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])

    # Get IC
    init_time = scale_delay[:, 1]
    init_pos = 5 * torch.ones([num_traj, cfg.num_dof])
    init_vel = torch.ones_like(init_pos) * 10
    end_pos = 0 * torch.ones([num_traj, cfg.num_dof])
    end_vel = torch.ones_like(end_pos) * 10

    return cfg, params, params_L, times, init_time, init_pos, init_vel, end_pos, end_vel


def set_traj():
    cfg, params, params_L, times, init_time, init_pos, init_vel, end_pos, end_vel = get_mp_config()
    mp = MPFactory.init_mp(**cfg.to_dict())

    mp.update_inputs(times, params, params_L, init_time, init_pos, init_vel, end_pos=end_pos, end_vel=end_vel)
    pos = mp.get_traj_pos()
    vel = mp.get_traj_vel()

    pos_flat = mp.get_traj_pos(flat_shape=True)
    pos_cov = mp.get_traj_pos_cov()
    mvn = torch.distributions.MultivariateNormal(loc=pos_flat,
                                                 covariance_matrix=pos_cov,
                                                 validate_args=False)

    y, y_dot = mp.sample_trajectories()

    util.debug_plot(x=None, y=[pos[0, :, 0], y[0, 0, :, 0], y[1, 0, :, 0], y[2, 0, :, 0]],
                    labels=['m', 's1', 's2', 's3'], title='Trajectories')


def test_learn_trajs():

    cfg, params, params_L, times, init_time, init_pos, init_vel, end_pos, end_vel = get_mp_config()
    cfg.tau = 4
    cfg.delay = 0
    cfg.learn_tau = False
    cfg.learn_delay = False
    mp = MPFactory.init_mp(**cfg.to_dict())

    times = util.tensor_linspace(0, 4, 2000).squeeze(-1)
    times = util.add_expand_dim(times, [0], [3])

    gt = torch.sin(times)
    gt = util.add_expand_dim(gt, [-1], [cfg.num_dof])
    para = mp.learn_mp_params_from_trajs(times, gt)

    pos = mp.get_traj_pos()[0, :, 0].numpy()

    import matplotlib.pyplot as plt
    plt.plot(times[0, :].numpy(), gt[0, :, 0].numpy(), label="gt")
    plt.plot(times[0, :].numpy(), pos, label="learned",)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    set_traj()
    test_learn_trajs()

