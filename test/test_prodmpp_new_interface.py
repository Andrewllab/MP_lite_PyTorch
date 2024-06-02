
import numpy as np
import torch
from addict import Dict

from mp_pytorch import util
from mp_pytorch.mp import MPFactory

def get_mp_config():


    cfg = Dict()
    cfg.mp_type = "prodmp+"
    cfg.num_dof = 2
    cfg.tau = 3
    # cfg.learn_tau = True
    # cfg.learn_delay = True
    cfg.mp_args.num_basis = 9
    cfg.mp_args.basis_bandwidth_factor = 1.7
    cfg.mp_args.num_basis_outside = 0
    cfg.mp_args.alpha = 25
    # cfg.mp_args.weights_scale = torch.ones([9]) * 0.1
    # cfg.mp_args.goal_scale = 1

    # assume we have 3 trajectories in a batch
    num_traj = 3

    # Get trajectory scaling
    tau, delay = 3, 0
    # scale_delay = torch.Tensor([tau, delay])
    # scale_delay = util.add_expand_dim(scale_delay, [0], [num_traj])

    # Get params
    params = torch.Tensor([100, 200, 300, -100, -200, -300,
                           100, 200, 300, -2] * cfg.num_dof)
    params = util.add_expand_dim(params, [0], [num_traj])
    # params = torch.cat([scale_delay, params], dim=-1)

    # Get times
    times = util.tensor_linspace(0, tau+delay, 10000).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])

    # Get IC
    init_time = times[:, 0]
    init_pos = 5 * torch.ones([num_traj, cfg.num_dof])
    init_vel = torch.zeros_like(init_pos)
    init_acc = torch.zeros_like(init_pos)

    return cfg, params, times, init_time, init_pos, init_vel, init_acc


def set_traj():
    cfg, params, times, init_time, init_pos, init_vel, init_acc =get_mp_config()

    cfg.mp_args.order = 2
    mp_2ord = MPFactory.init_mp(**cfg.to_dict())
    cfg.mp_args.order = 3
    mp_3ord = MPFactory.init_mp(**cfg.to_dict())

    mp_2ord.update_inputs(times=times, params=params,
                          init_time=init_time, init_pos=init_pos, init_vel=init_vel)

    mp_3ord.update_inputs(times=times, params=params,
                          init_time=init_time, init_pos=init_pos, init_vel=init_vel,
                          init_acc=init_acc)

    mp_2ord_pos = mp_2ord.get_traj_pos()
    mp_2ord_vel = mp_2ord.get_traj_vel()
    mp_3ord_pos = mp_3ord.get_traj_pos()
    mp_3ord_vel = mp_3ord.get_traj_vel()
    dt = torch.diff(times, dim=-1)[0][0]
    diff_vel_3 = torch.diff(mp_3ord_pos, dim=-2)/dt



    util.debug_plot(x=None, y=[mp_2ord_pos[0, :, 0], mp_3ord_pos[0, :, 0]],
                    labels=["2.", "3."], title="pos")
    util.debug_plot(x=None, y=[mp_2ord_vel[0, :, 0], mp_3ord_vel[0, :, 0], diff_vel_3[0, :, 0]],
                    labels=["2.", "3.", "3_diff"], title="vel")


def test_learn_trajs():

    cfg, params, times, init_time, init_pos, init_vel, init_acc = get_mp_config()
    cfg.mp_args.order = 2
    mp_2ord = MPFactory.init_mp(**cfg.to_dict())
    cfg.mp_args.order = 3
    mp_3ord = MPFactory.init_mp(**cfg.to_dict())

    gt = torch.sin(times)
    gt = util.add_expand_dim(gt, [-1], [cfg.num_dof])
    p1 = mp_2ord.learn_mp_params_from_trajs(times, gt)
    pw = mp_3ord.learn_mp_params_from_trajs(times, gt)

    pos_2ord = mp_2ord.get_traj_pos()[0,:,0].numpy()
    pos_3ord = mp_3ord.get_traj_pos()[0,:,0].numpy()

    import matplotlib.pyplot as plt
    plt.plot(times[0,:].numpy(), gt[0,:,0].numpy(), label="gt")
    plt.plot(times[0,:].numpy(), pos_2ord, label="2.",)
    plt.plot(times[0, :].numpy(),  pos_3ord,
             label= "3.")
    plt.legend()
    plt.show()





if __name__ == "__main__":
    set_traj()
    test_learn_trajs()