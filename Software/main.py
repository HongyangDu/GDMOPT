# Import necessary libraries
import argparse
from parameter_gui import create_gui
import json
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_aigc_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
import warnings
import sys
from types import SimpleNamespace
import traceback

# Ignore warnings
warnings.filterwarnings('ignore')
class GUILogger:
    def __init__(self, update_func):
        self.update_func = update_func

    def write(self, message):
        self.update_func(message)

    def flush(self):
        pass

def get_device(requested_device):
    if torch.cuda.is_available() and 'cuda' in requested_device:
        return requested_device
    print(f"CUDA is not available. Using CPU instead of {requested_device}")
    return 'cpu'

def main(args, update_output, stop_training):
    # 重定向标准输出到 GUI
    sys.stdout = GUILogger(update_output)
    sys.stderr = GUILogger(update_output)

    print("Starting training with parameters:")
    print(json.dumps(args, indent=2))

    # 创建一个 SimpleNamespace 对象
    args_obj = SimpleNamespace(**args)

    # create environments
    env, train_envs, test_envs = make_aigc_env(args_obj.training_num, args_obj.test_num)
    args_obj.state_shape = env.observation_space.shape[0]
    args_obj.action_shape = env.action_space.n
    args_obj.max_action = 1.

    args_obj.exploration_noise = args_obj.exploration_noise * args_obj.max_action

    # create actor
    actor_net = MLP(
        state_dim=args_obj.state_shape,
        action_dim=args_obj.action_shape
    )
    # Actor is a Diffusion model
    actor = Diffusion(
        state_dim=args_obj.state_shape,
        action_dim=args_obj.action_shape,
        model=actor_net,
        max_action=args_obj.max_action,
        beta_schedule=args_obj.beta_schedule,
        n_timesteps=args_obj.n_timesteps,
        bc_coef=args_obj.bc_coef
    ).to(args_obj.device)
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args_obj.actor_lr,
        weight_decay=args_obj.wd
    )

    # Create critic
    critic = DoubleCritic(
        state_dim=args_obj.state_shape,
        action_dim=args_obj.action_shape
    ).to(args_obj.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args_obj.critic_lr,
        weight_decay=args_obj.wd
    )

    ## Setup logging
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args_obj.logdir, args_obj.log_prefix, "diffusion", time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args_obj))
    logger = TensorboardLogger(writer)

    # Define policy
    policy = DiffusionOPT(
        args_obj.state_shape,
        actor,
        actor_optim,
        args_obj.action_shape,
        critic,
        critic_optim,
        args_obj.device,
        tau=args_obj.tau,
        gamma=args_obj.gamma,
        estimation_step=args_obj.n_step,
        lr_decay=args_obj.lr_decay,
        lr_maxt=args_obj.epoch,
        bc_coef=args_obj.bc_coef,
        action_space=env.action_space,
        exploration_noise=args_obj.exploration_noise,
    )

    # Load a previous policy if a path is provided
    if args_obj.resume_path:
        ckpt = torch.load(args_obj.resume_path, map_location=args_obj.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args_obj.resume_path)

    # Setup buffer
    if args_obj.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args_obj.buffer_size,
            buffer_num=len(train_envs),
            alpha=args_obj.prior_alpha,
            beta=args_obj.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(
            args_obj.buffer_size,
            buffer_num=len(train_envs)
        )

    # Setup collector
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_callback(epoch: int, env_step: int, **kwargs):
        print(f"Epoch: {epoch}, Env Step: {env_step}")
        # Add additional training logic or logging here

    def stop_fn(reward, **kwargs):
        if stop_training():
            print(f"Training stopped by user. Best reward: {reward}")
            return True
        return False

    # Trainer
    if not args_obj.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args_obj.epoch,
            args_obj.step_per_epoch,
            args_obj.step_per_collect,
            args_obj.test_num,
            args_obj.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
            stop_fn=stop_fn,
            train_fn=train_callback
        )
        pprint.pprint(result)
    print("Training finished.")

    # Watch the performance
    if args_obj.watch:
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1)
        print(result)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")



if __name__ == '__main__':
    # root, on_submit, update_output, stop_training = create_gui()
    # root.mainloop()
    # args_dict = on_submit()
    #
    # args = SimpleNamespace(**args_dict)
    #
    # # Set default values for any missing parameters
    # default_args = {
    #     "exploration_noise": 0.1,
    #     "algorithm": "diffusion_opt",
    #     "seed": 1,
    #     "buffer_size": int(1e6),
    #     "epoch": int(1e6),
    #     "step_per_epoch": 1,
    #     "step_per_collect": 1,
    #     "batch_size": 512,
    #     "wd": 1e-4,
    #     "gamma": 1,
    #     "n_step": 3,
    #     "training_num": 1,
    #     "test_num": 1,
    #     "logdir": "log",
    #     "log_prefix": "default",
    #     "render": 0.1,
    #     "rew_norm": 0,
    #     "device": "cuda:0",
    #     "resume_path": None,
    #     "watch": False,
    #     "lr_decay": False,
    #     "note": "",
    #     "actor_lr": 1e-4,
    #     "critic_lr": 1e-4,
    #     "tau": 0.005,
    #     "n_timesteps": 6,
    #     "beta_schedule": "vp",
    #     "bc_coef": False,
    #     "prioritized_replay": False,
    #     "prior_alpha": 0.4,
    #     "prior_beta": 0.4
    # }
    # for key, value in default_args.items():
    #     if not hasattr(args, key):
    #         setattr(args, key, value)
    #
    # args.device = get_device(args.device)

    root, start_training, update_output, stop_training = create_gui()
    root.mainloop()
