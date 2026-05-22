"""
This module is currently a one-file implementation of Huang's 2022 Blog Post and video series. Will be refactored
later to fit into the structure of this repo.
"""
import argparse
import os
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda t: t % 100 == 0)
        # Deprecated in modern gymnasium versions. Use envs.reset(seed)
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std = np.sqrt(2), bias_const = 0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1),std=1)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        # at 14:10
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment.")
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',help="the id of the gym env.")
    parser.add_argument('--learning-rate',type=float, default=2.5e-4, help="learning rate of optimizer")
    parser.add_argument("--seed",type=int,default=1,help="seed of experiment")
    parser.add_argument("--total-timesteps",type=int,default=25000, help="total timesteps of exp.")
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda',type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?',const=True, help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track',type=lambda x:bool(strtobool(x)), default=False,
                        nargs='?',const=True, help='if toggled, the experiment is tracked with wandb')
    parser.add_argument('--wandb-project-name',type=str, default='CleanRL',help="wandb project name")
    parser.add_argument('--wandb-entity',type=str,default=None,help='the entity (team) of wandbs project')
    parser.add_argument('--capture-video',type=lambda x:bool(strtobool(x)), default=False,
                        nargs='?',const=True,help='if video shall be recorded.')

    # Algorithm specific args
    parser.add_argument('--n-envs',type=int,default=4, help='number of environments')
    parser.add_argument('--num-steps',type=int,default=128, help='the number of steps to run in each env per policy rollout')
    args = parser.parse_args()
    args.batch_size = int(args.n_envs * args.num_steps)
    return args

if __name__=="__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparamers",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # Try not to modify SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Device setup
    device = torch.device("cude" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id,seed=args.seed + i,idx=i,capture_video=args.capture_video,
                                              run_name=run_name ) for i in range(args.n_envs)])
    assert isinstance(envs.single_action_space,gym.spaces.Discrete), "only discrete Action Spaces supported."
    print(f"envs.single_observation_space.shape: {envs.single_observation_space.shape}")
    print(f"envs.single_action_space.n: {envs.single_action_space.n}")

    agent = Agent(envs=envs).to(device)
    optimizer = optim.Adam(agent.parameters(),lr=args.learning_rate,eps=1e-5)

    # Setup storage
    obs = torch.zeros((args.num_steps, args.n_envs) + envs.single_observation_space.shape).to(device) # use tuple addition -> eg. (num_steps, n_envs, 4)
    actions = torch.zeros((args.num_steps, args.n_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.n_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.n_envs)).to(device)
    truncations = torch.zeros((args.num_steps, args.n_envs)).to(device)
    terminations = torch.zeros((args.num_steps, args.n_envs)).to(device)
    values = torch.zeros((args.num_steps, args.n_envs)).to(device)

    # DO not modify
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset(seed=[args.seed + i for i in range(args.n_envs)])[0]).to(device)
    next_done = torch.zeros(args.n_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
