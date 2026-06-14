"""Network factories used by ``configs/algorithm/*.yaml``.

Each factory takes ``(obs_shape, action_dim)`` positionally and keeps the
rest as keyword-only args, so a Hydra ``_partial_`` config can pre-bind the
kwargs while the algorithm's ``setup()`` supplies the runtime shape and
action count. ``action_dim`` is the discrete action count for value-based
algorithms (DQN) and the continuous action vector size for actor/critic
algorithms (DDPG).
"""
from __future__ import annotations

import math
from typing import Sequence, Type
import torch
import torch.nn as nn
from torchrl.modules import ConvNet, MLP

# ================== All of the following are just helper classes for factor initialization below ==========

class NaturePPOActorCritic(nn.Module):
    """ Shared Nature-DQN-style CNN network with PPO actor and critic heads.
    Considers Implementation Details [8,9] (see ppo_monolith_atari.py)
    """
    @staticmethod
    def _layer_init_conv_ppo(layer: nn.Conv2d,*,std: float = math.sqrt(2),bias_const: float = 0.0) -> nn.Conv2d:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    @staticmethod
    def _layer_init_linear_ppo(layer: nn.Linear,*,std: float = math.sqrt(2),bias_const: float = 0.0) -> nn.Linear:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def __init__(self,obs_shape: Sequence[int],num_actions: int,*,num_cells_cnn: Sequence[int] = (32, 64, 64),
                 kernel_sizes: Sequence[int] = (8, 4, 3),strides: Sequence[int] = (4, 2, 1),hidden_features: int = 512,
                 activation_class: Type[nn.Module] = nn.ReLU,scale_pixels: bool = True) -> None:
        super().__init__()
        self.scale_pixels = scale_pixels

        self.cnn_network: nn.Module = nn.Sequential(
            self._layer_init_conv_ppo(
                nn.Conv2d(
                    in_channels=int(obs_shape[0]),
                    out_channels=int(num_cells_cnn[0]),
                    kernel_size=int(kernel_sizes[0]),
                    stride=int(strides[0]),
                )
            ),
            activation_class(),
            self._layer_init_conv_ppo(
                nn.Conv2d(
                    in_channels=int(num_cells_cnn[0]),
                    out_channels=int(num_cells_cnn[1]),
                    kernel_size=int(kernel_sizes[1]),
                    stride=int(strides[1]),
                )
            ),
            activation_class(),
            self._layer_init_conv_ppo(
                nn.Conv2d(
                    in_channels=int(num_cells_cnn[1]),
                    out_channels=int(num_cells_cnn[2]),
                    kernel_size=int(kernel_sizes[2]),
                    stride=int(strides[2]),
                )
            ),
            activation_class(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_network_out = self.cnn_network(torch.zeros(1, *obs_shape))

        self.hidden = nn.Sequential(
            self._layer_init_linear_ppo(
                nn.Linear(cnn_network_out.shape[-1], hidden_features),
                std=math.sqrt(2),
            ),
            activation_class(),
        )
        self.actor = self._layer_init_linear_ppo(
            nn.Linear(hidden_features, num_actions),
            std=0.01,
        )
        self.critic = self._layer_init_linear_ppo(
            nn.Linear(hidden_features, 1),
            std=1.0,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        # receives single unbatched Atari observation with shape [4,84,84]
        if self.scale_pixels:
            obs = obs / 255.0

        batch_shape = obs.shape[:-3]
        obs = obs.reshape(-1,*obs.shape[-3:]) # now [1,4,84,84]
        cnn_output = self.cnn_network(obs) # [1,3136]
        return self.hidden(cnn_output) # [512]

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.encode(obs))

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(obs))

class PPOActorHead(nn.Module):
    def __init__(self, actor_critic: NaturePPOActorCritic) -> None:
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        return self.actor_critic.forward_actor(obs)

class PPOCriticHead(nn.Module):
    def __init__(self, actor_critic: NaturePPOActorCritic) -> None:
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_critic.forward_critic(obs)

# =========================================

# Factor function for PPO Atari:

def NaturePPOSharedActorCritic(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells_cnn: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
    hidden_features: int = 512,
    activation_class: Type[nn.Module] = nn.ReLU,
    scale_pixels: bool = True,
) -> tuple[nn.Module, nn.Module]:
    actor_critic = NaturePPOActorCritic(
        obs_shape,
        num_actions,
        num_cells_cnn=num_cells_cnn,
        kernel_sizes=kernel_sizes,
        strides=strides,
        hidden_features=hidden_features,
        activation_class=activation_class,
        scale_pixels=scale_pixels,
    )
    return PPOActorHead(actor_critic), PPOCriticHead(actor_critic)

def _layer_init_mlp_ppo(module: nn.Module,*,final_weight_std: float,
                        hidden_weight_std: float = math.sqrt(2),bias_const: float = 0.0) -> nn.Module:
    """
    Implementation detail [2] of PPO. Orthogonal initialization of weights and constant initialization of biases.
    """
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]

    for layer in linear_layers[:-1]:
        nn.init.orthogonal_(layer.weight, hidden_weight_std)
        nn.init.constant_(layer.bias, bias_const)

    if linear_layers:
        final_layer = linear_layers[-1]
        nn.init.orthogonal_(final_layer.weight, final_weight_std)
        nn.init.constant_(final_layer.bias, bias_const)

    return module

def make_mlp_ppo_actor(
        obs_shape: Sequence[int],
        num_actions: int,
        *, # all arguments after this must be passed by keyword
        num_cells: Sequence[int],
        activation_class: Type[nn.Module] = nn.Tanh,
) -> nn.Module:
    """
    Creates the multilayer perceptron (MLP) for the actor of PPO. MLP is the torchRL substitute of torch's
    nn.Sequential(...). However, the num_cells attribute makes it easier to add more layers without extra code.

    Args:
        obs_shape: the shape of a single observation space of the vect. environment.
        num_actions: the number of actions (single_action_space.n).
        num_cells: the length of the sequence is the number of hidden layers and the values of the sequence
            are the number of cells for the layers.
        activation_class: the type of activation function. Default tanh for PPO implementation.
    Returns:
        the actor network, with orthogonal initialization of weights and constant initialization of biases.
    """
    mlp: nn.Module = MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=num_actions,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )
    return _layer_init_mlp_ppo(module=mlp, final_weight_std=0.01)


def make_mlp_ppo_critic(
        obs_shape: Sequence[int],
        num_actions: int,
        *, # all arguments after this must be passed by keyword
        num_cells: Sequence[int],
        activation_class: Type[nn.Module] = nn.Tanh,
) -> nn.Module:
    """
    Creates the multilayer perceptron (MLP) for the critic of PPO.

    Args:
        obs_shape: the shape of a single observation space of the vect. environment.
        num_cells: the length of the sequence is the number of hidden layers and the values of the sequence
            are the number of cells for the layers.
        activation_class: the type of activation function. Default tanh for PPO implementation.
    Returns:
        the critic network, mapping observations to a single scalar V(s).
    """
    del num_actions
    mlp: nn.Module = MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )
    return _layer_init_mlp_ppo(module=mlp, final_weight_std=1.0)



def make_mlp_q_net(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """Plain MLP Q-network. Flattens ``obs_shape`` to ``in_features``."""
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=num_actions,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_ddpg_actor(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for a DDPG deterministic actor.

    Returns an MLP mapping the flattened observation to ``action_dim``
    unbounded outputs. The algorithm wraps this with ``TanhModule`` to
    rescale to the action spec, so this factory must NOT apply tanh itself.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=action_dim,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_ddpg_critic(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for a DDPG state-action value (critic).

    Returns an MLP mapping the concatenated ``[obs, action]`` vector to a
    single Q-value. ``ValueOperator`` concatenates inputs along the last
    dim before calling the module.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)) + int(action_dim),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_a2c_actor(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for an A2C stochastic actor.

    Returns an MLP mapping the flattened observation to ``2 * action_dim``
    outputs. The algorithm chains it with ``NormalParamExtractor`` to split
    the output into ``loc`` and (positive) ``scale`` for a TanhNormal policy.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=2 * int(action_dim),
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_a2c_value(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for an A2C state-value critic.

    Takes ``(obs_shape, action_dim)`` for signature parity with the actor
    factory; ``action_dim`` is unused — the critic estimates V(s) only.
    Returns an MLP mapping the flattened observation to a single value.
    """
    del action_dim  # signature parity with actor factory
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def NatureDQN(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells_cnn: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
    num_cells_mlp: Sequence[int] = (512,),
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """ConvNet -> MLP Q-network from Mnih et al. 2015 (\"Nature DQN\")."""
    return _nature_cnn_mlp(
        obs_shape=obs_shape,
        out_features=num_actions,
        num_cells_cnn=num_cells_cnn,
        kernel_sizes=kernel_sizes,
        strides=strides,
        num_cells_mlp=num_cells_mlp,
        activation_class=activation_class,
    )


def _nature_cnn_mlp(
    *,
    obs_shape: Sequence[int],
    out_features: int,
    num_cells_cnn: Sequence[int],
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    num_cells_mlp: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """Shared ConvNet -> MLP builder for Atari-style pixel observations."""
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells=list(num_cells_cnn),
        kernel_sizes=list(kernel_sizes),
        strides=list(strides),
    )
    with torch.no_grad():
        cnn_out = cnn(torch.zeros(1, *obs_shape))
    mlp = MLP(
        in_features=cnn_out.shape[-1],
        out_features=out_features,
        num_cells=list(num_cells_mlp),
        activation_class=activation_class,
    )
    return nn.Sequential(cnn, mlp)
