from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


def make_network(
    cfg: DictConfig,
    obs_shape: tuple[int, ...],
    out_features: int,
) -> nn.Module:
    """Build a neural network from the algorithm's nested network config.

    Args:
        cfg: algorithm_cfg.network (nested DictConfig with architecture key)
        obs_shape: observation shape as tuple, e.g. (4,) or (4, 84, 84)
        out_features: output size (action dim for actor, 1 for critic)

    Returns:
        nn.Module (not yet wrapped in a TensorDictModule)
    """
    arch = cfg.get("architecture", "mlp")

    if arch == "mlp":
        in_features = math.prod(obs_shape)
        return MLP(
            in_features=in_features,
            hidden_sizes=list(cfg.hidden_sizes),
            out_features=out_features,
            activation=cfg.get("activation", "relu"),
            layer_norm=cfg.get("layer_norm", False),
        )

    if arch == "cnn_atari":
        return AtariCNN(
            obs_shape=obs_shape,
            conv_channels=list(cfg.conv_channels),
            conv_kernels=list(cfg.conv_kernels),
            conv_strides=list(cfg.conv_strides),
            fc_hidden=list(cfg.fc_hidden),
            out_features=out_features,
            activation=cfg.get("activation", "relu"),
        )

    raise ValueError(
        f"Unknown network architecture '{arch}'. Choose from: 'mlp', 'cnn_atari'."
    )


class MLP(nn.Module):
    """Multi-layer perceptron with optional LayerNorm.

    Args:
        in_features: input dimension (flattened)
        hidden_sizes: list of hidden layer widths
        out_features: output dimension
        activation: activation function name
        layer_norm: if True, add LayerNorm after each hidden activation
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int],
        out_features: int,
        activation: str = "relu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = in_features
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            if layer_norm:
                layers.append(nn.LayerNorm(size))
            layers.append(act_cls())
            prev = size
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)


class AtariCNN(nn.Module):
    """Nature DQN convolutional network for Atari pixel observations.

    Input shape: (batch, C*frame_stack, H, W) — e.g. (batch, 4, 84, 84)

    Args:
        obs_shape: (C, H, W) observation shape after preprocessing
        conv_channels: output channels for each conv layer
        conv_kernels: kernel sizes for each conv layer
        conv_strides: strides for each conv layer
        fc_hidden: hidden sizes for the fully-connected head
        out_features: final output dimension
        activation: activation function name
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        conv_channels: list[int],
        conv_kernels: list[int],
        conv_strides: list[int],
        fc_hidden: list[int],
        out_features: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS[activation]

        in_c = obs_shape[0]
        conv_layers: list[nn.Module] = []
        for out_c, k, s in zip(conv_channels, conv_kernels, conv_strides):
            conv_layers += [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s), act_cls()]
            in_c = out_c
        self.conv = nn.Sequential(*conv_layers)

        # compute flattened size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            flat = self.conv(dummy).flatten(1).shape[1]

        fc_layers: list[nn.Module] = []
        prev = flat
        for h in fc_hidden:
            fc_layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        fc_layers.append(nn.Linear(prev, out_features))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, H, W) — handle batch dims including unbatched 3D input
        batch_shape = x.shape[:-3]
        # Merge all leading dims into a single batch dim for Conv2d
        x = x.reshape(-1, *x.shape[-3:])
        x = self.conv(x).flatten(1)
        x = self.fc(x)
        if batch_shape:
            return x.unflatten(0, batch_shape)
        return x.squeeze(0)
