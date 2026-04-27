# Claude Code instructions for torchrl-hydra-template

See `AGENTS.md` for the full codebase guide. This file adds Claude-specific notes.

## Maintenance rule

**Always update `README.md` and `AGENTS.md`** when changing a public API, adding an algorithm, renaming a class, or changing a convention. README targets human readers; AGENTS.md targets AI agents.

## Key patterns (quick reference)

### Algorithm HPs — explicit constructor kwargs, not dataclasses

```python
class MyAlgorithm(BaseAlgorithm):
    def __init__(self, cfg, device=None, *, lr=3e-4, gamma=0.99, network=None):
        super().__init__(cfg, device)
        self.lr = lr
        self.gamma = gamma
        self._network_cfg = network or {"architecture": "mlp", ...}
```

- Use `*` to make all HPs keyword-only.
- Nested dict groups (`network`, `replay_buffer`) default to `None`; apply body-level fallbacks.
- Wrap before passing to `make_network`: `OmegaConf.create(self._network_cfg)`.
- No `self.acfg`, no `_build_acfg`, no config dataclasses.

### Instantiation in `train.py`

```python
alg_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.algorithm, resolve=True).items()
              if k != "_target_"}
algorithm = AlgClass(cfg=cfg, device=None, **alg_kwargs)

env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
              if k != "_target_"}
environment = Environment(**env_kwargs)
```

### YAML convention

```yaml
_target_: src.algorithms.my_algo.MyAlgorithm
# Default values and parameter descriptions: src/algorithms/my_algo.py (MyAlgorithm.__init__)

lr: 3e-4
network:
  architecture: mlp
  hidden_sizes: [256, 256]
  activation: tanh
  layer_norm: false
```

## What not to do

- Do not create `XxxConfig` dataclasses for algorithm HPs.
- Do not use `self.acfg` or `_build_acfg`.
- Do not pass `cfg.environment` directly to `Environment()` — unpack it as `**kwargs`.
- Do not add `OmegaConf` imports to `base.py` — it has no config-merging logic anymore.
