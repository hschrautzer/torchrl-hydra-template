# PPO Implementation Notes

This document summarizes the PPO implementation in this repository and the
main design decisions made while translating the monolithic PPO examples into
the modular repo structure.

The goal was not to copy the monolith line by line. The goal was to preserve
the important PPO and Atari implementation details while fitting them into the
repo's three-part structure:

- `Environment`: task definition and preprocessing
- `Algorithm`: learning logic and hyperparameters
- `Trainer`: collection loop, logging, checkpointing, evaluation

## Starting Point

The implementation was built by comparing two references:

- `ppo_monolith.py`: simple PPO implementation for vector observations
- `ppo_monolith_atari.py`: Atari-specific PPO implementation for Breakout

The final repo-style implementation lives mainly in:

- `src/algorithms/ppo.py`
- `src/networks.py`
- `configs/algorithm/ppo.yaml`
- `configs/algorithm/ppo_atari.yaml`
- `configs/environment/breakout_train.yaml`
- `configs/environment/breakout_eval.yaml`

## High-Level PPO Idea

PPO learns from short batches of fresh rollout data.

For each rollout:

1. The current policy interacts with the environment.
2. The critic estimates how good each state is.
3. GAE computes advantages, meaning how much better or worse an action was than expected.
4. The same rollout is reused for several minibatch gradient updates.
5. The policy update is clipped so the policy does not change too much at once.
6. The critic is updated to better predict future returns.

PPO is still on-policy. The buffer in `ppo.py` is only a temporary rollout
buffer, not a long-term replay buffer like in DQN.

## Actor And Critic

The actor chooses actions. The critic estimates the value of the current state.

For simple environments such as CartPole, actor and critic are separate MLPs:

- actor: observation -> action logits
- critic: observation -> scalar value

For Atari, we chose a shared convolutional network:

- shared CNN encoder: pixels -> hidden features
- actor head: hidden features -> action logits
- critic head: hidden features -> scalar value

This matches the monolith Atari implementation and avoids running two separate
CNNs for the same image observation.

## Why `ProbabilisticActor` Is Used

TorchRL expects policies to read and write values inside a `TensorDict`.
The actor network itself only computes logits. `ProbabilisticActor` turns those
logits into a probability distribution and samples an action.

In our PPO implementation it writes:

- `action`: sampled action
- `sample_log_prob`: log probability of the sampled action

The stored log probability is important because PPO later compares the old
policy probability from rollout collection with the new policy probability
during training.

For CartPole and Atari we use `Categorical`, not `OneHotCategorical`, because
the action is represented as an integer class index.

## Critic With `ValueOperator`

The critic is wrapped in TorchRL's `ValueOperator`.

The critic reads the observation key and writes:

- `state_value`

This is the standard key expected by TorchRL's GAE module. Keeping this naming
convention avoids manual glue code in the PPO update.

## GAE

The monolith computes GAE explicitly with a backward loop over the rollout.
In this repo implementation, TorchRL's `GAE` module performs the same role.

It computes:

- `advantage`
- `value_target`
- `state_value`

We keep `average_gae=False` because advantage normalization is done later on
each minibatch, matching the PPO implementation details from the reference.

## PPO Update

The `step()` function in `src/algorithms/ppo.py` corresponds to one PPO update:

1. Anneal the learning rate.
2. Move the rollout batch to the training device.
3. Compute GAE.
4. Flatten the rollout.
5. Refill the temporary on-policy buffer.
6. Iterate over epochs and minibatches.
7. Recompute action log probabilities.
8. Compute the clipped policy loss.
9. Compute the clipped value loss.
10. Add entropy regularization.
11. Clip gradients.
12. Step the optimizer.

This keeps the trainer simple. The trainer only collects batches and calls
`algorithm.step(batch)`.

## Learning Rate Annealing

The monolith linearly decays the learning rate from its initial value to zero.
We kept that behavior in `ppo.py`.

The number of PPO updates is computed from:

```text
total_frames / frames_per_batch
```

This matters when resuming from checkpoints: `algorithm.total_frames` should
still refer to the full intended training length, otherwise the learning rate
schedule changes.

## Orthogonal Initialization

We added orthogonal initialization to match common PPO practice.

The important choices are:

- hidden layers use a larger gain suitable for ReLU or Tanh networks
- actor head uses a small gain (`0.01`)
- critic head uses gain `1.0`
- biases are initialized to zero

The small actor gain makes the initial policy close to uniform random, which is
useful for exploration.

## Atari Environment Design

For Breakout we translated the Atari preprocessing from the monolith into the
repo's environment configuration.

The training environment uses:

- `NoopResetEnv`: random no-ops at reset
- `MaxAndSkipEnv`: action repeat and max-pooling over frames
- `EpisodicLifeEnv`: life loss is treated as episode end during training
- `FireResetEnv`: presses FIRE after reset
- `ClipRewardEnv`: clips rewards to `-1`, `0`, or `1`
- image transforms: tensor conversion, grayscale, resize to `84x84`, frame stack

The eval environment is different:

- no `EpisodicLifeEnv`
- no reward clipping

This makes evaluation closer to real game scoring.

## Pixel Scaling

The monolith divides image observations by `255.0` inside the network because
the observations are raw image values.

TorchRL's transform stack already gives us normalized floating point pixels.
Therefore the Atari network config uses:

```yaml
scale_pixels: false
```

This was an important difference. Dividing by `255.0` again made the inputs too
small.

## Rollout Size And Minibatches

For monolith parity on Atari we use:

```text
num_envs = 8
num_steps = 128
frames_per_batch = 1024
num_minibatches = 4
minibatch_size = 256
num_epochs = 4
```

In the repo, `frames_per_batch` replaces the monolith's explicit
`num_envs * num_steps` batch size.

## Logging Rewards

One important lesson was that reward metrics can look different even when the
agent behaves similarly.

The monolith logs raw episodic return from Gym's episode statistics.
The repo originally logged TorchRL's `RewardSum`, which was affected by:

- reward clipping
- life-episode termination

To make comparison easier, we added raw episode logging through a custom Gym
wrapper:

- `RawEpisodeStatistics`

This exposes:

- `raw_episode_return`
- `raw_episode_length`
- `raw_episode_done`

The trainer then logs:

- `train/raw_episodic_return`
- `train/raw_episodic_length`

This is the metric to compare with the monolith's raw episodic return.

## Checkpointing

PPO checkpoints save:

- actor state
- critic state
- optimizer state
- PPO update counter

The update counter matters because learning rate annealing depends on how many
updates have already been completed.

We used periodic checkpoints so long Atari runs can be resumed and so different
agents can be rendered later as beginner, intermediate, and expert policies.

## Video Rendering

Videos are generated from saved checkpoints using the eval environment.

For video rendering we added a separate environment config:

- `breakout_eval_video.yaml`

It uses Gymnasium's `RecordVideo` wrapper.

Two video-specific issues came up:

1. After losing a life, Breakout waits for FIRE.
2. A deterministic policy can get stuck in loops.

For the first issue, we added:

- `FireAfterLifeLoss`

For the second issue, we discussed two options:

- cap video length with `StepCounter(max_steps=...)`
- render with stochastic policy sampling instead of deterministic mode

For fair score reporting, deterministic evaluation is preferable. For watchable
videos, stochastic rendering can be useful if the deterministic policy loops.

## Experiments Beyond Monolith Parity

Once parity with the monolith was no longer the main goal, we started treating
Breakout performance as an experimental question.

Useful experiment directions:

- train longer, for example `40M` frames instead of `10M`
- use sticky actions with `repeat_action_probability: 0.25`
- tune entropy coefficient, for example `0.02`
- compare clipped and unclipped rewards
- test no-progress timeouts only as a separate experiment

The current training config uses sticky actions in `breakout_train.yaml`:

```yaml
repeat_action_probability: 0.25
```

This makes exact action loops less reliable and can improve robustness.

## Summary Of Main Decisions

- PPO lives in `src/algorithms/ppo.py`, not in the trainer.
- The trainer only collects data and calls `step()`.
- TorchRL's `ProbabilisticActor` is used to sample actions and store log probs.
- TorchRL's `GAE` replaces the manual backward advantage loop.
- Atari uses a shared CNN actor-critic trunk.
- Pixel scaling is disabled for Atari because TorchRL already normalizes pixels.
- Reward logging was separated into clipped training reward and raw episodic return.
- Evaluation and video rendering use separate environment configs.
- Later experiments intentionally moved beyond strict monolith parity.

