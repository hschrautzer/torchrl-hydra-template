import torch
import torch.nn as nn
from torchrl.envs import GymEnv
from torchrl.modules import QValueActor, EGreedyModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict.nn import TensorDictModule, TensorDictSequential

torch.manual_seed(0)

env = GymEnv("CartPole-v1", categorical_action_encoding=True)

qnet = TensorDictModule(
    nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2)),
    in_keys=["observation"],
    out_keys=["action_value"],
)
policy = QValueActor(qnet, in_keys=["observation"], spec=env.action_spec)

exploration_policy = EGreedyModule(spec=env.action_spec, eps_init=1.0, eps_end=0.05, annealing_num_steps=50_000)
exploration_module = TensorDictSequential(policy, exploration_policy)

collector = SyncDataCollector(
    create_env_fn=lambda: GymEnv("CartPole-v1", categorical_action_encoding=True),
    policy=exploration_module,
    frames_per_batch=1000,
    total_frames=500_000,
)

buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=100_000))
loss_fn = DQNLoss(policy, action_space=env.action_spec)
loss_fn.make_value_estimator(gamma=0.99)
updater = SoftUpdate(loss_fn, tau=0.001)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

total_frames = 0
best_avg_reward = 0
for i, batch in enumerate(collector):
    buffer.extend(batch)
    total_frames += batch.numel()

    if len(buffer) < 10000:
        continue

    for _ in range(10):
        loss = loss_fn(buffer.sample(256))
        optimizer.zero_grad()
        loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        updater.step()

    exploration_policy.step(frames=batch.numel())

    if i % 10 == 0:
        rewards = []
        with torch.no_grad():
            for _ in range(5):
                rewards.append(env.rollout(max_steps=500, policy=policy)["next", "reward"].sum().item())
        avg_reward = sum(rewards) / 5
        best_avg_reward = max(best_avg_reward, avg_reward)
        print(f"Step {total_frames}: loss={loss['loss'].item():.3f}, reward={avg_reward:.1f}, eps={exploration_policy.eps:.3f}")
        if avg_reward >= 490:
            print(f"Goal achieved at {total_frames} frames!")
            break

collector.shutdown()
env.close()

# Final evaluation: 5 episodes with learned policy
final_env = GymEnv("CartPole-v1", categorical_action_encoding=True)
final_rewards = []
with torch.no_grad():
    for _ in range(5):
        final_rewards.append(final_env.rollout(max_steps=500, policy=policy)["next", "reward"].sum().item())
final_env.close()
final_avg = sum(final_rewards) / 5
print(f"METRIC: {final_avg:.1f}")