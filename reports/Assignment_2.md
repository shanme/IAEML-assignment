# Assignment 2: Reinforcement learning

**Author:** Vadim Poponnikov  
**Group:** AA-03  
**Email:** v.poponnikov@innopolis.university

## 1. Experimental setup and model

The experiment trains a policy to drive the vehicle from Assignment 1 to the goal. The policy receives the 26 observation values and outputs acceleration and steering inputs. An episode ends at the goal, on collision, or after 600 steps. The reward includes route progress, control penalties, and the final outcome.

The experiments use map 0. At reset, the starting position changes by up to 0.15 units, heading changes by up to 0.12 radians, and moving obstacles are advanced by a random time between zero and eight seconds. Evaluation uses separate random keys for 32 new resets on the same map. This tests small changes in the starting conditions, rather than driving on an unseen map.

The implementation uses [Proximal Policy Optimization (PPO), Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) in `src/algorithms/ppo.py`. There are two neural networks:

- The actor chooses acceleration and steering. It outputs the means of two Gaussian distributions, with learned standard deviations. During training, actions are sampled to allow exploration. Applying tanh keeps each input between -1 and 1. The probability calculation accounts for this transformation.
- The critic estimates the expected discounted return from an observation. This estimate helps judge whether an action led to a better result than expected.

Each network has two hidden layers of 64 units with tanh activations. The actor has two output means, and the critic has one output value. Evaluation uses tanh of the actor's means, so it does not add action sampling noise.

PPO first collects trajectories with the current policy. It then uses these observations, actions, and rewards for several training passes. Its clipped objective is:

$$
L=\mathbb{E}_t\left[\min\left(\rho_t A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right],
\qquad
\rho_t=\frac{\pi_\theta(u_t\mid o_t)}{\pi_{\mathrm{old}}(u_t\mid o_t)}.
$$

Here, $A_t$ is the estimated advantage of the action, and $\rho_t$ compares its probability under the updated and previous policies. Clipping reduces the incentive to make a large policy change from one batch. The critic is trained with a clipped value loss. A small exploration bonus uses the Gaussian entropy before tanh, rather than the exact entropy of the final bounded actions.

Advantage estimates use [generalized advantage estimation, Schulman et al.](https://arxiv.org/abs/1506.02438), with discount 0.99 and parameter 0.95. At a goal or collision, there is no next-state value. At a time limit, the critic estimates the value of the final observation before reset. Advantage calculation stops across every reset, so rewards from different episodes are not mixed.

| Setting | Value |
| --- | --- |
| Training seed | 7 |
| Parallel environments | 32 |
| Steps collected per environment | 128 per rollout |
| Total batch size | 4,096 transitions |
| Training budgets | 524,288 and 2,097,152 transitions |
| PPO passes per rollout | 4 |
| Minibatches per pass | 4, with 1,024 transitions each |
| Learning rate | 0.0003, using Adam |
| Discount / GAE parameter | 0.99 / 0.95 |
| PPO clipping value | 0.2 |
| Value loss / exploration coefficients | 0.5 / 0.001 |
| Gradient norm limit | 0.5 |

The two runs start from the same network weights and use the same settings. Only the training budget changes. The route from the environment helps through the observation and reward; there is no imitation-learning stage or hand-written teacher policy.

## 2. Computational optimizations

Both the environment and the networks use JAX. `vmap` runs the 32 environments as a batch. `lax.scan` handles repeated rollout steps, advantage calculation, and training passes. `jit` compiles these operations, reducing the amount of work handled by Python during training. These tools follow the approach described in the [JAX documentation](https://docs.jax.dev/en/latest/jit-compilation.html).

Arrays keep the same shapes during a run, and calculations use float32. The route is planned once when the environment is initialized. Observation scaling uses fixed ranges, so no running normalization statistics are needed. Training copies summary statistics to the CPU-side Python code once per update, rather than copying every observation.

The short run took 29.28 seconds, or about 17,908 transitions per second. The long run took 107.34 seconds, or about 19,537 transitions per second. These are recorded CPU training-loop times, including rollout and update compilation. They exclude installation, initial environment and network construction, evaluation, and video recording. Different hardware can give different times. The run used Python 3.12.13, JAX 0.12.0.1, and NumPy 2.5.4 on Windows 11; exact dependencies are in `requirements.txt`.

## 3. Reflection

The first training run showed that return alone does not indicate whether the policy reaches the goal. Mean evaluation return increased from about 2.08 to 112.06, but none of the 32 episodes reached the goal. Goal, collision, and timeout counts therefore need to be considered alongside return.

The training budget was then increased while the reward, seed, and PPO settings remained unchanged. The longer run reached the goal in all 32 evaluation episodes. Keeping the other settings fixed allowed the effect of additional training to be compared.

Episode endings required separate collision and timeout flags. A collision ends the task, while a timeout cuts off an episode that could otherwise continue. The critic's next-state value is calculated before reset and set to zero for a goal or collision. Loading the saved model was also checked against the recorded evaluation results.

Evaluation covered 32 resets on one map. These results do not establish performance on other maps or substantially different starting conditions. Further evaluation could use additional maps and a wider range of starts.

## 4. Results and reproduction

All policies are evaluated deterministically on the same 32 resets, using evaluation seed 10000.

| Policy | Goals | Collisions | Timeouts | Mean return ± standard deviation | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before training | 0/32 | 0/32 | 32/32 | 2.076 ± 0.647 | 600.00 |
| After 524,288 transitions | 0/32 | 0/32 | 32/32 | 112.057 ± 1.776 | 600.00 |
| After 2,097,152 transitions | 32/32 | 0/32 | 0/32 | 188.998 ± 0.279 | 150.97 |

The final episodes took 150–152 steps, or 15.0–15.2 simulated seconds. There were no collisions in this evaluation. The result comes from one training seed and one map, so it does not establish performance on unseen maps.

The submitted results are:

- [Trained policy video](../artifacts/ppo/trained.mp4): reaches the goal in 150 steps, with return 188.616.
- [Untrained policy video](../artifacts/ppo/untrained.mp4): times out after 600 steps, with return 1.9135.
- [Final model checkpoint](../artifacts/ppo/checkpoint.npz) and [untrained model](../artifacts/ppo/initial.npz).
- [Final evaluation](../artifacts/ppo/evaluation.json), [training log](../artifacts/ppo/metrics.csv), and [configuration](../artifacts/ppo/config.json).
- [Short-run evaluation](../artifacts/ppo_524288/evaluation.json), [checkpoint](../artifacts/ppo_524288/checkpoint.npz), [training log](../artifacts/ppo_524288/metrics.csv), and [configuration](../artifacts/ppo_524288/config.json).

The two videos use the same direct reset key, 10000. Batch evaluation splits its seed into separate keys, so its first episode is not the video episode. Reloading the final checkpoint reproduces the saved evaluation exactly on this machine. The checkpoint stores the model weights and settings needed for replay. It does not store Adam's optimizer state for resuming training exactly.

After installing the dependencies as described in the [README](../readme.md), run these commands from the repository root to evaluate the saved model and recreate the videos:

```powershell
.venv\Scripts\python.exe src/algorithms/ppo.py evaluate --checkpoint artifacts/ppo/checkpoint.npz --episodes 32 --seed 10000 --output artifacts/ppo/evaluation_reproduced.json
.venv\Scripts\python.exe src/main.py --mode policy --checkpoint artifacts/ppo/checkpoint.npz --headless --seed 10000 --record artifacts/ppo/trained.mp4
.venv\Scripts\python.exe src/main.py --mode policy --checkpoint artifacts/ppo/initial.npz --headless --seed 10000 --record artifacts/ppo/untrained.mp4
```

To repeat training from the beginning:

```powershell
.venv\Scripts\python.exe src/algorithms/ppo.py train --steps 524288 --num-envs 32 --rollout-steps 128 --seed 7 --eval-episodes 32 --output artifacts/ppo_524288
.venv\Scripts\python.exe src/algorithms/ppo.py train --steps 2097152 --num-envs 32 --rollout-steps 128 --seed 7 --eval-episodes 32 --output artifacts/ppo
```

Omit `--headless` to watch the replay in a window. On Linux or macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. Small numerical differences are possible across systems.
