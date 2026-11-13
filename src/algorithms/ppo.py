import argparse
import csv
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.robotaxi import RobotaxiEnv


@dataclass(frozen=True)
class PPOConfig:
    steps: int = 524288
    num_envs: int = 32
    rollout_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 0.5
    hidden_sizes: tuple = (64, 64)
    seed: int = 7
    map_id: int = 0
    max_steps: int = 600
    eval_episodes: int = 16
    eval_seed: int = 10000


def initialize_network(key, observation_size, hidden_sizes=(64, 64)):
    def network(layer_key, output_size, output_scale):
        sizes = (observation_size, *hidden_sizes, output_size)
        keys = jax.random.split(layer_key, len(sizes) - 1)
        layers = []
        for index, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = output_scale if index == len(sizes) - 2 else np.sqrt(2.0)
            weight = jax.nn.initializers.orthogonal(scale)(
                keys[index], (in_size, out_size), jnp.float32
            )
            layers.append({"w": weight, "b": jnp.zeros(out_size, jnp.float32)})
        return tuple(layers)

    actor_key, critic_key = jax.random.split(key)
    return {
        "actor": network(actor_key, 2, 0.01),
        "critic": network(critic_key, 1, 1.0),
        "log_std": jnp.full((2,), -0.5, jnp.float32),
    }


def network_forward(layers, observation):
    value = observation
    for layer in layers[:-1]:
        value = jnp.tanh(value @ layer["w"] + layer["b"])
    return value @ layers[-1]["w"] + layers[-1]["b"]


def actor_critic(weights, observation):
    mean = network_forward(weights["actor"], observation)
    log_std = jnp.clip(weights["log_std"], -5.0, 1.0)
    value = network_forward(weights["critic"], observation)[..., 0]
    return mean, log_std, value


def squashed_log_probability(raw_action, mean, log_std):
    normal = -0.5 * (
        jnp.square((raw_action - mean) * jnp.exp(-log_std))
        + 2.0 * log_std
        + jnp.log(2.0 * jnp.pi)
    )
    log_jacobian = 2.0 * (
        jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)
    )
    return jnp.sum(normal - log_jacobian, axis=-1)


@partial(jax.jit, static_argnames=("deterministic",))
def policy_action(weights, observation, key=None, deterministic=True):
    mean, log_std, _ = actor_critic(weights, observation)
    if deterministic:
        return jnp.tanh(mean)
    if key is None:
        raise ValueError("Stochastic actions require a PRNG key.")
    return jnp.tanh(mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape))


def _environment_metadata(params):
    values = {}
    for field_name in params.__dataclass_fields__:
        value = getattr(params, field_name)
        if isinstance(value, (str, bool, int, float)):
            values[field_name] = value
        elif isinstance(value, tuple):
            if all(isinstance(item, (str, bool, int, float)) for item in value):
                values[field_name] = list(value)
    return values


def save_checkpoint(path, weights, metadata):
    arrays = {"metadata": np.array(json.dumps(metadata)), "log_std": np.asarray(weights["log_std"])}
    for name in ("actor", "critic"):
        for index, layer in enumerate(weights[name]):
            for field_name in ("w", "b"):
                arrays[f"{name}_{index}_{field_name}"] = np.asarray(layer[field_name])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"].item()))
        number_layers = len(metadata["config"]["hidden_sizes"]) + 1
        weights = {"log_std": jnp.asarray(arrays["log_std"])}
        for name in ("actor", "critic"):
            weights[name] = tuple(
                {field_name: jnp.asarray(arrays[f"{name}_{index}_{field_name}"])
                 for field_name in ("w", "b")}
                for index in range(number_layers)
            )
    return weights, metadata


def environment_from_metadata(metadata):
    config = metadata["config"]
    saved = metadata.get("environment", {})
    arguments = {
        name: saved[field]
        for name, field in (
            ("fps", "fps"), ("path_length", "path_length"),
            ("perception_radius", "perception_radius"),
            ("num_ray_sensors", "num_ray_sensors"), ("fov", "agent_FOV"),
        )
        if field in saved
    }
    params, state = RobotaxiEnv.init_params(
        jax.random.PRNGKey(config["seed"]),
        map_id=config["map_id"], max_steps=config["max_steps"],
        **arguments,
    )
    actual = _environment_metadata(params)
    mismatched = [name for name, value in saved.items() if actual.get(name) != value]
    if mismatched:
        raise ValueError("Environment settings differ from the checkpoint: " + ", ".join(mismatched))
    return params, state


def _select_states(mask, selected, otherwise):
    return jax.tree.map(
        lambda yes, no: jnp.where(mask.reshape(mask.shape + (1,) * (yes.ndim - mask.ndim)), yes, no),
        selected, otherwise,
    )


def make_training_functions(config, env_params, init_state):
    batch_size = config.num_envs * config.rollout_steps
    minibatch_size = batch_size // config.minibatches
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )
    reset_batch = jax.vmap(lambda key: RobotaxiEnv.reset(key, env_params, init_state))
    step_batch = jax.vmap(lambda key, state, action: RobotaxiEnv.step(key, state, action, env_params))

    @jax.jit
    def reset(key):
        observations, states = reset_batch(jax.random.split(key, config.num_envs))
        return jax.vmap(lambda observation: observation.as_vector())(observations), states

    @jax.jit
    def collect(weights, carry):
        def step(carry, _):
            observation, state, key, episode_return, episode_length = carry
            key, action_key, step_key, reset_key = jax.random.split(key, 4)
            mean, log_std, value = actor_critic(weights, observation)
            raw_action = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
            log_probability = squashed_log_probability(raw_action, mean, log_std)
            next_observation, next_state, reward, done, info = step_batch(
                jax.random.split(step_key, config.num_envs), state, jnp.tanh(raw_action)
            )
            next_vector = jax.vmap(lambda obs: obs.as_vector())(next_observation)
            _, _, next_value = actor_critic(weights, next_vector)
            terminated = info["terminated"]
            next_value = jnp.where(terminated, 0.0, next_value)
            episode_return = episode_return + reward
            episode_length = episode_length + 1
            transition = {
                "observation": observation,
                "raw_action": raw_action,
                "log_probability": log_probability,
                "value": value,
                "reward": reward,
                "done": done,
                "next_value": next_value,
                "finished_return": jnp.where(done, episode_return, 0.0),
                "finished_length": jnp.where(done, episode_length, 0),
                "success": done & info["goal_reached"],
                "collision": done & info["collision"],
                "timeout": done & info["truncated"],
            }
            reset_observations, reset_states = reset_batch(jax.random.split(reset_key, config.num_envs))
            reset_vectors = jax.vmap(lambda obs: obs.as_vector())(reset_observations)
            next_vector = jnp.where(done[:, None], reset_vectors, next_vector)
            next_state = _select_states(done, reset_states, next_state)
            return (
                next_vector, next_state, key,
                jnp.where(done, 0.0, episode_return),
                jnp.where(done, 0, episode_length),
            ), transition

        carry, transitions = jax.lax.scan(step, carry, None, length=config.rollout_steps)

        def advantage_step(advantage, transition):
            delta = transition["reward"] + config.gamma * transition["next_value"] - transition["value"]
            advantage = delta + config.gamma * config.gae_lambda * (~transition["done"]) * advantage
            return advantage, advantage

        _, advantages = jax.lax.scan(
            advantage_step, jnp.zeros(config.num_envs), transitions, reverse=True
        )
        transitions["target"] = advantages + transitions["value"]
        transitions["advantage"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return carry, transitions

    def loss(weights, batch):
        mean, log_std, values = actor_critic(weights, batch["observation"])
        log_probability = squashed_log_probability(batch["raw_action"], mean, log_std)
        log_ratio = log_probability - batch["log_probability"]
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        policy_loss = -jnp.minimum(ratio * batch["advantage"], clipped_ratio * batch["advantage"]).mean()
        clipped_values = batch["value"] + jnp.clip(values - batch["value"], -config.clip_epsilon, config.clip_epsilon)
        value_loss = 0.5 * jnp.maximum(
            jnp.square(values - batch["target"]),
            jnp.square(clipped_values - batch["target"]),
        ).mean()
        latent_entropy = jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)))
        total_loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * latent_entropy
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (jnp.abs(ratio - 1.0) > config.clip_epsilon).mean()
        return total_loss, jnp.array([policy_loss, value_loss, latent_entropy, approximate_kl, clip_fraction])

    @jax.jit
    def update(weights, optimizer_state, transitions, key):
        names = ("observation", "raw_action", "log_probability", "value", "target", "advantage")
        flat_batch = {name: transitions[name].reshape((batch_size,) + transitions[name].shape[2:]) for name in names}

        def epoch(carry, _):
            weights, optimizer_state, key = carry
            key, shuffle_key = jax.random.split(key)
            indices = jax.random.permutation(shuffle_key, batch_size)
            shuffled = jax.tree.map(
                lambda values: values[indices].reshape((config.minibatches, minibatch_size) + values.shape[1:]),
                flat_batch,
            )

            def minibatch(carry, batch):
                weights, optimizer_state = carry
                (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(weights, batch)
                updates, optimizer_state = optimizer.update(gradients, optimizer_state, weights)
                return (optax.apply_updates(weights, updates), optimizer_state), metrics

            (weights, optimizer_state), metrics = jax.lax.scan(minibatch, (weights, optimizer_state), shuffled)
            return (weights, optimizer_state, key), metrics.mean(axis=0)

        (weights, optimizer_state, _), metrics = jax.lax.scan(
            epoch, (weights, optimizer_state, key), None, length=config.epochs
        )
        finished = transitions["done"].sum()
        statistics = jnp.array([
            finished, transitions["success"].sum(), transitions["collision"].sum(),
            transitions["timeout"].sum(), transitions["finished_return"].sum(),
            transitions["finished_length"].sum(),
        ])
        return weights, optimizer_state, metrics.mean(axis=0), statistics

    return optimizer, reset, collect, update


@partial(jax.jit, static_argnames=("env_params", "episodes", "deterministic"))
def _evaluate_batch(weights, env_params, init_state, key, episodes, deterministic):
    reset_key, key = jax.random.split(key)
    observations, states = jax.vmap(lambda subkey: RobotaxiEnv.reset(subkey, env_params, init_state))(
        jax.random.split(reset_key, episodes)
    )
    vectors = jax.vmap(lambda observation: observation.as_vector())(observations)

    def step(carry, _):
        states, vectors, key, finished, returns, lengths, success, collision, timeout = carry
        key, action_key, step_key = jax.random.split(key, 3)
        action = policy_action(weights, vectors, action_key, deterministic)
        next_observations, next_states, reward, done, info = jax.vmap(
            lambda subkey, state, one_action: RobotaxiEnv.step(subkey, state, one_action, env_params)
        )(jax.random.split(step_key, episodes), states, action)
        active = ~finished
        returns = returns + jnp.where(active, reward, 0.0)
        lengths = lengths + active.astype(jnp.int32)
        success = success | (active & done & info["goal_reached"])
        collision = collision | (active & done & info["collision"])
        timeout = timeout | (active & done & info["truncated"])
        vectors = jax.vmap(lambda observation: observation.as_vector())(next_observations)
        return (next_states, vectors, key, finished | done, returns, lengths, success, collision, timeout), None

    zeros = jnp.zeros(episodes)
    flags = jnp.zeros(episodes, dtype=bool)
    carry = (states, vectors, key, flags, zeros, jnp.zeros(episodes, jnp.int32), flags, flags, flags)
    carry, _ = jax.lax.scan(step, carry, None, length=env_params.max_steps_in_episode)
    return carry[4:]


def evaluate_policy(weights, env_params, init_state, episodes=16, seed=10000, deterministic=True):
    returns, lengths, success, collisions, timeouts = jax.device_get(_evaluate_batch(
        weights, env_params, init_state, jax.random.PRNGKey(seed), episodes, deterministic
    ))
    return {
        "episodes": episodes,
        "reset_seed": seed,
        "deterministic": deterministic,
        "success_rate": float(success.mean()),
        "collision_rate": float(collisions.mean()),
        "timeout_rate": float(timeouts.mean()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_episode_steps": float(lengths.mean()),
        "episode_returns": returns.tolist(),
        "episode_steps": lengths.tolist(),
        "episode_success": success.tolist(),
        "episode_collision": collisions.tolist(),
        "episode_timeout": timeouts.tolist(),
    }


def train(config, output):
    batch_size = config.num_envs * config.rollout_steps
    if min(config.num_envs, config.rollout_steps, config.epochs, config.minibatches, config.eval_episodes) < 1:
        raise ValueError("Environment, rollout, optimization, and evaluation counts must be positive.")
    if config.steps < batch_size or batch_size % config.minibatches:
        raise ValueError("Steps must include at least one rollout, and batch size must divide into minibatches.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(config.seed)
    key, init_key, network_key, reset_key = jax.random.split(key, 4)
    env_params, init_state = RobotaxiEnv.init_params(init_key, map_id=config.map_id, max_steps=config.max_steps)
    observation, _ = RobotaxiEnv.reset(reset_key, env_params, init_state)
    observation_size = observation.as_vector().shape[0]
    weights = initialize_network(network_key, observation_size, config.hidden_sizes)
    initial_weights = weights
    metadata = {
        "format_version": 1,
        "algorithm": "PPO with GAE and tanh Gaussian policy",
        "config": asdict(config),
        "observation_size": observation_size,
        "action_size": 2,
        "observation_normalization": "Fixed scaling supplied by RobotaxiObservation.as_vector; no fitted statistics.",
        "environment": _environment_metadata(env_params),
        "actual_training_steps": 0,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
    }
    save_checkpoint(output / "initial.npz", initial_weights, metadata)
    optimizer, reset, collect, update = make_training_functions(config, env_params, init_state)
    optimizer_state = optimizer.init(weights)
    vectors, states = reset(reset_key)
    carry = (vectors, states, key, jnp.zeros(config.num_envs), jnp.zeros(config.num_envs, jnp.int32))
    updates = config.steps // batch_size
    started = time.perf_counter()
    completed_episodes = 0
    successes = 0
    collisions = 0
    timeouts = 0
    fields = [
        "update", "steps", "elapsed_seconds", "steps_per_second", "episodes", "successes", "collisions", "timeouts",
        "completed_mean_return", "completed_mean_steps", "policy_loss", "value_loss", "latent_entropy", "approx_kl", "clip_fraction",
    ]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(updates):
            carry, transitions = collect(weights, carry)
            key, update_key = jax.random.split(carry[2])
            carry = (carry[0], carry[1], key, carry[3], carry[4])
            weights, optimizer_state, losses, statistics = update(weights, optimizer_state, transitions, update_key)
            losses, statistics = jax.device_get((losses, statistics))
            count, new_successes, new_collisions, new_timeouts, sum_returns, sum_lengths = statistics.tolist()
            completed_episodes += int(count)
            successes += int(new_successes)
            collisions += int(new_collisions)
            timeouts += int(new_timeouts)
            elapsed = time.perf_counter() - started
            row = dict(zip(fields, [
                index + 1, (index + 1) * batch_size, elapsed, (index + 1) * batch_size / elapsed,
                completed_episodes, successes, collisions, timeouts,
                sum_returns / max(count, 1), sum_lengths / max(count, 1), *losses.tolist(),
            ]))
            writer.writerow(row)
            handle.flush()
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == updates:
                metadata["actual_training_steps"] = (index + 1) * batch_size
                save_checkpoint(output / "checkpoint.npz", weights, metadata)
                print(json.dumps(row), flush=True)
    metadata["actual_training_steps"] = updates * batch_size
    metadata["training_seconds_including_compilation"] = time.perf_counter() - started
    save_checkpoint(output / "checkpoint.npz", weights, metadata)
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    results = {
        "trained": evaluate_policy(weights, env_params, init_state, config.eval_episodes, config.eval_seed),
        "untrained": evaluate_policy(initial_weights, env_params, init_state, config.eval_episodes, config.eval_seed),
        "scope": "Held-out reset seeds on the training map; no claim of generalization to unseen maps.",
    }
    with (output / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"trained": results["trained"], "untrained": results["untrained"]}), flush=True)
    return weights, metadata


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--steps", type=int, default=PPOConfig.steps)
    training.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    training.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    training.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    training.add_argument("--minibatches", type=int, default=PPOConfig.minibatches)
    training.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    training.add_argument("--seed", type=int, default=PPOConfig.seed)
    training.add_argument("--map-id", type=int, default=PPOConfig.map_id)
    training.add_argument("--max-steps", type=int, default=PPOConfig.max_steps)
    training.add_argument("--eval-episodes", type=int, default=PPOConfig.eval_episodes)
    training.add_argument("--output", default="artifacts/ppo")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--episodes", type=int, default=32)
    evaluation.add_argument("--seed", type=int, default=10000)
    evaluation.add_argument("--stochastic", action="store_true")
    evaluation.add_argument("--output", default="artifacts/ppo/evaluation_reproduced.json")
    args = parser.parse_args()
    if args.command == "train":
        arguments = vars(args).copy()
        arguments.pop("command")
        output = arguments.pop("output")
        train(PPOConfig(**arguments), output)
    else:
        weights, metadata = load_checkpoint(args.checkpoint)
        env_params, init_state = environment_from_metadata(metadata)
        result = evaluate_policy(weights, env_params, init_state, args.episodes, args.seed, not args.stochastic)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
