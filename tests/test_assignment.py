from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algorithms.ppo import (
    PPOConfig,
    initialize_network,
    load_checkpoint,
    make_training_functions,
    policy_action,
    save_checkpoint,
    squashed_log_probability,
)
from env.robotaxi import RobotaxiEnv, RobotaxiParams, RobotaxiState


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = jax.random.PRNGKey(42)
        cls.params = RobotaxiParams()
        cls.state = RobotaxiState(
            time=jnp.asarray(0, dtype=jnp.int32),
            goal_pos=jnp.array([40.0, 10.0]),
            agent_pos=jnp.array([10.0, 10.0]),
            agent_forward_dir=jnp.array([1.0, 0.0]),
            static_obstacles=jnp.array([[50.0, 25.0]]),
            kinematic_obstacles=jnp.array([[45.0, 24.0]]),
            kinematic_obst_velocities=jnp.zeros((1, 2)),
            path_array=jnp.stack([jnp.linspace(10.0, 40.0, 100), jnp.full(100, 10.0)], axis=1),
            speed=jnp.asarray(0.0),
            previous_action=jnp.zeros(2),
            episode_done=jnp.asarray(False),
        )

    def step(self, state=None, action=(0.0, 0.0), params=None):
        return RobotaxiEnv.step(
            self.key, self.state if state is None else state,
            jnp.asarray(action), self.params if params is None else params,
        )

    def assert_tree_close(self, first, second):
        first_leaves, first_structure = jax.tree.flatten(first)
        second_leaves, second_structure = jax.tree.flatten(second)
        self.assertEqual(first_structure, second_structure)
        for first_leaf, second_leaf in zip(first_leaves, second_leaves):
            np.testing.assert_allclose(first_leaf, second_leaf, rtol=1e-5, atol=1e-6)

    def test_stationary_and_acceleration(self):
        _, resting, _, done, _ = self.step()
        np.testing.assert_array_equal(resting.agent_pos, self.state.agent_pos)
        self.assertEqual(float(resting.speed), 0.0)
        self.assertFalse(bool(done))
        _, accelerating, _, done, _ = self.step(action=(1.0, 0.0))
        self.assertGreater(float(accelerating.speed), 0.0)
        self.assertGreater(float(accelerating.agent_pos[0]), float(self.state.agent_pos[0]))
        self.assertFalse(bool(done))

    def test_coasting_drag_and_braking(self):
        moving = eqx.tree_at(lambda state: state.speed, self.state, jnp.asarray(2.0))
        _, coasted, _, _, _ = self.step(moving)
        _, braked, _, _, _ = self.step(moving, (-1.0, 0.0))
        self.assertLess(float(coasted.speed), 2.0)
        self.assertLess(float(braked.speed), float(coasted.speed))
        stopped = self.step(action=(-1.0, 0.0))[1]
        self.assertEqual(float(stopped.speed), 0.0)

    def test_steering_requires_speed_and_turns_left(self):
        resting = self.step(action=(0.0, 1.0))[1]
        np.testing.assert_allclose(resting.agent_forward_dir, self.state.agent_forward_dir)
        moving = eqx.tree_at(lambda state: state.speed, self.state, jnp.asarray(2.0))
        turned = self.step(moving, (0.0, 1.0))[1]
        self.assertGreater(float(turned.agent_forward_dir[1]), 0.0)
        self.assertGreater(float(turned.agent_pos[1]), float(moving.agent_pos[1]))
        self.assertAlmostEqual(float(jnp.linalg.norm(turned.agent_forward_dir)), 1.0, places=5)

    def test_action_and_speed_bounds(self):
        moving = eqx.tree_at(lambda state: state.speed, self.state, jnp.asarray(self.params.max_speed))
        excessive = self.step(moving, (50.0, -50.0))[1]
        clipped = self.step(moving, (1.0, -1.0))[1]
        self.assert_tree_close(excessive, clipped)
        self.assertLessEqual(float(excessive.speed), self.params.max_speed)

    def test_sum_of_circle_radii(self):
        for separation, expected_collision in ((2.1, False), (2.0, True), (1.9, True)):
            with self.subTest(separation=separation):
                state = eqx.tree_at(
                    lambda state: state.static_obstacles, self.state,
                    jnp.array([[10.0 + separation, 10.0]]),
                )
                _, _, _, done, info = self.step(state)
                self.assertEqual(bool(info["collision"]), expected_collision)
                self.assertEqual(bool(done), expected_collision)

    def test_swept_collision_between_separated_endpoints(self):
        params = replace(
            self.params, step_size=0.5, physics_substeps=1,
            max_speed=20.0, linear_drag=0.0, quadratic_drag=0.0,
        )
        state = eqx.tree_at(
            lambda state: (state.speed, state.static_obstacles), self.state,
            (jnp.asarray(20.0), jnp.array([[15.0, 10.0]])),
        )
        _, next_state, _, done, info = self.step(state, params=params)
        self.assertGreater(float(jnp.linalg.norm(state.agent_pos - state.static_obstacles[0])), 2.0)
        self.assertGreater(float(jnp.linalg.norm(next_state.agent_pos - state.static_obstacles[0])), 2.0)
        self.assertTrue(bool(info["collision"]))
        self.assertTrue(bool(done))

    def test_goal_reward_and_absorbing_terminal(self):
        state = eqx.tree_at(lambda state: state.goal_pos, self.state, jnp.array([10.5, 10.0]))
        _, terminal, reward, done, info = self.step(state)
        self.assertTrue(bool(done))
        self.assertTrue(bool(info["goal_reached"]))
        self.assertTrue(bool(info["terminated"]))
        self.assertFalse(bool(info["truncated"]))
        self.assertGreater(float(reward), 40.0)
        _, unchanged, reward, done, info = self.step(terminal, (1.0, 1.0))
        self.assert_tree_close(terminal, unchanged)
        self.assertTrue(bool(done))
        self.assertEqual(float(reward), 0.0)
        self.assertFalse(bool(info["goal_reached"]))

    def test_timeout_bootstrap_flag(self):
        params = replace(self.params, max_steps_in_episode=1)
        _, terminal, _, done, info = self.step(params=params)
        self.assertEqual(int(terminal.time), 1)
        self.assertTrue(bool(done))
        self.assertTrue(bool(info["truncated"]))
        self.assertFalse(bool(info["terminated"]))
        self.assertFalse(bool(info["collision"]))

    def test_absorbing_timeout_has_no_new_terminal_event(self):
        params = replace(self.params, max_steps_in_episode=1)
        _, terminal, _, done, info = self.step(params=params)
        self.assertTrue(bool(done))
        self.assertTrue(bool(info["truncated"]))
        _, unchanged, reward, done, info = self.step(terminal, (1.0, 1.0), params)
        self.assert_tree_close(terminal, unchanged)
        self.assertEqual(float(reward), 0.0)
        self.assertTrue(bool(done))
        for flag in ("terminated", "truncated", "goal_reached", "collision"):
            self.assertFalse(bool(info[flag]), flag)

    def test_observations_and_jit_vmap_consistency(self):
        keys = jax.random.split(self.key, 3)
        reset = lambda key: RobotaxiEnv.reset(key, self.params, self.state)
        observations, states = jax.jit(jax.vmap(reset))(keys)
        vector = np.asarray(observations.as_vector())
        self.assertEqual(vector.shape, (3, self.params.num_ray_sensors + 10))
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue((np.abs(vector) <= 1.00001).all())
        actions = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        batched = jax.jit(jax.vmap(
            lambda key, state, action: RobotaxiEnv.step(key, state, action, self.params)
        ))(keys, states, actions)
        for index in range(3):
            scalar_state = jax.tree.map(lambda values: values[index], states)
            scalar = RobotaxiEnv.step(keys[index], scalar_state, actions[index], self.params)
            one_batched = jax.tree.map(lambda values: values[index], batched)
            self.assert_tree_close(scalar, one_batched)

    def test_all_map_routes_have_collision_clearance(self):
        for map_id in range(3):
            with self.subTest(map_id=map_id):
                _, state = RobotaxiEnv.init_params(self.key, map_id=map_id)
                path = np.asarray(state.path_array)
                np.testing.assert_allclose(path[0], state.agent_pos)
                np.testing.assert_allclose(path[-1], state.goal_pos)
                starts = path[:-1]
                deltas = path[1:] - starts
                obstacles = np.asarray(state.static_obstacles)
                relative = obstacles[None, :, :] - starts[:, None, :]
                fraction = np.clip(
                    np.sum(relative * deltas[:, None, :], axis=-1)
                    / np.maximum(np.sum(deltas ** 2, axis=-1)[:, None], 1e-10), 0.0, 1.0,
                )
                closest = starts[:, None, :] + fraction[:, :, None] * deltas[:, None, :]
                clearance = np.linalg.norm(closest - obstacles[None, :, :], axis=-1)
                self.assertGreaterEqual(float(clearance.min()), 2.0)
                self.assertLessEqual(float(np.linalg.norm(path[-2] - path[-1])), 1.5)

    def test_checkpoint_round_trip_preserves_actions(self):
        observation = RobotaxiEnv.get_observation(self.state, self.params).as_vector()
        weights = initialize_network(self.key, observation.shape[0])
        metadata = {"config": asdict(PPOConfig()), "observation_size": observation.shape[0]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, weights, metadata)
            restored, restored_metadata = load_checkpoint(path)
        self.assertEqual(restored_metadata["config"]["seed"], metadata["config"]["seed"])
        self.assert_tree_close(weights, restored)
        np.testing.assert_array_equal(policy_action(weights, observation), policy_action(restored, observation))

    def test_squashed_policy_density_is_normalized(self):
        sample_count = 20000
        spacing = 2.0 / sample_count
        bounded_actions = -1.0 + spacing * (jnp.arange(sample_count) + 0.5)
        raw_actions = jnp.arctanh(bounded_actions)[:, None]
        density = jnp.exp(squashed_log_probability(raw_actions, jnp.array([0.4]), jnp.array([-0.5])))
        self.assertAlmostEqual(float(density.sum() * spacing), 1.0, places=4)
        extreme = jnp.array([-30.0, 30.0])
        value, gradient = jax.value_and_grad(
            lambda mean: squashed_log_probability(extreme, mean, jnp.zeros(2))
        )(jnp.zeros(2))
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertTrue(bool(jnp.isfinite(gradient).all()))

    def test_gae_distinguishes_goal_and_timeout(self):
        config = replace(PPOConfig(), num_envs=2, rollout_steps=2, epochs=1, minibatches=1, max_steps=1)
        params = replace(self.params, max_steps_in_episode=1)
        goal_state = eqx.tree_at(lambda state: state.goal_pos, self.state, self.state.agent_pos)
        states = jax.tree.map(lambda first, second: jnp.stack([first, second]), goal_state, self.state)
        observations = jax.vmap(lambda state: RobotaxiEnv.get_observation(state, params).as_vector())(states)
        weights = initialize_network(self.key, observations.shape[-1])
        critic = tuple({name: jnp.zeros_like(value) for name, value in layer.items()} for layer in weights["critic"])
        critic[-1]["b"] = jnp.full_like(critic[-1]["b"], 2.0)
        weights = {**weights, "critic": critic}
        _, _, collect, _ = make_training_functions(config, params, self.state)
        carry = (observations, states, self.key, jnp.zeros(2), jnp.zeros(2, dtype=jnp.int32))
        _, transitions = collect(weights, carry)
        self.assertTrue(bool(transitions["done"].all()))
        self.assertTrue(bool(transitions["success"][0, 0]))
        self.assertTrue(bool(transitions["timeout"][0, 1]))
        expected_bootstrap = np.array([[0.0, 2.0], [2.0, 2.0]])
        np.testing.assert_allclose(transitions["next_value"], expected_bootstrap)
        np.testing.assert_allclose(
            transitions["target"], transitions["reward"] + config.gamma * expected_bootstrap,
            rtol=1e-6, atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
