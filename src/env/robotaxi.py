from functools import partial
import math

import equinox as eqx
import jax
import jax.numpy as jnp

from env.base import BaseEnv, BaseEnvState
from env.tasks import BaseTaskSampler


class RobotaxiParams(eqx.Module):
    max_steps_in_episode: int = 600
    fps: int = 10
    step_size: float = 0.1
    map_height_width: tuple = (30, 60)
    num_ray_sensors: int = 16
    agent_FOV: float = math.pi
    perception_radius: float = 10.0
    wheelbase: float = 1.5
    max_speed: float = 4.0
    max_acceleration: float = 3.0
    max_steering: float = 0.6
    linear_drag: float = 0.15
    quadratic_drag: float = 0.03
    physics_substeps: int = 4
    map_scale: int = 3
    path_length: int = 100


class RobotaxiState(BaseEnvState):
    speed: jax.Array
    previous_action: jax.Array
    episode_done: jax.Array


class RobotaxiObservation(eqx.Module):
    distance_to_path: jax.Array
    direction_of_path: jax.Array
    collision_rays: jax.Array
    vector: jax.Array

    def as_vector(self):
        return self.vector


class RobotaxiEnv(BaseEnv):
    def init_params(
        key,
        map_id=0,
        max_steps=600,
        path_length=100,
        discretization_scale=1,
        perception_radius=10.0,
        num_ray_sensors=16,
        fov=math.pi,
        fps=10,
    ):
        if discretization_scale != 1:
            raise ValueError("Only discretization_scale=1 is supported.")
        if max_steps < 1 or path_length < 2 or fps < 1 or num_ray_sensors < 4:
            raise ValueError("Invalid episode length, path length, fps, or ray count.")
        if perception_radius <= 1.0 or not 0 < fov <= math.pi:
            raise ValueError("Invalid perception radius or field of view.")
        task = BaseTaskSampler(map_id)
        scale = 3
        params = RobotaxiParams(
            max_steps_in_episode=max_steps,
            fps=fps,
            step_size=1.0 / fps,
            map_height_width=(task.height * scale, task.width * scale),
            num_ray_sensors=int(num_ray_sensors),
            agent_FOV=float(fov),
            perception_radius=float(perception_radius),
            path_length=path_length,
        )

        def to_world(positions):
            positions = jnp.asarray(positions, dtype=jnp.float32)
            return jnp.stack(
                [(positions[..., 1] + 0.5) * scale,
                 (task.height - positions[..., 0] - 0.5) * scale], axis=-1
            )

        start = to_world(task.agent_pos)
        goal = to_world(task.goal_pos)
        static = to_world(task.static_obstacles)
        path = RobotaxiEnv._find_path(start, goal, static, params)
        if float(jnp.linalg.norm(path[-2] - goal)) > 1.5:
            raise ValueError("No complete route: increase path_length or choose a reachable map.")
        direction = path[min(4, path_length - 1)] - path[0]
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-6)
        state = RobotaxiState(
            time=jnp.asarray(0, dtype=jnp.int32),
            goal_pos=goal,
            agent_pos=start,
            agent_forward_dir=direction,
            static_obstacles=static,
            kinematic_obstacles=to_world(task.kinematic_obstacles),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(jnp.float32),
            path_array=path,
            speed=jnp.asarray(0.0, dtype=jnp.float32),
            previous_action=jnp.zeros(2, dtype=jnp.float32),
            episode_done=jnp.asarray(False),
        )
        return params, state

    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        key_position, key_heading, key_phase = jax.random.split(key, 3)
        phase = jax.random.uniform(key_phase, (), minval=0.0, maxval=8.0)
        bounds = jnp.array([env_params.map_height_width[1], env_params.map_height_width[0]])
        span = bounds - 2.0
        unfolded = init_state.kinematic_obstacles - 1.0 + phase * init_state.kinematic_obst_velocities
        wrapped = jnp.mod(unfolded, 2.0 * span)
        moving = 1.0 + jnp.where(wrapped <= span, wrapped, 2.0 * span - wrapped)
        velocities = jnp.where(wrapped <= span, init_state.kinematic_obst_velocities, -init_state.kinematic_obst_velocities)
        candidate = init_state.agent_pos + jax.random.uniform(
            key_position, (2,), minval=-0.15, maxval=0.15
        )
        obstacles = jnp.concatenate(
            [init_state.static_obstacles, moving], axis=0
        )
        safe = jnp.all(jnp.linalg.norm(candidate - obstacles, axis=-1) > 2.1)
        position = jnp.where(safe, candidate, init_state.agent_pos)
        angle = jnp.arctan2(init_state.agent_forward_dir[1], init_state.agent_forward_dir[0])
        angle += jax.random.uniform(key_heading, (), minval=-0.12, maxval=0.12)
        state = eqx.tree_at(
            lambda s: (s.agent_pos, s.agent_forward_dir, s.speed, s.previous_action,
                       s.time, s.episode_done, s.kinematic_obstacles, s.kinematic_obst_velocities),
            init_state,
            (position, jnp.array([jnp.cos(angle), jnp.sin(angle)]),
             jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros(2, dtype=jnp.float32),
             jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False), moving, velocities),
        )
        return RobotaxiEnv.get_observation(state, env_params), state

    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
        action = jnp.where(env_state.episode_done, env_state.previous_action, action)
        dt = env_params.step_size / env_params.physics_substeps
        heading = jnp.arctan2(env_state.agent_forward_dir[1], env_state.agent_forward_dir[0])
        bounds = jnp.array([env_params.map_height_width[1], env_params.map_height_width[0]])

        def integrate(_, carry):
            position, angle, speed, moving, velocities, collided = carry
            acceleration = (
                action[0] * env_params.max_acceleration
                - env_params.linear_drag * speed
                - env_params.quadratic_drag * speed * jnp.abs(speed)
            )
            new_speed = jnp.clip(speed + acceleration * dt, 0.0, env_params.max_speed)
            yaw_rate = new_speed * jnp.tan(action[1] * env_params.max_steering) / env_params.wheelbase
            new_angle = angle + yaw_rate * dt
            mid_angle = angle + 0.5 * yaw_rate * dt
            new_position = position + new_speed * jnp.array([jnp.cos(mid_angle), jnp.sin(mid_angle)]) * dt
            proposed_moving = moving + velocities * dt
            reflected = (proposed_moving < 1.0) | (proposed_moving > bounds - 1.0)
            new_velocities = jnp.where(reflected, -velocities, velocities)
            new_moving = jnp.clip(proposed_moving, 1.0, bounds - 1.0)
            old_obstacles = jnp.concatenate([env_state.static_obstacles, moving], axis=0)
            new_obstacles = jnp.concatenate([env_state.static_obstacles, new_moving], axis=0)
            relative_start = position - old_obstacles
            relative_delta = (new_position - new_obstacles) - relative_start
            fraction = jnp.clip(
                -jnp.sum(relative_start * relative_delta, axis=-1)
                / jnp.maximum(jnp.sum(relative_delta ** 2, axis=-1), 1e-10), 0.0, 1.0
            )
            closest = relative_start + fraction[:, None] * relative_delta
            hit = jnp.any(jnp.sum(closest ** 2, axis=-1) <= 4.0)
            hit |= jnp.any((new_position < 1.0) | (new_position > bounds - 1.0))
            active = ~collided & ~env_state.episode_done
            return (
                jnp.where(active, new_position, position),
                jnp.where(active, new_angle, angle),
                jnp.where(active, new_speed, speed),
                jnp.where(active, new_moving, moving),
                jnp.where(active, new_velocities, velocities),
                collided | (hit & active),
            )

        position, angle, speed, moving, velocities, collision = jax.lax.fori_loop(
            0, env_params.physics_substeps, integrate,
            (env_state.agent_pos, heading, env_state.speed,
             env_state.kinematic_obstacles, env_state.kinematic_obst_velocities,
             jnp.asarray(False)),
        )
        new_time = env_state.time + (~env_state.episode_done).astype(jnp.int32)
        goal = (jnp.linalg.norm(position - env_state.goal_pos) <= 1.0) & ~collision
        terminated = goal | collision
        truncated = (new_time >= env_params.max_steps_in_episode) & ~terminated
        done = env_state.episode_done | terminated | truncated
        state = eqx.tree_at(
            lambda s: (s.agent_pos, s.agent_forward_dir, s.speed, s.previous_action,
                       s.time, s.kinematic_obstacles, s.kinematic_obst_velocities, s.episode_done),
            env_state,
            (position, jnp.array([jnp.cos(angle), jnp.sin(angle)]), speed, action,
             new_time, moving, velocities, done),
        )
        old_remaining, _, _, _ = RobotaxiEnv._path_features(env_state)
        remaining, cross_track, _, _ = RobotaxiEnv._path_features(state)
        progress = old_remaining - remaining
        reward = (
            2.0 * progress - 0.01
            - 0.005 * cross_track ** 2
            - 0.005 * jnp.sum(action ** 2)
            - 0.002 * jnp.sum((action - env_state.previous_action) ** 2)
            + 50.0 * goal.astype(jnp.float32)
            - 50.0 * collision.astype(jnp.float32)
        )
        reward = jnp.where(env_state.episode_done, 0.0, reward)
        obs = RobotaxiEnv.get_observation(state, env_params)
        info = {
            "time": new_time,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "goal_reached": goal & ~env_state.episode_done,
            "collision": collision & ~env_state.episode_done,
            "terminated": terminated & ~env_state.episode_done,
            "truncated": truncated & ~env_state.episode_done,
            "route_remaining": remaining,
            "speed": speed,
        }
        return obs, state, reward, done, info

    def _path_features(state):
        starts = state.path_array[:-1]
        segments = state.path_array[1:] - starts
        lengths = jnp.linalg.norm(segments, axis=-1)
        relative = state.agent_pos - starts
        fractions = jnp.clip(jnp.sum(relative * segments, axis=-1) / jnp.maximum(lengths ** 2, 1e-8), 0.0, 1.0)
        projections = starts + fractions[:, None] * segments
        distances = jnp.linalg.norm(state.agent_pos - projections, axis=-1)
        index = jnp.argmin(jnp.where(lengths > 1e-6, distances, jnp.inf))
        cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(lengths)])
        progress = cumulative[index] + fractions[index] * lengths[index]
        remaining = jnp.maximum(cumulative[-1] - progress, 0.0)
        target_distance = jnp.minimum(progress + 3.5, cumulative[-1])
        target_index = jnp.clip(jnp.searchsorted(cumulative, target_distance, side="right") - 1, 0, lengths.shape[0] - 1)
        target_fraction = jnp.clip((target_distance - cumulative[target_index]) / jnp.maximum(lengths[target_index], 1e-8), 0.0, 1.0)
        target = starts[target_index] + target_fraction * segments[target_index]
        tangent = segments[index] / jnp.maximum(lengths[index], 1e-8)
        offset = state.agent_pos - projections[index]
        cross_track = tangent[0] * offset[1] - tangent[1] * offset[0]
        return remaining, cross_track, target, distances[index]

    def get_observation(env_state, env_params):
        remaining, cross_track, target, distance_to_path = RobotaxiEnv._path_features(env_state)
        forward = env_state.agent_forward_dir
        left = jnp.array([-forward[1], forward[0]])

        def to_body(vector):
            return jnp.array([jnp.dot(vector, forward), jnp.dot(vector, left)])

        obstacles = jnp.concatenate([env_state.static_obstacles, env_state.kinematic_obstacles], axis=0)
        rays, distances = BaseEnv._collision_ray_intersections(
            env_state.agent_pos, forward, obstacles, env_params
        )
        goal_relative = to_body(env_state.goal_pos - env_state.agent_pos)
        target_vector = target - env_state.agent_pos
        direction = target_vector / jnp.maximum(jnp.linalg.norm(target_vector), 1e-8)
        total_length = jnp.sum(jnp.linalg.norm(jnp.diff(env_state.path_array, axis=0), axis=-1))
        diagonal = math.hypot(*env_params.map_height_width)
        vector = jnp.concatenate([
            distances / env_params.perception_radius,
            jnp.atleast_1d(env_state.speed / env_params.max_speed),
            env_state.previous_action,
            goal_relative / diagonal,
            jnp.clip(to_body(target_vector) / env_params.perception_radius, -1.0, 1.0),
            jnp.atleast_1d(jnp.clip(cross_track / env_params.perception_radius, -1.0, 1.0)),
            jnp.atleast_1d(remaining / jnp.maximum(total_length, 1e-8)),
            jnp.atleast_1d(env_state.time / env_params.max_steps_in_episode),
        ]).astype(jnp.float32)
        return RobotaxiObservation(distance_to_path, direction, rays * distances[:, None], vector)

    @partial(jax.jit, static_argnames=("env_params",))
    def _find_path(agent_pos, goal_pos, obstacles, env_params):
        height, width = env_params.map_height_width
        yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
        grid = jnp.stack([xx, yy], axis=-1).astype(jnp.float32)
        clearance = jnp.min(jnp.linalg.norm(grid[:, :, None, :] - obstacles[None, None, :, :], axis=-1), axis=-1)
        blocked = (clearance < 2.15) | (xx < 1) | (xx > width - 1) | (yy < 1) | (yy > height - 1)
        goal = jnp.clip(jnp.round(goal_pos).astype(jnp.int32), jnp.array([1, 1]), jnp.array([width - 2, height - 2]))
        values = jnp.full((height, width), 1e6, dtype=jnp.float32).at[goal[1], goal[0]].set(0.0)

        def update(_, current):
            padded = jnp.pad(current, 1, constant_values=1e6)
            neighbors = jnp.stack([
                padded[:-2, 1:-1], padded[2:, 1:-1],
                padded[1:-1, :-2], padded[1:-1, 2:],
            ], axis=-1)
            best = jnp.minimum(current, jnp.min(neighbors, axis=-1) + 1.0)
            return jnp.where(blocked, 1e6, best)

        values = jax.lax.fori_loop(0, height * width, update, values)
        start = jnp.clip(jnp.round(agent_pos).astype(jnp.int32), jnp.array([1, 1]), jnp.array([width - 2, height - 2]))
        moves = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], dtype=jnp.int32)

        def trace(position, _):
            candidates = jnp.clip(position + moves, jnp.array([0, 0]), jnp.array([width - 1, height - 1]))
            scores = values[candidates[:, 1], candidates[:, 0]]
            next_position = candidates[jnp.argmin(scores)]
            return next_position, next_position.astype(jnp.float32)

        _, path = jax.lax.scan(trace, start, None, length=env_params.path_length - 2)
        path = jnp.concatenate([agent_pos[None, :], path, goal_pos[None, :]], axis=0)
        return path
