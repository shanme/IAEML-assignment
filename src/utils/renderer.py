import jax
import jax.numpy as jnp
import numpy as np
import pygame


class PygameFrontend:
    def __init__(self, env, env_params, init_state, eval_mode=False, agent_fn=None):
        pygame.init()
        self.env = env
        self.params = env_params
        self.init_state = init_state
        self.eval_mode = eval_mode
        self.agent_fn = agent_fn
        height, width = env_params.map_height_width
        self.cell_size = min(30, 1000 // width, 600 // height)
        self.screen = pygame.display.set_mode(
            (width * self.cell_size, height * self.cell_size + 68)
        )
        pygame.display.set_caption("Robotaxi: W/S acceleration, A/D steering")
        self.font = pygame.font.Font(None, 22)
        self.clock = pygame.time.Clock()
        self.key = jax.random.PRNGKey(0)
        self.obs, self.state = env.reset(self.key, env_params, init_state)
        self.info = {}
        self.action = jnp.zeros(2)
        self.running = True

    def pixel(self, position):
        height = self.params.map_height_width[0]
        x, y = np.asarray(position)
        return (
            round((float(x) + 0.5) * self.cell_size),
            round((height - float(y) - 0.5) * self.cell_size),
        )

    def draw(self):
        self.screen.fill((242, 244, 247))
        state = jax.device_get(self.state)
        radius = self.cell_size
        path = [self.pixel(p) for p in state.path_array]
        if len(path) > 1:
            pygame.draw.lines(self.screen, (165, 186, 214), False, path, 2)
        for position in state.static_obstacles:
            pygame.draw.circle(self.screen, (62, 69, 80), self.pixel(position), radius)
        for position in state.kinematic_obstacles:
            pygame.draw.circle(self.screen, (215, 78, 69), self.pixel(position), radius)
        pygame.draw.circle(self.screen, (42, 171, 104), self.pixel(state.goal_pos), radius)
        for ray in np.asarray(self.obs.collision_rays):
            pygame.draw.line(
                self.screen, (226, 180, 173),
                self.pixel(state.agent_pos), self.pixel(state.agent_pos + ray),
            )
        center = self.pixel(state.agent_pos)
        pygame.draw.circle(self.screen, (43, 100, 196), center, radius)
        pygame.draw.line(
            self.screen, (255, 230, 90), center,
            self.pixel(state.agent_pos + state.agent_forward_dir), 3,
        )
        speed = float(getattr(state, "speed", 0.0))
        action = np.asarray(self.action)
        lines = [
            f"Step {int(state.time)}   Speed {speed:.2f}   "
            f"Acceleration input {action[0]:+.2f}   Steering input {action[1]:+.2f}",
            "Blue: vehicle   Green: goal   Red: moving obstacle   Gray: static obstacle",
        ]
        if self.info:
            outcome = (
                "GOAL" if bool(self.info.get("goal_reached", False))
                else "COLLISION" if bool(self.info.get("collision", False))
                else "TIME LIMIT" if bool(self.info.get("truncated", False))
                else "RUNNING"
            )
            lines[1] += f"   {outcome}"
        map_height = self.params.map_height_width[0] * self.cell_size
        for index, line in enumerate(lines):
            self.screen.blit(
                self.font.render(line, True, (25, 34, 48)),
                (10, map_height + 8 + index * 26),
            )
        pygame.display.flip()

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        acceleration = int(keys[pygame.K_UP] or keys[pygame.K_w]) - int(
            keys[pygame.K_DOWN] or keys[pygame.K_s]
        )
        steering = int(keys[pygame.K_LEFT] or keys[pygame.K_a]) - int(
            keys[pygame.K_RIGHT] or keys[pygame.K_d]
        )
        return jnp.asarray([acceleration, steering], dtype=jnp.float32)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.action = (
                self.agent_fn(self.obs.as_vector())
                if self.eval_mode and self.agent_fn is not None
                else self.handle_keys()
            )
            self.obs, self.state, _, done, self.info = self.env.step(
                self.key, self.state, self.action, self.params
            )
            self.draw()
            if bool(done):
                self.key, reset_key = jax.random.split(self.key)
                self.obs, self.state = self.env.reset(reset_key, self.params, self.init_state)
            self.clock.tick(self.params.fps)
        pygame.quit()
