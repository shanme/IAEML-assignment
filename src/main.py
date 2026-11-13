import argparse
import os
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiEnv


def demonstration_action(step):
    if step < 20:
        return jnp.asarray([0.65, 0.0])
    if step < 40:
        return jnp.asarray([0.0, 0.0])
    if step < 55:
        return jnp.asarray([0.0, -0.35])
    if step < 70:
        return jnp.asarray([0.0, 0.35])
    return jnp.asarray([-0.7, 0.0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manual", "dynamics", "policy"], default="manual")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--map-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    from utils.renderer import PygameFrontend
    import pygame

    key = jax.random.PRNGKey(args.seed)
    env_options = {"map_id": args.map_id, "max_steps": args.steps}
    policy = None
    if args.mode == "policy":
        if args.checkpoint is None:
            parser.error("--mode policy requires --checkpoint")
        from algorithms.ppo import environment_from_metadata, load_checkpoint, policy_action

        weights, metadata = load_checkpoint(args.checkpoint)
        policy = jax.jit(lambda observation: policy_action(weights, observation, deterministic=True))
    if policy is None:
        params, initial = RobotaxiEnv.init_params(key, **env_options)
    else:
        params, initial = environment_from_metadata(metadata)
    frontend = PygameFrontend(RobotaxiEnv, params, initial, agent_fn=policy, eval_mode=policy is not None)
    frontend.key = key
    frontend.obs, frontend.state = RobotaxiEnv.reset(key, params, initial)
    if args.mode == "manual" and args.record is None:
        frontend.run()
        return

    writer = None
    if args.record:
        ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE") or shutil.which("ffmpeg")
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        process_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        encoders = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
            check=True, creationflags=process_flags,
        ).stdout
        codec = "libx264" if "libx264" in encoders else "libopenh264"
        args.record.parent.mkdir(parents=True, exist_ok=True)
        width, height = frontend.screen.get_size()
        writer = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(params.fps), "-i", "pipe:0",
             "-an", "-c:v", codec, "-pix_fmt", "yuv420p", str(args.record)],
            stdin=subprocess.PIPE,
            creationflags=process_flags,
        )
    total_return = 0.0
    try:
        for step in range(args.steps):
            pygame.event.pump()
            frontend.action = (
                policy(frontend.obs.as_vector()) if policy is not None
                else demonstration_action(step) if args.mode == "dynamics"
                else frontend.handle_keys()
            )
            frontend.obs, frontend.state, reward, done, frontend.info = RobotaxiEnv.step(
                key, frontend.state, frontend.action, params
            )
            total_return += float(reward)
            frontend.draw()
            if writer is not None:
                frame = np.transpose(pygame.surfarray.array3d(frontend.screen), (1, 0, 2))
                writer.stdin.write(frame.tobytes())
            if bool(done):
                break
            if not args.headless:
                frontend.clock.tick(params.fps)
        print({
            "steps": int(frontend.state.time), "return": round(total_return, 4),
            "goal_reached": bool(frontend.info.get("goal_reached", False)),
            "collision": bool(frontend.info.get("collision", False)),
            "truncated": bool(frontend.info.get("truncated", False)),
        })
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("FFmpeg failed to encode the recording.")
        pygame.quit()


if __name__ == "__main__":
    main()
