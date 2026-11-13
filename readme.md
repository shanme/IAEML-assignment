# IAEML assignments

Vadim Poponnikov, AA-03

- [Assignment 1 report](reports/Assignment_1.md): vehicle model, observations, reward, reflection, and environment demonstration.
- [Assignment 2 report](reports/Assignment_2.md): PPO experiments, computation, reflection, results, and model replay.

The vehicle environment is implemented in `src/env/robotaxi.py` and inherits from the supplied `BaseEnv`. PPO training and evaluation are in `src/algorithms/ppo.py`.

## Installation

Use Python 3.12. From the repository root, create an environment and install the pinned dependencies:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Alternatively, create a Python 3.12 virtual environment with `python -m venv .venv`, activate it, and run `python -m pip install -r requirements.txt`.

The commands below use Windows paths. On Linux or macOS, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`. The recorded experiments used Windows 11 and JAX on CPU.

## Run and check

Open the vehicle environment:

```powershell
.venv\Scripts\python.exe src/main.py
```

W/up accelerates, S/down brakes, A/left steers left, and D/right steers right.

Replay the trained policy or evaluate its saved checkpoint:

```powershell
.venv\Scripts\python.exe src/main.py --mode policy --checkpoint artifacts/ppo/checkpoint.npz
.venv\Scripts\python.exe src/algorithms/ppo.py evaluate --checkpoint artifacts/ppo/checkpoint.npz --episodes 32 --seed 10000
```

Run the automated checks:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

There are 14 checks for vehicle behavior, collision detection, episode endings, route clearance, JAX batching, checkpoint loading, action probabilities, and advantage calculation. Detailed commands for repeating both training runs and recording the videos are in the reports.

## Submitted results

- [Environment demonstration](artifacts/environment.mp4)
- [Trained policy video](artifacts/ppo/trained.mp4)
- [Untrained policy video](artifacts/ppo/untrained.mp4)
- [Final checkpoint](artifacts/ppo/checkpoint.npz) and [initial checkpoint](artifacts/ppo/initial.npz)
- Configuration, evaluation results, and training logs in `artifacts/ppo/` and `artifacts/ppo_524288/`

The final policy reached the goal in all 32 evaluation resets on map 0, with no collisions or timeouts. The reports explain the limited scope of this evaluation. The short-run checkpoint is retained to allow the earlier result to be checked as well.

The supplied environment and maps are the starting point for this work. References for the vehicle model, PPO, GAE, and JAX are included in the reports.
