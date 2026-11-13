# Assignment 1: Vehicle environment

**Author:** Vadim Poponnikov  
**Group:** AA-03  
**Email:** v.poponnikov@innopolis.university

## 1. Vehicle model and observations

The environment is implemented in `src/env/robotaxi.py`. The `RobotaxiEnv` class inherits from `BaseEnv` and uses JAX for the simulation. It provides the initialization, reset, step, and observation functions required by the assignment.

The vehicle uses a bicycle model, which represents the front and rear wheel pairs as two wheels. This gives a simple relation between steering and the direction of movement. The vehicle state contains its position $(x,y)$, heading $\psi$, and forward speed $v$. The environment also stores the previous action, episode time, obstacles, goal, and planned route.

The action has two values in $[-1,1]$: acceleration input $u_a$ and steering input $u_\delta$. They are converted into acceleration and steering angle as follows:

$$
a=3u_a,\qquad \delta=0.6u_\delta.
$$

The equations of motion are:

$$
\dot v=a-0.15v-0.03v|v|,\qquad
\dot\psi=\frac{v\tan\delta}{1.5},\qquad
\dot x=v\cos\psi,\qquad \dot y=v\sin\psi.
$$

The wheelbase is 1.5 world units. The two resistance terms slow the vehicle down when acceleration is zero. They represent friction and drag after dividing forces by mass. Speed is limited to 4 units per second. Negative acceleration brakes the vehicle, but it cannot drive backwards. Steering a stationary vehicle does not rotate it.

One action lasts 0.1 seconds and is divided into four smaller physics steps to improve the movement and collision checks. Position is updated using the heading halfway through each small step. This model includes acceleration, braking, resistance, and a turning constraint without requiring a detailed tire model. It is intended for low speeds and does not simulate tire slip or suspension. Background on bicycle models is given by [Kong et al. (2015)](https://nuhuo08.github.io/control/IV_KinematicMPC_jason.pdf).

The vehicle and obstacles are circles with radius 1. Their collision distance is therefore 2. The original map cells were too close together for these circles, so the spacing was increased to three world units while keeping their radius unchanged. The map layouts remain the same, and each map covers 60 by 30 units. Moving obstacles travel along their assigned axis and reflect at the boundary. Their movement is prescribed; they do not avoid other obstacles.

A grid-based dynamic programming method plans a route around static obstacles. The route is calculated once when the map is initialized. Moving obstacles are detected by the range sensors and collision checks. Collision detection checks the closest relative distance during each small step, rather than only comparing the final positions.

The observation contains 26 values:

| Information | Number of values |
| --- | ---: |
| Range sensors covering 360 degrees | 16 |
| Speed | 1 |
| Previous acceleration and steering inputs | 2 |
| Goal position relative to the vehicle | 2 |
| Route point 3.5 units ahead, relative to the vehicle | 2 |
| Signed sideways distance from the route | 1 |
| Remaining route length | 1 |
| Elapsed fraction of the episode | 1 |

Distances, speed, and time are divided by fixed reference values, such as the 10-unit sensor range and the maximum speed. This keeps the inputs on similar scales. Relative positions use the vehicle's forward and left directions. The sensors do not provide obstacle velocities, so one observation does not fully describe their future movement.

An episode ends when the vehicle comes within one unit of the goal, collides, or reaches 600 steps. The time limit is 60 simulated seconds. Reaching the goal means entering this region; the vehicle is not required to stop there. After an episode ends, further steps leave the state unchanged and give zero reward until reset.

## 2. Reward function

The reward gives feedback during movement as well as at the end of an episode. Let $D_t$ be the remaining route distance, $e_{t+1}$ the sideways route error, and $u_t$ the two action inputs. The reward is:

$$
\begin{aligned}
r_t={}&2(D_t-D_{t+1})-0.01-0.005e_{t+1}^2\\
&-0.005\|u_t\|^2-0.002\|u_t-u_{t-1}\|^2\\
&+50\mathbf{1}_{\text{goal}}-50\mathbf{1}_{\text{collision}}.
\end{aligned}
$$

Progress along the route gives a positive reward. Using route distance helps guide the vehicle around walls, where moving directly toward the goal may be unsafe. The small time penalty discourages waiting. The other penalties discourage leaving the route, using large control inputs, and changing them sharply. Reaching the goal adds 50, while a collision subtracts 50.

These terms provide more frequent feedback than a reward based only on the final outcome. However, a higher return does not always mean that the vehicle reaches the goal. Goals, collisions, and timeouts are therefore also measured in Assignment 2.

## 3. Reflection

Development began with a review of movement in the starter environment. Its actions directly changed the agent's position, so the movement rules needed to be replaced to support acceleration and steering. Separate checks then verified acceleration, coasting, turning, and braking before RL training.

The main difficulty was matching the map spacing to the required object size. Radius-one circles could not pass through some corridors at the original spacing. Increasing the spacing resolved this while preserving the required radius and map layouts. Collision detection also needed checks during movement, since checking only the final position could miss an obstacle crossed within a step.

Testing the environment before training helped separate simulation errors from learning problems. The demonstration confirmed that speed decreases during coasting and reaches zero after braking, as expected from the resistance and control rules. The main remaining limitations are immediate steering changes and the absence of obstacle velocities in the observation. These simplify the environment but limit the range of driving behavior it can represent.

## 4. Results and reproduction

The [environment video](../artifacts/environment.mp4) shows a fixed sequence of acceleration, coasting, right and left steering, and braking. It lasts eight seconds and has no collision.

| Step | Completed phase | Speed, units/s |
| --- | --- | ---: |
| 20 | Acceleration | 3.168 |
| 40 | Coasting | 2.013 |
| 55 | Right steering while coasting | 1.485 |
| 70 | Left steering while coasting | 1.118 |
| 80 | Braking | 0.000 |

This demonstration ends at its 80-step time limit. It is a physics demonstration rather than a trained driving policy. All three supplied maps have complete planned routes with a minimum static-obstacle center clearance of 2.5 units. The automated checks also cover movement, collisions, episode endings, batched execution, and the RL calculations.

Install Python 3.12 and the dependencies using the instructions in the [README](../readme.md). From the repository root, run:

```powershell
.venv\Scripts\python.exe src/main.py
.venv\Scripts\python.exe src/main.py --mode dynamics --headless --seed 10000 --steps 80 --record artifacts/environment.mp4
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The first command opens the interactive environment. W/up accelerates, S/down brakes, A/left steers left, and D/right steers right. The second command recreates the attached video. On Linux or macOS, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.
