# Step vs trajectory vs trajectory + replanning: which one when?

Moving a ball from one point on the Stewart platform's plate to another, three ways:

| | Method | Loops |
|---|---|---|
| (a) | **Step**: the PID aims straight at the target | 1: PID |
| (b) | **Trajectory**: a smooth cubic Hermite path from the ball's state to the target, planned once; the PID follows it | 2: PID + position update along the path |
| (c) | **Trajectory + replanning**: the path is planned again from the *measured* state every so often | 3: PID + position update + replanning |

All results below come from **simulation** (`trajectory_study.py`, run `python trajectory_study.py` to regenerate
every number, table and figure here). The simulation runs the same planner, PID, alpha-beta filter and metrics as the real
controller, and I checked that the real `control_step` reproduces the simulation exactly in 15 of 15 test cases. The
*plant* is a model with assumed numbers, so the platform tests (last section) are what confirm it.

## What was simulated

- **Plant:** a solid ball rolling on the plate, `a = 5/7 g sin(tilt) - 0.3 v`; the servo follows the tilt command with a
  0.05 s lag; the camera sees the ball 100 ms late (3 frames at 30 fps) with about 0.6 px of noise; tilt limit 10 degrees; plate radius 13.5 cm.
- **Default settings:** PID `Kp 0.75, Ki 0, Kd 0.35`; trajectory duration 3 s; replan every 0.5 s.
- **Starting conditions:** x0 in {5, 10, 12} cm, v0 in {-15, 0, +15} cm/s (negative = towards the target), target 0.
- **Metrics:** settling time (stays within 0.5 cm of the target), overshoot, RMS tracking error to the plan, cross-track error,
  final accuracy, peak tilt, and whether the ball left the plate. Score used for tuning: `mean settling + 2 x mean overshoot + 20 per ball lost`.
- **Feedforward (not in the notes):** every trajectory method is also run with *acceleration feedforward*: the tilt that
  gives the ball the planned acceleration is added to the PID output. Gain 0.75 (Experiment 8 explains why).

## What the simulation found

### 1. At the notes' default gains, trajectories look far better. That is mostly a damping effect.
10 cm to 0 from rest, mean of 10 noise seeds (Experiment 1):

| Method | Settling (s) | Overshoot (cm) | Peak tilt (deg) |
|---|---|---|---|
| Step | 8.80 +- 0.41 | 2.53 | 8.8 |
| Trajectory | 2.42 +- 0.02 | 0.22 | 1.2 |
| Trajectory + replanning | 4.90 +- 0.56 | 0.94 | 2.1 |
| Trajectory + feedforward | 2.61 +- 0.05 | 0.15 | 1.1 |
| Trajectory + replanning + feedforward | 2.63 +- 0.02 | 0.27 | 1.1 |

The step controller at Kp 0.75 / Kd 0.35 rings for the whole 10 s (figure 1): these gains are under-damped for a plant with
a 100 ms camera delay. A trajectory hides that by asking for a gentle motion (peak tilt 1.2 degrees instead of 8.8).

### 2. Tuning the step controller helped more than changing method (Experiment 4)
Each configuration's settings (gains, and path duration where there is one) searched with a grid plus hill-climbing, scored over the nine starting conditions:

| Method | Default score | Best-tuned score | Tuned settings | Mean settling | Mean overshoot | Balls lost |
|---|---|---|---|---|---|---|
| **Step** | 11.9 | **2.1** | Kp 0.4, Kd 0.3 | 1.85 s | 0.11 cm | 0 |
| Trajectory | 44.4 | 23.9 | Kp 0.25, Kd 0.5, T 1.5 s | 3.09 s | 0.38 cm | 1 |
| Trajectory + replanning | 49.9 | 27.5 | Kp 0.25, Kd 0.5, T 1.5 s | 4.84 s | 1.32 cm | 1 |
| Trajectory + feedforward | 5.8 | 3.6 | Kp 0.4, Kd 0.3, T 2.0 s | 2.13 s | 0.71 cm | 0 |
| Trajectory + replanning + feedforward | 5.9 | 3.3 | Kp 0.25, Kd 0.3, T 2.0 s | 2.34 s | 0.51 cm | 0 |

Re-tuning the step PID (11.9 to 2.1) is a bigger gain than the best change of method at the same gains (11.9 to 5.8). **On the
undisturbed plant a well-tuned step controller is the best of the five.** The planned-path methods without feedforward still lose
a ball even when tuned.

### 3. Where the method *does* matter: delay, and moves that must be smooth
Each method with its own tuned settings, then tested on degraded plants (Experiment 9; 7 disturbances x 2 starts = 14 runs):

| Method (tuned) | Settled within 10 s | Ball lost | Mean final error |
|---|---|---|---|
| Step | 12 of 14 | 0 | 0.45 cm |
| Trajectory | 8 of 14 | 1 | 2.08 cm |
| Trajectory + replanning | 8 of 14 | 1 | 1.92 cm |
| Trajectory + feedforward | 11 of 14 | 1 | 1.23 cm |
| Trajectory + replanning + feedforward | 11 of 14 | 1 | 1.32 cm |

- **Late camera (167 ms, controllers tuned for 100 ms):** step needs 7.5 s with 1.6 cm overshoot (9.8 s and 3.4 cm from the +15 start);
  trajectory + feedforward settles in 2.6 s with 0.2 cm (2.2 s, 0.1 cm). The PID-only trajectories never settle. *Feedforward
  makes a trajectory tolerant of latency.*
- **Slow servo (0.15 s):** step survives everything (6.1 s, 8.5 s); every trajectory method loses the ball from the fast outward start.
- **Noisy camera (0.3 cm):** from rest, step settles in 1.9 s and the feedforward trajectories in 2.7 s, while the PID-only ones take about 9.4 s (the
  replanning one also overshoots by 5.9 cm: noise gets into the plan). From the fast outward start, noise hurts the feedforward trajectories
  too (4.7 to 4.8 cm overshoot) while step still settles in 2.9 s.
- **Plate bias (-0.9 degrees):** *nothing* recovers (final error 2.3 to 3.6 cm). Every method needs an integral term here, and tuning on a clean plant chose Ki = 0.
- **Bump:** step 3.0 s / 0.3 cm; trajectory + feedforward 3.0 s / 0.2 cm. Equal.

### 4. Starting with velocity (Experiment 2, default gains)
| Method | Settled (of 9) | Mean overshoot | Worst overshoot | Ball lost |
|---|---|---|---|---|
| Step | 9 | 2.27 cm | 3.50 cm | 0 |
| Trajectory | 7 | 0.27 cm | 0.72 cm | 2 |
| Trajectory + replanning | 7 | 1.85 cm | 9.63 cm | 2 |
| Trajectory + feedforward | 8 | 1.02 cm | 6.14 cm | 0 |
| Trajectory + replanning + feedforward | 9 | 0.90 cm | 3.90 cm | 0 |

A ball already moving *towards* the target is handled well by every trajectory method (they plan from its velocity, and a
shorter path avoids overshooting the target). A ball moving *away* fast (+15 cm/s from 10 or 12 cm) is the dangerous case: the
PID alone can only start braking after an error has built up, and the plan itself is anchored on a 100 ms-old measurement, so
the PID-only trajectory methods lose the ball. Step brakes immediately because its error is already large. Feedforward fixes this.

### 5. Replanning only pays off in one situation (Experiments 5 and 7)
- **Replanning on a timer is not better than planning once.** Without feedforward it is clearly worse (10 cm move: 5.5 s vs 2.4 s).
  Replanning resets the reference to the ball's own measured state, which removes the position error the PID relies on; the more
  often it happens the worse it gets (replan every 0.1 s: 9.5 s; 0.25 s: 6.1 s; 0.5 s: 5.5 s; 1 s: 2.4 s). With feedforward it
  stops mattering (2.5 to 3.4 s for every interval).
- **Replanning when the ball is knocked off the plan does help.** After a bump mid-move (+10 cm/s at 1 s), replanning *only when the
  ball is more than 1 cm off the plan*, with feedforward, settles in **2.9 s with 0.1 cm** overshoot, against 6.8 s or more (or never)
  for the other variants. (A well-tuned step controller also handles a bump in about 3 s, see section 3; this comparison is at the default gains.)
- A version that replans forever, as the pseudocode reads literally, never settles (the ball hunts about 1 cm). That is why the planner
  stops replanning in its last 0.5 s and holds the target afterwards (a restart if the ball is knocked more than 3 cm away).

### 6. Following a moving reference (Experiment 6, circle of radius 5 cm)
The step method cannot follow a moving target at all. For the trajectory controller, using the *reference velocity* in the D term
(the notes' `Kd (v_ref - v)`) is what makes it work: at a period of 8 s the RMS error falls from 1.50 cm to 0.31 cm. Adding acceleration
feedforward does *not* improve steady circular motion under the camera delay (0.34 cm at gain 0.75, 0.38 cm at 1.0, period 8 s). At a period
of 6 s it helps on an ideal plant with no delay or lag (0.37 cm without, 0.16 cm at 0.75) but not on the default plant (0.44 cm without, 0.51 cm at 0.75).

### 7. How much feedforward (Experiment 8)
Balls lost out of nine starting conditions, for both trajectory methods: 2 at gain 0, 1 at 0.5, **0 at 0.75 or more**. Gain 0.75 has the best score, but for
the plain trajectory method the mean overshoot rises with the gain (0.27 cm at 0, 1.02 cm at 0.75), and 1.0 is worse than 0.75 for steady tracking.
0.5 is a gentler choice that gives up only the single hardest start.

## Which method, when

| Situation | Use | Why (evidence) |
|---|---|---|
| Ordinary move to a fixed target, plant behaves | **Step**, well tuned | Fastest (1.85 s mean settling), least overshoot (0.11 cm), simplest, no assumptions (Exp. 4) |
| Ball may start fast, especially away from the target | **Step**, or a trajectory **with feedforward** | PID-only trajectories lose the ball (Exp. 2, 9) |
| Slow servo / sluggish actuator | **Step** | It was the only method that never lost the ball (Exp. 9) |
| Camera latency or the plate responds later than assumed | **Trajectory + feedforward** | 2.6 s vs 7.5 s with 167 ms delay (Exp. 9) |
| You want a gentle, predictable move (low tilt, known arrival time) | **Trajectory** (with feedforward) | Peak tilt 1.1 degrees vs 8.8, arrival close to the planned time (Exp. 1, 5) |
| The target moves (circle, path following) | **Trajectory** with reference velocity (and some feedforward) | Step cannot follow; D on velocity error cut error 5x (Exp. 6) |
| Large disturbances during a long move | **Replanning, triggered by deviation, with feedforward** | 2.9 s vs 6.8 s or more after a bump, at default gains (Exp. 7) |
| Anything else | **Not** timer-based replanning | No better than planning once, and worse without feedforward (Exp. 5, 7) |
| Constant bias (a plate that is not level) | **Any method with an integral term** | None of them recovers without it (Exp. 9) |

**Does changing method help more than tuning?** Not on average. The one big step was tuning the PID (score 11.9 to 2.1), and the best
method change at fixed gains reached only 5.8. A method change helps a lot in particular conditions (camera latency, moving targets,
mid-move disturbances) but not across the board.

## Limits of this study
- It is a **simulation with assumed plant numbers**. The camera delay, servo lag, friction and noise are estimates; the results
  above will shift with the real values, particularly the latency comparison.
- Controllers were tuned on the clean plant and then tested on the disturbed ones; no integral action was chosen, which is why plate bias defeats all of them.
- One simulated run per starting condition (the 10 cm move uses 10 noise seeds, where the spread is small).
- The tuning score weights (2 x overshoot, 20 per lost ball) are a choice; the ranking between the tuned methods is close
  (2.1 to 3.6) and could change with other weights.
- The feedforward gain (0.75) was chosen by this score; it is a trade-off, not a universal constant.
- Flaws found and fixed while building it: replanning with no time left chattered; paths could swing past the plate edge or the
  target; and the lost-ball flag was never set at first. The planners' extra rules exist because of these.

## Repeating the comparison on the real platform
1. Tune the step PID first (the study suggests starting around `Kp 0.4, Kd 0.3`, then adjust on the hardware: P, then D, then a little Ki).
2. In the controller window, pick the **Method**, set the **Move duration** (start at 3 s), **Replan interval** (0.5 s) and **Feedforward**
   (0 = the notes' PID-only, 0.75 = the study's setting), set the target sliders, and tick **Enable control** with the ball resting at the
   start point. The move starts from the ball's state when you tick the box.
3. For a starting velocity, give the ball a push before ticking Enable (the log records the actual start).
4. Use **one program run per trial**: each run saves one CSV (`tuning_runs/csv/`) that records the method, feedforward and duration.
5. Score a trial: `python trajectory_study.py --csv tuning_runs/csv/run_xxx.csv`. Do at least three trials per condition and compare with the tables above.
6. The first thing to check is the one the simulation cannot know: how late the camera really is. If the real latency is longer than 100 ms, expect
   the step controller to suffer and the feedforward trajectory to hold up better.

Files: `trajectory.py` (planners), `pid.py`, `metrics.py`, `trajectory_study.py` (this study), `basic_controller.py` (the controller).
The figures (`fig1` to `fig9`), `results.csv` and `summary.md` are generated into this folder by the study; the PNG and CSV files are git-ignored.
