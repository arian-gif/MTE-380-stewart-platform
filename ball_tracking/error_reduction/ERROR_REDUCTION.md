# Reducing the tracking error of the planned paths

Errors are RMS over the run (after the ball has reached the path), in cm. Pointwise = distance to the planned position at the same time; cross-track = distance to the planned path. For Step the planned position jumps from corner to corner, so its pointwise error is not meaningful; compare Step by cross-track.

Default = the controller's starting gains (Kp 0.3, Ki 0.05, Kd 0.35, feedforward 0). Tuned = the result of the iterative search (Kp, Ki, Kd, acceleration feedforward).

Training plant = the simulated platform the search ran on; harder plant = slower camera, more friction, a tilted plate and different noise, never used while tuning (its column is pointwise + cross-track RMS, plus 10 cm per run in which the ball fell).

## Before and after tuning

| Path | Method | Pointwise default | Pointwise tuned | Cross-track default | Cross-track tuned | Harder plant default | Harder plant tuned | Tuned Kp / Ki / Kd / FF |
|---|---|---|---|---|---|---|---|---|
| line | Step | 4.06 | 3.51 | 0.03 | 0.04 | 4.09 | 4.45 | 0.20 / 0.00 / 0.43 / 0.00 |
| line | Trajectory | 0.18 | 0.07 | 0.20 | 0.09 | 0.49 | 0.96 | 0.13 / 0.01 / 0.33 / 0.25 |
| line | Trajectory + replanning | 0.75 | 0.39 | 3.70 | 0.07 | 3.93 | 1.02 | 0.20 / 0.04 / 0.22 / 0.75 |
| triangle | Step | 3.82 | 3.28 | 0.29 | 0.41 | 4.10 | 4.39 | 0.20 / 0.00 / 0.49 / 0.00 |
| triangle | Trajectory | 0.17 | 0.12 | 0.38 | 0.10 | 0.65 | 0.41 | 0.12 / 0.07 / 0.35 / 0.44 |
| triangle | Trajectory + replanning | 0.39 | 0.33 | 1.88 | 0.11 | 2.64 | 1.00 | 0.27 / 0.00 / 0.26 / 1.00 |
| square | Step | 3.44 | 3.08 | 0.26 | 0.24 | 3.62 | 5.15 | 0.38 / 0.02 / 0.35 / 0.00 |
| square | Trajectory | 0.15 | 0.12 | 0.37 | 0.10 | 0.60 | 0.44 | 0.13 / 0.04 / 0.35 / 0.50 |
| square | Trajectory + replanning | 0.31 | 0.27 | 1.05 | 0.11 | 1.69 | 1.03 | 0.27 / 0.00 / 0.15 / 1.00 |
| circle | Step | 2.71 | 0.44 | 0.09 | 0.14 | 2.87 | 1.28 | 0.68 / 0.00 / 0.15 / 0.00 |
| circle | Trajectory | 0.18 | 0.04 | 0.17 | 0.03 | 0.34 | 1.72 | 0.21 / 0.00 / 0.33 / 0.25 |
| circle | Trajectory + replanning | 0.18 | 0.04 | 0.18 | 0.03 | 0.36 | 1.72 | 0.21 / 0.00 / 0.33 / 0.25 |
| infinity | Step | 2.28 | 0.44 | 0.43 | 0.18 | 2.69 | 1.25 | 0.72 / 0.00 / 0.15 / 0.00 |
| infinity | Trajectory | 0.19 | 0.05 | 0.16 | 0.04 | 0.28 | 1.97 | 0.18 / 0.00 / 0.33 / 0.28 |
| infinity | Trajectory + replanning | 0.19 | 0.05 | 0.17 | 0.04 | 0.31 | 1.99 | 0.18 / 0.00 / 0.33 / 0.28 |
| to centre | Step | 2.78 | 2.20 | 0.89 | 0.08 | 3.57 | 2.63 | 0.48 / 0.00 / 0.29 / 0.00 |
| to centre | Trajectory to centre | 0.34 | 0.22 | 0.09 | 0.03 | 0.79 | 1.77 | 0.18 / 0.00 / 0.33 / 0.31 |

## The other lever: how fast the path asks the ball to go

Tuned settings, training plant. Slower paths always track better; this is paid for in time, so it is a trade-off and not a free improvement.

| Path | Setting | Value | Pointwise RMS (cm) | Cross-track RMS (cm) |
|---|---|---|---|---|
| line | time per side (s) | 1.5 | 0.68 | 0.58 |
| line | time per side (s) | 2 | 0.22 | 0.15 |
| line | time per side (s) | 3 | 0.07 | 0.09 |
| line | time per side (s) | 4 | 0.09 | 0.11 |
| triangle | time per side (s) | 1.5 | 0.63 | 0.23 |
| triangle | time per side (s) | 2 | 0.36 | 0.15 |
| triangle | time per side (s) | 3 | 0.12 | 0.10 |
| triangle | time per side (s) | 4 | 0.12 | 0.21 |
| square | time per side (s) | 1.5 | 0.45 | 0.18 |
| square | time per side (s) | 2 | 0.31 | 0.13 |
| square | time per side (s) | 3 | 0.12 | 0.10 |
| square | time per side (s) | 4 | 0.10 | 0.16 |
| circle | lap time (s) | 6 | 0.08 | 0.08 |
| circle | lap time (s) | 8 | 0.04 | 0.03 |
| circle | lap time (s) | 10 | 0.04 | 0.03 |
| circle | lap time (s) | 14 | 0.04 | 0.03 |
| infinity | lap time (s) | 6 | 0.17 | 0.14 |
| infinity | lap time (s) | 8 | 0.08 | 0.07 |
| infinity | lap time (s) | 10 | 0.05 | 0.04 |
| infinity | lap time (s) | 14 | 0.05 | 0.04 |
| to centre | cruise speed (cm/s) | 4 | 0.13 | 0.03 |
| to centre | cruise speed (cm/s) | 6 | 0.17 | 0.03 |
| to centre | cruise speed (cm/s) | 8 | 0.22 | 0.03 |
| to centre | cruise speed (cm/s) | 12 | 0.36 | 0.03 |
