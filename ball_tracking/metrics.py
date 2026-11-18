"""Metrics for one move of the ball to a target. Used on simulated runs and on real platform runs.

Positions in cm, times in seconds. The ball starts at `start`, the target is `target`.
"""
import numpy as np

PLATE_RADIUS_CM = 13.5


def compute_metrics(t, x, y, target, start, reference_x=None, reference_y=None, tilt_x=None, tilt_y=None,
                    band=0.5, final_window=2.0, plate_radius=PLATE_RADIUS_CM):
    """Score a move. x, y are the ball positions over time; reference_* is what the method asked for
    (planned position); tilt_* are the plate tilt commands.

    Returns a dict:
        settling_time_s   first time after which the ball stays within `band` cm of the target (nan = never)
        overshoot_cm      furthest the ball went past the target, along the line from the start
        overshoot_pct     the same as a percentage of the starting distance
        rms_to_target_cm  RMS distance from the target over the whole run
        rms_to_reference_cm  RMS distance from the planned position (nan if no reference given)
        cross_track_max_cm / cross_track_rms_cm  how far the ball strays sideways from the straight
                          line between the start and the target
        final_error_cm    mean distance from the target over the last `final_window` seconds
        final_jitter_cm   how much the ball moves about during those last seconds (std)
        peak_tilt_deg     largest tilt asked for
        saturated_pct     percentage of the run spent at the tilt limit
        fell              True if the ball left the plate
        stable            no fall and (nearly) still at the end
    """
    t = np.asarray(t, dtype=float)
    t = t - t[0]
    position = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    target = np.asarray(target, dtype=float)
    start = np.asarray(start, dtype=float)

    to_target = np.linalg.norm(position - target, axis=1)
    distance = np.linalg.norm(start - target)

    # Settling time: the last moment the ball was outside the band, and what came after
    outside = np.where(to_target > band)[0]
    if outside.size == 0:
        settling_time = 0.0
    elif outside[-1] == len(t) - 1:
        settling_time = float("nan")      # still outside the band at the end of the run
    else:
        settling_time = float(t[outside[-1] + 1])

    # Overshoot and sideways deviation, relative to the straight line from the start to the target
    if distance > 1e-9:
        along = (start - target) / distance                        # points from the target back to the start
        past_target = (position - target) @ along                  # positive before the target, negative past it
        overshoot = max(0.0, float(-np.nanmin(past_target)))
        sideways = np.abs((position - start)[:, 0] * along[1] - (position - start)[:, 1] * along[0])
    else:
        overshoot = float("nan")
        sideways = np.zeros(len(t))

    result = {
        "settling_time_s": settling_time,
        "overshoot_cm": overshoot,
        "overshoot_pct": 100 * overshoot / distance if distance > 1e-9 else float("nan"),
        "rms_to_target_cm": float(np.sqrt(np.nanmean(to_target ** 2))),
        "cross_track_max_cm": float(np.nanmax(sideways)),
        "cross_track_rms_cm": float(np.sqrt(np.nanmean(sideways ** 2))),
    }

    if reference_x is not None and reference_y is not None:
        reference = np.column_stack([np.asarray(reference_x, dtype=float), np.asarray(reference_y, dtype=float)])
        result["rms_to_reference_cm"] = float(np.sqrt(np.nanmean(np.sum((reference - position) ** 2, axis=1))))
    else:
        result["rms_to_reference_cm"] = float("nan")

    # Where the ball ends up
    last = t >= t[-1] - final_window
    result["final_error_cm"] = float(np.nanmean(to_target[last]))
    result["final_jitter_cm"] = float(np.sqrt(np.nanvar(position[last, 0]) + np.nanvar(position[last, 1])))

    # How hard the plate had to work
    if tilt_x is not None and tilt_y is not None:
        tilt = np.column_stack([np.asarray(tilt_x, dtype=float), np.asarray(tilt_y, dtype=float)])
        result["peak_tilt_deg"] = float(np.nanmax(np.abs(tilt)))
        result["saturated_pct"] = float(100 * np.nanmean(np.any(np.abs(tilt) >= 9.9, axis=1)))
    else:
        result["peak_tilt_deg"] = float("nan")
        result["saturated_pct"] = float("nan")

    result["fell"] = bool(np.any(np.linalg.norm(position, axis=1) > plate_radius))
    result["stable"] = bool(not result["fell"] and result["final_jitter_cm"] < 1.0)
    return result
