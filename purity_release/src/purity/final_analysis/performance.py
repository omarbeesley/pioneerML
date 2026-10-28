"""Port of the final perf_fl2.py table recipe; see docs/reproduction.md."""

import numpy as np

from ._shared import iter_arrays, parse_args, resolve_files, write_result

ZMIN, ZMAX, XY = 1.2, 4.8, 8.0
LO, HI = -300.0, 495.0
COLUMNS = [
    "pred_htp", "pred_polar_angle", "truth_theta", "pred_pion_stop_x",
    "pred_pion_stop_y", "pred_pion_stop_z", "truth_pion_stop_x",
    "truth_pion_stop_y", "truth_pion_stop_z", "truth_positron_t",
    "truth_event_type", "truth_positron_energy", "truth_htp",
]


def summarize(shards):
    radial_residuals, angles, angular_residuals = [], [], []
    denominator = failures = transverse_denominator = transverse_failures = 0
    for data in shards:
        truth_time = data["truth_positron_t"]
        fiducial = (
            (data["truth_pion_stop_z"] > ZMIN) & (data["truth_pion_stop_z"] < ZMAX)
            & (np.abs(data["truth_pion_stop_x"]) < XY)
            & (np.abs(data["truth_pion_stop_y"]) < XY)
        )
        in_window = (truth_time >= LO) & (truth_time <= HI)
        base = fiducial & in_window & (data["truth_htp"] > 0.5)
        has_positron = data["pred_htp"] > 0.5
        denominator += int(base.sum())
        failures += int((base & ~has_positron).sum())
        theta = np.degrees(data["truth_theta"])
        predicted_angle = np.degrees(data["pred_polar_angle"])
        transverse = base & (theta >= 87) & (theta <= 93)
        transverse_denominator += int(transverse.sum())
        transverse_failures += int((transverse & ~has_positron).sum())
        valid_angle = base & has_positron & np.isfinite(predicted_angle) & np.isfinite(theta)
        angles.append(theta[valid_angle])
        angular_residuals.append((predicted_angle - theta)[valid_angle])
        radial = np.sqrt(
            (data["pred_pion_stop_x"] - data["truth_pion_stop_x"]) ** 2
            + (data["pred_pion_stop_y"] - data["truth_pion_stop_y"]) ** 2
            + (data["pred_pion_stop_z"] - data["truth_pion_stop_z"]) ** 2
        )
        radial_residuals.append(radial[base & has_positron & np.isfinite(radial)])

    if not radial_residuals:
        raise ValueError("No input shards")
    radial = np.concatenate(radial_residuals)
    theta = np.concatenate(angles)
    residual = np.concatenate(angular_residuals)
    if not len(radial):
        raise ValueError("No events pass the performance selection")
    profile = []
    for lower in range(0, 180, 6):
        mask = (theta >= lower) & (theta < lower + 6)
        if mask.sum() > 200:
            half_width = (np.percentile(residual[mask], 84)
                          - np.percentile(residual[mask], 16)) / 2
            profile.append((lower + 3, half_width, np.median(residual[mask])))
    widths = [point[1] for point in profile]
    boundary = [point for point in profile if 117 <= point[0] <= 123]
    # Preserve the source's first qualifying bin: usually centre 117 degrees.
    # Sparse examples yield null instead of nonstandard JSON NaN.
    return {
        "n_events": len(radial),
        "stop_median_um": float(np.median(radial) * 1000),
        "stop_p68_um": float(np.percentile(radial, 68) * 1000),
        "stop_p95_um": float(np.percentile(radial, 95) * 1000),
        "stop_gt1mm_pct": float(100 * np.mean(radial > 1.0)),
        "hw_min_deg": float(np.nanmin(widths)) if widths else None,
        "hw_max_deg": float(np.nanmax(widths)) if widths else None,
        "hw_at120_deg": float(boundary[0][1]) if boundary else None,
        "medbias_at120_deg": float(boundary[0][2]) if boundary else None,
        "match_fail_all_pct": 100 * failures / max(denominator, 1),
        "match_fail_transverse_pct": 100 * transverse_failures / max(transverse_denominator, 1),
        "den_all": denominator,
        "den_transverse": transverse_denominator,
    }


def main():
    args = parse_args(__doc__)
    files, excluded, results = {}, {}, {}
    for channel in ("pie", "michel"):
        files[channel], excluded[channel] = resolve_files(
            getattr(args, channel), args.min_shard_bytes
        )
        results[channel] = summarize(iter_arrays(
            files[channel], COLUMNS, purify_michel=(channel == "michel")
        ))
    write_result(args, results, files, excluded, "performance")


if __name__ == "__main__":
    main()
