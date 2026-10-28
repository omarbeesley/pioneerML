"""Port of the final accbias_final.py table recipe; see docs/reproduction.md."""

import numpy as np

from ._shared import iter_arrays, parse_args, resolve_files, write_result

ZMIN, ZMAX, XY = 1.2, 4.8, 8.0
LO, TOL = -300.0, 5.0
COLUMNS = [
    "pred_accepted", "pred_polar_angle", "pred_htp", "pred_pion_stop_x",
    "pred_pion_stop_y", "pred_pion_stop_z", "truth_theta", "truth_acceptance",
    "truth_positron_t", "pred_positron_time_consensus_ns", "truth_gen_weight",
    "truth_event_type", "truth_positron_energy", "truth_pion_stop_x",
    "truth_pion_stop_y", "truth_pion_stop_z",
]
ROWS = [
    ("FN_id", "Identification (chain matching)", -1),
    ("FN_timeoth", "Timing", -1),
    ("FN_fid", "Fiducial, false negative", -1),
    ("FP_fid", "Fiducial, false positive", +1),
    ("FP_both", "Fiducial and angle, false positive", +1),
    ("FN_angle", "Angular migration, false negative", -1),
    ("FP_angle", "Angular migration, false positive", +1),
    ("FN_edge", "Window edge, false negative", -1),
    ("FP_edge", "Window edge, false positive", +1),
    ("FN_other", "Unclassified, false negative", -1),
    ("FP_other", "Unclassified, false positive", +1),
]


def terms(shards, upper_edge):
    totals = {key: 0.0 for key in (
        "W", "FN_id", "FN_timeoth", "FN_edge", "FN_fid", "FN_angle", "fn_all",
        "FP_angle", "FP_fid", "FP_both", "FP_edge", "fp_all", "accr", "acct",
    )}
    for data in shards:
        weights = data["truth_gen_weight"].astype(float)
        truth_time = data["truth_positron_t"]
        reco_time = data["pred_positron_time_consensus_ns"]
        theta = np.degrees(data["truth_theta"])
        reco_angle = np.degrees(data["pred_polar_angle"])
        has_positron = data["pred_htp"] > 0.5
        time_ok = ((reco_time >= LO) & (reco_time <= upper_edge)
                   & (np.abs(reco_time - truth_time) < TOL))
        reco_fiducial = (
            (data["pred_pion_stop_z"] > ZMIN) & (data["pred_pion_stop_z"] < ZMAX)
            & (np.abs(data["pred_pion_stop_x"]) < XY)
            & (np.abs(data["pred_pion_stop_y"]) < XY)
        )
        truth_fiducial = (
            (data["truth_pion_stop_z"] > ZMIN) & (data["truth_pion_stop_z"] < ZMAX)
            & (np.abs(data["truth_pion_stop_x"]) < XY)
            & (np.abs(data["truth_pion_stop_y"]) < XY)
        )
        angle_ok = reco_angle < 120.0
        reco_accepted = (data["pred_accepted"] >= 0.5) & time_ok
        truth_accepted = (data["truth_acceptance"] > 0.5) & (truth_time <= upper_edge)
        false_negative = truth_accepted & ~reco_accepted
        timing_failure = false_negative & has_positron & ~time_ok
        edge_inside = truth_time > upper_edge - TOL
        false_positive_any = (~truth_accepted) & reco_accepted
        false_positive = false_positive_any & (truth_time <= upper_edge)
        truth_angle_rejected = theta >= 120.0
        masks = {
            "W": truth_accepted,
            "FN_id": false_negative & ~has_positron,
            "FN_edge": timing_failure & edge_inside,
            "FN_timeoth": timing_failure & ~edge_inside,
            "FN_fid": false_negative & has_positron & time_ok & ~reco_fiducial,
            "FN_angle": false_negative & has_positron & time_ok & reco_fiducial & ~angle_ok,
            "fn_all": false_negative,
            "FP_angle": false_positive & truth_angle_rejected & truth_fiducial,
            "FP_fid": false_positive & ~truth_angle_rejected & ~truth_fiducial,
            "FP_both": false_positive & truth_angle_rejected & ~truth_fiducial,
            "FP_edge": false_positive_any & (truth_time > upper_edge),
            "fp_all": false_positive_any,
            "accr": reco_accepted,
            "acct": truth_accepted,
        }
        for key, mask in masks.items():
            totals[key] += weights[mask].sum()
    denominator = totals["W"]
    if denominator <= 0:
        raise ValueError("No positive truth-accepted weight for the acceptance table")
    result = {key: 100 * totals[key] / denominator for key in (
        "FN_id", "FN_timeoth", "FN_edge", "FN_fid", "FN_angle",
        "FP_angle", "FP_fid", "FP_both", "FP_edge",
    )}
    result["FN_other"] = 100 * totals["fn_all"] / denominator - (
        result["FN_id"] + result["FN_timeoth"] + result["FN_edge"]
        + result["FN_fid"] + result["FN_angle"]
    )
    result["FP_other"] = 100 * totals["fp_all"] / denominator - (
        result["FP_angle"] + result["FP_fid"] + result["FP_both"] + result["FP_edge"]
    )
    result["NET"] = 100 * (totals["accr"] - totals["acct"]) / denominator
    return result


def combine(pie, michel):
    rows = [[label, pie[key], michel[key], sign * (pie[key] - michel[key])]
            for key, label, sign in ROWS]
    total = sum(row[3] for row in rows)
    direct = 100 * ((1 + pie["NET"] / 100) / (1 + michel["NET"] / 100) - 1)
    return {"rows": rows, "sum": total, "direct": direct}


def main():
    args = parse_args(__doc__)
    files, excluded = {}, {}
    for channel in ("pie", "michel"):
        files[channel], excluded[channel] = resolve_files(
            getattr(args, channel), args.min_shard_bytes
        )
    result = {}
    for upper in (500.0, 495.0):
        # Re-read one shard at a time for each window; no whole-pool frame in memory.
        per_channel = {
            channel: terms(iter_arrays(paths, COLUMNS, purify_michel=(channel == "michel")), upper)
            for channel, paths in files.items()
        }
        result[str(int(upper))] = combine(per_channel["pie"], per_channel["michel"])
    write_result(args, result, files, excluded, "acceptance_bias")


if __name__ == "__main__":
    main()
