"""Check the standalone release using only its local code, weights and examples.

Run in the pinned environment:
    python -I tests/check_standalone.py --output /path/to/new/report.json

This checks import closure, strict checkpoint compatibility, public output shapes
and deterministic seeded CPU inference on small fixtures. It does not measure
scientific performance or rerun training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Report already exists; choose a new path")
    root = Path(__file__).resolve().parents[1]
    source_root = root / "src"
    # The supported invocation uses -I, which excludes the current directory and
    # inherited PYTHONPATH. Add this release's code explicitly.
    sys.path.insert(0, str(source_root))
    config_paths = [root / "configs/main_filmg2_ep19.json", root / "configs/tail10_ep10.json"]
    configs = [json.loads(path.read_text()) for path in config_paths]
    defaults = {"PURITY_GATE_MODE": "3d", "PURITY_BEST_XY_BETA": "0.5",
                "PURITY_ROLE_FILM": "0", "PURITY_ENERGY_TIME_GATE_NS": "0",
                "PURITY_MAX_HITS": "250"}
    os.environ.update(defaults | configs[0]["environment"])
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"

    import numpy as np
    import torch
    import torch_geometric
    from torch_geometric.data import Batch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    # Import every supplied module, so missing local helpers fail even when a
    # particular two-event model forward would not reach that helper.
    module_names = []
    for path in sorted((source_root / "purity").rglob("*.py")):
        parts = list(path.relative_to(source_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        importlib.import_module(name)
        module_names.append(name)
    from purity.benchmark import TASK_WEIGHTS
    from purity.dataset import PURITYDataset
    from purity.models import PURITYModel
    from purity.models import PURITYTailModel

    def local_modules():
        paths = {}
        for name, module in sorted(sys.modules.items()):
            if name == "purity" or name.startswith("purity."):
                path = Path(module.__file__).resolve()
                assert path.is_relative_to(source_root), f"Module resolved outside release: {name}"
                paths[name] = str(path.relative_to(root))
        return paths

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    def tensors(value, prefix="output"):
        if isinstance(value, torch.Tensor):
            yield prefix, value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from tensors(item, f"{prefix}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from tensors(item, f"{prefix}[{index}]")

    report = {"schema_version": 1, "status": "running",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Standalone import/checkpoint/inference checks on bundled examples; not performance reproduction.",
              "runtime": {"python": sys.version, "torch": torch.__version__,
                          "torch_geometric": torch_geometric.__version__,
                          "device": "cpu", "threads": torch.get_num_threads(),
                          "python_isolated_mode": bool(sys.flags.isolated)},
              "models": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for config, config_path, model_class in zip(configs, config_paths, [PURITYModel, PURITYTailModel]):
            checkpoint = (root / config["checkpoint"]).resolve()
            assert checkpoint.is_relative_to(root) and checkpoint.is_file() and not checkpoint.is_symlink()
            checkpoint_hash = sha256(checkpoint)
            assert checkpoint_hash == config["checkpoint_sha256"], "Checkpoint checksum mismatch"
            os.environ.update(defaults | config["environment"])
            stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = model_class(**config["model_kwargs"]).train()
            model.load_state_dict(stored["model"], strict=True)
            epoch = stored.get("epoch")
            del stored
            local_classes = {}
            for layer in model.modules():
                cls = type(layer)
                if cls.__module__.startswith("purity."):
                    path = Path(inspect.getfile(cls)).resolve()
                    assert path.is_relative_to(source_root)
                    local_classes[cls.__name__] = str(path.relative_to(root))
            dropouts = sorted({layer.p for layer in model.modules() if isinstance(layer, torch.nn.Dropout)})
            if config["model"] == "main":
                assert dropouts == [0.0]
            else:
                assert 0.05 in dropouts and 0.1 in dropouts
            result = {"model": config["model"], "profile": str(config_path.relative_to(root)),
                      "checkpoint": config["checkpoint"], "checkpoint_sha256": checkpoint_hash,
                      "checkpoint_epoch": epoch, "strict_checkpoint_load": True,
                      "local_model_classes": local_classes, "dropout_probabilities": dropouts,
                      "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                      "samples": []}
            for sample in ("pie", "michel"):
                fixture = root / "examples" / f"{sample}.parquet"
                dataset = PURITYDataset(str(fixture), max_hits=config["inference"]["max_hits"])
                assert len(dataset) >= 2, "Fixture must retain two events after hit filtering"
                batch = Batch.from_data_list([dataset[index] for index in range(2)])
                with torch.inference_mode():
                    def forward():
                        seed(config["inference"]["seed"])
                        if config["model"] == "main":
                            return model(batch.x, batch.batch, task_weights=TASK_WEIGHTS,
                                         triggering_pion_slice=getattr(batch, "atar_triggering_pion_slice", None),
                                         truth_positron_mask=None)
                        return model(batch.x, batch.batch)
                    prediction = forward()
                    replay = forward()
                first, second = dict(tensors(prediction)), dict(tensors(replay))
                assert first.keys() == second.keys() and first
                for name, value in first.items():
                    torch.testing.assert_close(value, second[name], rtol=0, atol=0, equal_nan=True,
                                               msg=lambda message: f"{name}: {message}")
                if config["model"] == "main":
                    summary = prediction["event_summary"]
                    for key in ("positron_polar_angle", "positron_time", "positron_energy", "dead_energy", "accepted"):
                        assert summary[key].shape == (2,) and torch.isfinite(summary[key]).all(), key
                    assert summary["pion_stop"].shape == (2, 3)
                    assert summary["positron_dir"].shape == (2, 3)
                    assert ((summary["accepted"] >= 0) & (summary["accepted"] <= 1)).all()
                else:
                    for key in ("pie_logit", "pie_topo_logit", "muon_logit", "muon_dif_logit", "pion_dif_logit", "pileup_logit"):
                        assert prediction[key].shape == (2,) and torch.isfinite(prediction[key]).all(), key
                result["samples"].append({"fixture": str(fixture.relative_to(root)),
                                           "fixture_sha256": sha256(fixture), "events": 2,
                                           "output_event_rows": int((prediction["event_summary"]["accepted"] if config["model"] == "main" else prediction["pie_logit"]).shape[0]),
                                           "output_tensors": len(first),
                                           "output_tensor_elements": sum(tensor.numel() for tensor in first.values()),
                                           "seeded_replay_exact": True, "public_output_contract": "passed"})
            report["models"].append(result)
            del model
        report["local_imports"] = local_modules()
        report["imported_release_modules"] = len(set(module_names))
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Standalone checks passed: {report['imported_release_modules']} local modules; both checkpoints; four two-event batches.")


if __name__ == "__main__":
    main()
