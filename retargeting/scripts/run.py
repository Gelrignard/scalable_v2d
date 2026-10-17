"""Run the pinned body-only baseline and our contact-aware reference generator."""

import _bootstrap  # noqa: F401 -- initialize local source/runtime paths
import argparse
import json
from pathlib import Path
from v2d_retarget.paths import MODULE
from v2d_retarget.dataprocessing import load_demo, load_npz, save_npz
from v2d_retarget.core.robot import G1Kinematics
from v2d_retarget.core.surface import ObjectSurface
from v2d_retarget.core.targets import build_body_targets
from v2d_retarget.core.contacts import build_contacts
from v2d_retarget.core.solver import solve_baseline, solve_contact_aware
from v2d_retarget.core.diagnostics import evaluate
from v2d_retarget.output import save_result
from v2d_retarget.output.motion_v1 import export_motion_v1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        help="Canonical HumanMotion NPZ; default: verified NVIDIA example",
    )
    p.add_argument(
        "--method", choices=["baseline", "contact_aware", "both"], default="both"
    )
    p.add_argument("--output", type=Path, default=MODULE / "outputs/snack_box")
    p.add_argument("--config", type=Path, default=MODULE / "configs/contact_aware.json")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--fps", type=float, help="Explicit timing override for cached demo only"
    )
    p.add_argument("--max-iterations", type=int)
    p.add_argument("--export-parquet", action="store_true")
    args = p.parse_args()
    if args.input and args.fps is not None:
        p.error("Canonical input carries timestamps; --fps applies only to the example")
    cfg = json.loads(args.config.read_text())
    if args.max_iterations is not None:
        cfg["max_iterations"] = args.max_iterations
    if cfg["max_iterations"] < 1 or not 0 < cfg["velocity_fraction"] <= 1:
        p.error("Invalid solver iteration/velocity settings")
    motion = (load_npz(args.input) if args.input else load_demo(args.fps)).subset(
        args.start, args.stop, args.stride
    )
    if args.export_parquet and motion.upstream_motion is None:
        p.error(
            "Parquet export needs demo scene metadata; canonical inputs export JSON/NPZ"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    save_npz(motion, args.output / "human_motion.npz")
    print(
        f"Preparing surface and contact hypotheses for {len(motion.timestamps)} frames",
        flush=True,
    )
    surface = ObjectSurface(motion.object_mesh)
    contacts = build_contacts(motion, surface, cfg)
    print(
        f"Active fingertip frames: {contacts.active.sum(axis=0).tolist()}", flush=True
    )
    comparison = {
        "sequence_id": motion.sequence_id,
        "config": cfg,
        "source": motion.metadata,
        "methods": {},
    }
    for method in (
        ["baseline", "contact_aware"] if args.method == "both" else [args.method]
    ):
        kin = G1Kinematics()
        body = build_body_targets(motion, kin.config)
        solved = (
            solve_baseline(kin, motion)
            if method == "baseline"
            else solve_contact_aware(kin, motion, body, contacts, surface, cfg)
        )
        print(f"Evaluating {method} on final saved poses", flush=True)
        metrics, arrays = evaluate(kin, motion, solved.q, body, contacts, surface)
        metrics["elapsed_seconds"] = solved.elapsed_seconds
        path = save_result(
            args.output,
            method,
            motion,
            kin,
            solved,
            body,
            contacts,
            cfg,
            metrics,
            arrays,
        )
        comparison["methods"][method] = metrics
        (args.output / f"{method}_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False)
        )
        if args.export_parquet and method == "contact_aware":
            export_motion_v1(
                args.output / "motion_v1",
                motion,
                kin,
                solved,
                contacts,
                metrics,
                arrays,
            )
        print(f"Saved {path}", flush=True)
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False)
    )
    print(
        json.dumps(
            {
                name: {
                    k: r[k]
                    for k in [
                        "active_contact_anchor_error_m",
                        "active_contact_surface_distance_m",
                        "hand_probe_penetration_m",
                        "speed_violating_transitions",
                    ]
                }
                for name, r in comparison["methods"].items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
