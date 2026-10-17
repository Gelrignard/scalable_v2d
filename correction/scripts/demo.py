"""Controlled contact-repair experiment, explicitly not a human demonstration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from correction.src.dynamics.models import ContactPatch
from correction.src.optimization.contact_optimizer import CorrectionSettings, PhaseCorrector
from correction.src.pass_traj.phase_result import save_result


def pinch_case(force_cap=20., samples=16):
    # Two side patches on a 10 cm wide box. The reference grasp is displaced
    # 4 cm in y; lifting there produces a moment about x that the initial
    # collinear pair cannot cancel. Local contact correction can remove it.
    patches = [ContactPatch(
        name=name, center_object_m=np.array([x, 0., 0.]),
        inward_normal_object=np.array([normal, 0., 0.]),
        tangent_u_object=np.array([0., 1., 0.]), tangent_v_object=np.array([0., 0., 1.]),
        half_extent_uv_m=np.array([0.08, 0.08]), friction=0.5,
        max_normal_force_n=force_cap)
        for name, x, normal in (("left_contact", -0.05, 1.), ("right_contact", 0.05, -1.))]
    times = np.linspace(0, 1, samples)
    reference = np.tile(np.array([[-0.06, 0.04, 0.], [0.06, 0.04, 0.]]), (samples, 1, 1))
    required = np.zeros((samples, 6))
    # A prescribed 1 kg object accelerating vertically, then decelerating.
    required[:, 2] = 9.81 + 1.5 * np.sin(2 * np.pi * times)
    return times, patches, reference, required


def run_demo(output):
    times, patches, reference, required = pinch_case()
    settings = CorrectionSettings()
    config = {"contact_mode": "sticking", "object_frame_id": "synthetic_box_object",
              "provenance": "CONTROLLED synthetic grasp-offset example; no human/robot data; 1 kg prescribed object motion",
              "patches": [asdict(patch) for patch in patches], "settings": asdict(settings)}
    config = json.loads(json.dumps(config, default=lambda value: value.tolist()))
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(output / "input_phase.npz", timestamp_s=times,
                        reference_points_object_m=reference, required_wrench_object=required,
                        com_object_m=np.zeros(3), object_frame_id=np.array(config["object_frame_id"]))
    result = PhaseCorrector(times, patches, reference, required, np.zeros(3), settings).solve()
    report = save_result(result, output, config["provenance"])
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("correction/outputs/controlled_pinch"))
    run_demo(parser.parse_args().output)
