# Trajectory correction

Refine a retargeted manipulation reference using object dynamics, contact-force
optimization, and local arm/hand trajectory optimization. Dataset decoding is
outside this module.

## Supported manipulation demo

The current manipulation experiment removes locomotion and arm IK from the
physics test. Six force-limited spherical fingertips carry a freely simulated
book using corrected contact commands. It does **not** claim a feasible G1 hand
configuration. See the [method, results, and limits](../docs/correction/SUPPORTED_MANIPULATION.md).

```text
python -m pip install -r setup/correction-simulation-requirements.txt
python -m correction.scripts.supported_book --verify
python -m correction.scripts.view_supported --port 8083
```

The default is episode 11, frames 520–565, using the saved retargeting input at
`correction/outputs/reorganization_check_20261017/retargeted/contact_aware.npz`.
Pass `--input` to use another saved input. The core experiment requires the
saved arrays, object mesh, and support-surface file, but does not import G1,
Pinocchio, or NVIDIA code. Its initial state is an established grasp; it does
not solve grasp acquisition.

## Layout

```text
correction/
├── tests/                  Mechanics, optimization, interface, and artifact tests
├── scripts/                Commands for individual stages or the complete trial
│   ├── prepare_phase.py
│   ├── run_phase.py
│   ├── book_pipeline.py
│   ├── book_trial.py
│   ├── repair_robot.py
│   ├── audit_book.py
│   ├── inspect_reference.py
│   ├── demo.py
│   ├── view_book.py
│   └── plot_book.py
└── src/
    ├── preprocessing/      Read retargeting outputs and prepare phase inputs
    ├── dynamics/           Body/contact models, object wrench, and physical audit
    ├── optimization/       Contact placement and robot-trajectory refinement
    ├── pass_traj/          Write contact plans and corrected NPZ/JSON references
    └── paths.py            Checkout-relative paths
```

The book pipeline is `preprocessing → dynamics → optimization → pass_traj`,
followed by an independent audit of the saved robot contacts. The reusable
optimization returns results; `pass_traj` owns writing the downstream reference.

## Run

From the repository root, with the [CPU environment](../docs/SETUP.md) active:

```text
python -m correction.scripts.demo
python -m correction.scripts.book_pipeline
python -m correction.scripts.view_book
python -m unittest discover -s correction/tests -v
```

The book pipeline expects the existing retargeted episode in
`correction/outputs/tier1_book/`. It selects frames 135–175 by default. Use
`--directory`, `--start`, and `--stop` to select an explicit book trial. Generic
phase preparation and optimization have their own commands; use `--help` and
see the [input/output contract](../docs/correction/INTEGRATION.md).

Dataset downloads, diagnostics, and SOMA model decoding live together in
[data_tools/](../data_tools/README.md). Use the Python entry points above;
machine-specific convenience launchers are not part of the public repository.

## Output and scope

The new **bounded book experiment** uses sparse force allocation, a small
beam of contact-face/pinch hypotheses, overlapping robot SQP windows, explicit
velocity/acceleration limits, URDF/table/self-collision checks, and optional
CPU physical rollouts. It writes a separate candidate reference and never
overwrites the original retargeting output. See the
[bounded method, commands, measurements and failures](../docs/correction/BOUNDED_CORRECTION.md).

```text
python -m pip install -r correction/requirements.txt
python -m pip install -r setup/correction-simulation-requirements.txt
python -m correction.scripts.bounded_book --start 400 --stop 601 --budget 120 --rollout
python -m correction.scripts.view_bounded --port 8083
```

This trial requires the optional G1 CPU runtime described in setup. Its local
MuJoCo scene is a diagnostic, not the official challenge simulator. The first
full-interval trial met the correction runtime budget but **failed physical
acceptance**. Smoother joints and reduced penetration did not establish a
usable manipulation.

`pass_traj` preserves the existing `dynamics_guided.npz` / `.json` robot-reference
schema. The training agent still needs to add the simulator scene and training
adapter. These artifacts are references, not a trained policy or a physical
rollout. The book audit still reports that the result is **not accepted as a
feasible manipulation**. This refactor changes organization, not the method or
its physical claims.

All detailed documents are under the repository's `docs/` directory:

- [Architecture and migration map](../docs/correction/ARCHITECTURE.md)
- [Mathematical method](../docs/correction/METHOD.md)
- [Reference interface](../docs/correction/INTEGRATION.md)
- [Experiments and measured limitations](../docs/correction/EXPERIMENTS.md)
- [Compute and implementation handoff](../docs/COMPUTE_AND_IMPLEMENTATION_HANDOFF.md)
