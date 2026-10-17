# Trajectory correction

Refine a retargeted manipulation reference using object dynamics, contact-force
optimization, and local arm/hand trajectory optimization. Dataset decoding is
outside this module.

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
│   ├── plot_book.py
│   └── run.ps1
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

Windows convenience launcher: `./correction/scripts/run.ps1 -Mode book`.
Dataset downloads and diagnostics live in [data_fetch/](../data_fetch/README.md).
SOMA model downloads and decoding live in [soma_tools/](../soma_tools/README.md).

## Output and scope

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
