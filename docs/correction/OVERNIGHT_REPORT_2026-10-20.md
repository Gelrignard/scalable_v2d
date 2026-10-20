# Overnight report — 20 October 2026

**Final morning report, 08:13 EDT. Overnight experiments have ended; the research milestones below remain incomplete.**

We have a useful local prototype and measured physical improvements, but **no accepted full manipulation, training-ready reference, or demonstrated policy-training saving**. All experiments below use one bin episode: 858 frames, 28.57 seconds. The robot has a supported pelvis following the same recorded/filtered path; upper joints and the object are physically simulated. Walking and balance are outside this benchmark.

## What you can inspect

The [existing comparison viewer](http://127.0.0.1:8087/) still shows exactly two channels:

1. Original retargeting versus our retained best robot result, with the intended object motion visible separately.
2. The corresponding independent fingertip-pad simulation, which also fails and is labeled as a distinct actuator model.

The robot channel uses [rest_matched_v1](../../correction/outputs/overnight_20261020/rest_matched_v1/report.json); the pad channel uses [rest_support_v1](../../correction/outputs/overnight_20261020/rest_support_v1/report.json). No later failed candidate replaced them. One server remains on port 8087, Python PID 21928 / launcher 15676. No new ports were opened.

## Main original-versus-method result

These cases share the physical model, gains, actuator limits, original initial posture, at-rest initialization, task target, root path and uninterrupted clock. Geometry correction and contact guidance are intentional experimental variables. Thus the combined result does not by itself isolate offline correction from force guidance.

| Full episode | Object position RMS | Object rotation RMS | Peak requested torque / real cap | Accepted? |
|---|---:|---:|---:|---|
| Original retargeting | 444.99 mm | 64.60° | 34.30 | No |
| Geometry/continuity correction | 325.32 mm | 67.37° | 3.09 | No |
| Retained mechanics + contact guidance | **48.48 mm** | **5.91°** | **2.31** | **No** |

The retained mechanics result finishes upright, with 42.73 mm / 3.78° final error. Maximum errors are 132.05 mm / 12.35°, above the existing 10 mm / 5° acceptance limits. Actual torques are clipped at the real actuator caps; excessive *requested* torque exposes controller/model incompatibility. Joint-limit excursion is 9.05 mrad, below the current 10 mrad gate. This is partial physical progress, not a feasibility certificate.

The local model uses a 1 kg object, COM at the object origin, inertia diag(0.01, 0.01, 0.01) kg·m², physical friction 1 and conservative planning friction 0.8. The mass/inertia and physical friction were derived from official source settings at commit `0593d80de5d39ab6f56f73ed16ca464ba5053499`, rather than measured properties of this particular bin. Arm/hand position gains are 300/2, with gravity and nominal inertia compensation. Collision decomposition, simulator settings, gains and supported pelvis still differ from the official runtime. These source-derived parameters do **not** establish official-runtime parity; earlier 2 kg experiments remain preserved separately.

The complete motion is less accelerated on average than the original (joint acceleration RMS 3.99 versus 25.66 rad/s²), but has a maximum adjacent joint step of 0.209 rad and EEF step of 60.49 mm. These maxima remain visible in the record; the gate does not yet enforce a separate absolute adjacent-step threshold. Do not claim all motion is smooth from an RMS measure.

Independent pads improve from approximately 595 mm to 219 mm RMS but still fail badly. Their Cartesian actuation differs from the articulated robot. A pad failure or success cannot substitute for the robot result.

## What was implemented and learned

**A shared local mechanics/control contract.** Actual robot collision mesh patches, object surface regions, force frames, mass/inertia, friction, gains, actuator limits, timing and initial states are carried through the planner, controller and export. Source snapshots and input hashes accompany experiments. Object motion, planning hypotheses, commands and actual physical results remain separate. Every-step contact impulses and exact torque-peak components distinguish allocated forces from delivered forces.

**Robot-compatible contact hypotheses and cheap screening.** Candidate patches stay near human-derived regions. Quasistatic wrench allocation includes friction, contact compression and actuator constraints. It is only a necessary local screen: the completed five-contact grasp passes it but performs worse physically because the planned thumb hardly engages and the actual internal force distribution differs. A four-contact variant without that thumb has a 2.573 N gravity-support deficit. [Contact completion](../../correction/outputs/overnight_20261020/completed_replay_v1/report.json).

A [read-only constraint isolation](../../correction/outputs/overnight_20261020/capacity_isolation_v1/report.json) shows interacting limits. Changing friction, compression allowance or actuator capacity alone does not eliminate the deficit. Friction 1 plus 24.86 N compression reaches static balance at the real torque limit. This counterfactual was neither exported nor replayed, and does not authorize increasing runtime limits. The 15 N planning/guidance bound is not a clamp on all servo-induced physical contact force.

**Continuous command and collision repair.** Sparse cubic-spline optimization includes joint/rate bounds, nominal dynamics and geometric constraints. An initially missed between-frame joint excursion led to coefficient/derivative bounds; clear sample endpoints also proved insufficient for collision clearance. The later tail repair incorporates signed collision-distance constraints in the same spline variables and independently checks the resulting continuous path. Numerical acceptance remains separate from physical acceptance. [Mathematical implementation notes](SEQUENCE_CORRECTION.md).

**A negative local search result.** Three load-informed servo-offset proposals and three uninformed proposals were compared in the same four-parameter space with the same seed, controller and evaluation budget. All six fail. Every load-informed proposal worsens position tracking from its 259 mm seed. The best-ranked uninformed proposal also fails. This small, single-seed experiment gives no mechanics-search cost advantage and no general ranking of methods. It warns against tracking one arbitrary force allocation instead of optimizing delivered task motion. [Matched search audit](../../correction/outputs/overnight_20261020/servo_search_v2/comparison_audit.json).

**Useful phase consistency, with exact causal verification.** Rebuilding the second approach around the preceding simulated object SE3 pose improves that phase from 302.81 mm / 57.55° to 64.16 mm / 13.01°. A constrained C2 tail refit preserves the entire earlier command, and the physical replay is bitwise identical through frame 600. The full episode improves from 259.28 mm to 153.61 mm RMS but tips over later. Contact is already lost before planned release, so merely postponing withdrawal is unsupported. [Replay and audit](../../correction/outputs/overnight_20261020/predicted_replay_v1/report.json).

**Bounded surface feedback helps but does not solve task execution.** A saved-state decomposition separates contact gap into arm tracking, nominal contact geometry and object-prediction error. Over frames 645–720, tracking averages 3.3–4.1 mm and geometry 0.7–1.0 mm, while prediction error averages 35.8–47.1 mm. These averages include the failed tail; the early timeline is retained. [Persistence diagnostic](../../correction/outputs/overnight_20261020/persistence_audit_v1/report.json).

The follow-up controller uses the current simulated object pose as an explicitly **perfect bench sensor**. It maps half of a surface displacement, clipped at 15 mm, to small arm/waist offsets while keeping the saved reference, finger shapes, force plan and actuator caps unchanged. It includes position/rate/acceleration bounds. The first trial violates its acceleration contract due to hard joint-limit clipping and is rejected. The second restricts offsets to joints with sufficient continuous nominal margin; all added offset bounds pass, with zero conflicts.

| Same reference; controller ablation | Full position / rotation RMS | Final position / rotation error | Physical acceptance |
|---|---:|---:|---|
| Before surface feedback | 153.61 mm / 30.34° | 460.41 mm / 89.15° | Fail |
| Bounded surface feedback v2 | **57.72 mm / 7.66°** | **142.60 mm / 9.04°** | **Fail** |

All reference arrays and the actual robot/object prefix through frame 660 are exactly equal. The v2 offset reaches 0.0330 rad, 0.224 rad/s and 2 rad/s²; no offset-limit conflict occurs. This benefit belongs to the declared feedback change, not to an isolated offline optimizer improvement. [Comparison](../../correction/outputs/overnight_20261020/surface_follow_comparison_v1/report.json).

The bin stays upright but misses its final location. The unchanged first phase still requests 2.639 times torque capacity. During frames 680–700 robot vertical support averages 5.334 N and environment support 4.140 N: selected contacts persist longer, but this is not a continuous airborne carry. Unplanned finger contact also occurs after release, including a 68.45 N brief peak in the final interval. Dense impulse momentum balance closes to 2.44e-14 N·s. Following the object helps preserve interaction but can follow the wrong motion; it does not enforce the task objective or clean withdrawal. [Physical result](../../correction/outputs/overnight_20261020/surface_follow_v2/report.json), [force audit](../../correction/outputs/overnight_20261020/surface_follow_v2/release_impulse_audit.json).

## Why the method is still rejected

1. A geometrically valid contact and feasible allocated wrench do not ensure physical contact acquisition, persistent support or the allocated internal force distribution.
2. Nominal dynamics omit important servo tracking responses. At the retained result's worst wrist peak, position PD contributes 9.630 N·m, velocity PD 3.550 N·m, compensation −0.766 N·m and guidance −0.877 N·m, against a 5 N·m cap. Stronger force guidance alone does not address this peak. [Torque decomposition](../../correction/outputs/overnight_20261020/torque_peak_audit_v1/torque_peak_audit.json).
3. Updating approach geometry helps a later grasp, but contact persistence, task progress and release must remain coupled to the preceding physical state and actual controller.
4. The latest feedback assumes accurate object pose. No noisy-observation, mass/friction perturbation or multi-episode generalization result exists.

## Milestone status

| Milestone | Completed evidence | Still missing |
|---|---|---|
| Early official-runtime compatibility | Pinned source/settings inspection | Official executable/evaluator and whole-body replay |
| M0: failure/model contract | Local contract, contact/torque instrumentation, reproducible failures | Official runtime parity |
| M1: compact evidence | Human-derived regions/phases for the bin | General extraction and noisy-input validation |
| M2: robot contact hypotheses | Actual surface candidates, approach/collision/continuity checks | Reliable acquisition across episodes |
| M3: wrench screening | Actuator-bounded static screening/allocation | Uncertainty-aware horizon and delivered-force model |
| M4: acquisition/persistence | Useful partial carries in full simulation | Reliable complete contact sequence |
| M5: controller-aware correction | Spline/geometry prototypes, exact-prefix updates, bounded search comparison | Accepted reference and demonstrated compute advantage |
| M6: robustness/recovery | Bounded perfect-pose feedback ablation | Declared perception/physics perturbation tests and recovery |
| M7: complete episode | Uninterrupted 858-frame results; both requested viewer channels | A full episode that passes physical acceptance |
| M8: batch/generalization cost | Not run | Frozen settings, several episodes, cost distribution |
| M9: policy cost | Not run | Matched learner/seeds and cost to a common quality threshold |
| M10: Tier 3 | Not implemented in this overnight work | Reconstruction/uncertainty integration and cost |
| M11: official whole-body evaluation | Not run | Policy/controller integration and official validation |

## Compute, verification and reproducibility

CPU only, using the existing local environment. No credits, paid servers, GPU training, large installations, pushes, external messages or new agents. Physical replay processes peak around 1.1–1.2 GB resident memory. A detailed full 28.57-second replay takes roughly 82–98 seconds including setup; earlier less-instrumented trials were faster. This is not an RL runtime estimate.

The latest two feedback experiments took 87.17 and 82.20 seconds wall, and 87.23 and 82.44 CPU-seconds. The contact-persistence diagnostic took 7.08 seconds, capacity isolation 7.13 seconds, matched comparison 0.59 seconds. The v2 force/impulse sidecars add 0.20 and 6.90 seconds, respectively. Failed candidates and controller-contract failures are retained.

The [07:30 experiment index](../../correction/outputs/overnight_20261020/experiment_index_0730.json) contains 99 top-level reports: **53.28 minutes recorded process wall time and 48.14 minutes CPU time**, across 76 CPU records. All previously indexed reports are unchanged. Search parents are counted once, without adding nested candidate times again. These are partial recorded costs: sidecar diagnostics, tests, startup, development and other missing costs are excluded. They are neither total project effort nor total policy cost.

The latest focused suite passes 11 tests, including actual MuJoCo contact geometry/sign, no direct object writes, unchanged prefix, bounded offset response, observable limit conflicts and between-frame nominal margin. Earlier 28- and 37-test suites cover spline, collision, controller, mechanics and instrumentation changes. Compilation and `git diff --check` pass. Full unassisted physical replay, rather than test counts, determines the rejection above. Test logs and source snapshots are under `correction/outputs/overnight_20261020/`.

## Next decisions

Keep the 48.48 mm result as the visible fallback. Do not promote the later completed-contact or feedback branches merely because one interval improves. The feedback branch is useful evidence for a controller-aware formulation but performs worse globally than the retained best result.

The next mechanics experiment should optimize a small executable command/contact hypothesis against long-horizon object progress while accounting for contact availability and actuator saturation. It needs a matched direct-simulation comparator and a hard total budget. The earlier four-parameter force-error proposal failed; repeating it or imposing a stronger arbitrary force trace is not justified. First-phase saturation and clean withdrawal remain explicit targets. A tail-only fix cannot establish whole-episode feasibility while the first phase is unchanged and fails.

Before claiming policy savings, run a small matched learning pilot only once a useful physically checked reference/controller combination exists, then compare total preprocessing/search/training cost to the same quality threshold. Official runtime compatibility remains a prerequisite for challenge claims. Our supported-pelvis benchmark is not an official submission.

Detailed commands, failures, process IDs and continuation instructions are in the [overnight log](OVERNIGHT_LOG_2026-10-20.md). The prior reports are preserved as `correction/outputs/overnight_20261020/report_checkpoint_0630.md` and `report_checkpoint_0733.md`. Morning verification found all 99 indexed reports unchanged, all report links present, the viewer responding HTTP 200, and no active experiment processes. No new physical experiment was launched for this report. The app confirmed the overnight automation is **PAUSED**, as requested. Only the existing viewer remains running.
