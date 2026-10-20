# Research review and implementation decision

Reviewed on **19 October 2026**. This document combines a primary-source literature review, a review of the current V2D release, and a read-only audit of our code and saved experiments. It proposes the next implementation; it does not report a new successful method. No training or new physical rollout was run for this review.

The [earlier literature review](../LITERATURE_REVIEW.md) remains a historical snapshot. This document supersedes its description of the current implementation and submission interface. The [46-paper reading list](../LITERATURE_REFERENCES.md) is broader; this review rechecks the closest work in depth rather than claiming to have independently revalidated every entry.

## 1. Research question and decision

Our objective remains **reducing the total computational cost of turning imperfect human demonstrations into useful robot policies**. Mechanics and trajectory optimization are the proposed means. A compact representation is valuable only if it helps that objective while preserving manipulation quality.

The question I recommend testing is:

> Can a small search over demonstration-supported, robot-compatible contact strategies, followed by controller-aware trajectory correction, reduce the total compute needed to reach a fixed object-tracking quality?

Three decisions follow from this review:

1. **Preserve human intent, but search over robot contact surfaces.** Human motion constrains where, when and in which direction the robot should interact. It need not specify the same fingertips or a unique force trace.
2. **Optimize commands that the actual controller can execute.** A feasible force allocation and a nearby hand trajectory are insufficient if they cannot establish and maintain the same contacts through that controller.
3. **Measure the learning benefit early.** A useful correction need not make every open-loop episode perfect before any learning experiment. Nevertheless, the requested demonstration must remain a complete uninterrupted physical episode, and failed frames must remain visible.

This is a research hypothesis, not yet an established contribution. Its strongest possible result is a favorable total-cost-versus-quality curve, explained by better contact acquisition and persistence. A friction cone, a residual action, or a smoother trajectory alone does not establish that result.

## 2. What the current challenge requires

The posted deadline is **4 November 2026, 17:00 EST**, approximately 16 days after this review date. Track 2 uses G1 with Dex3 hands and evaluates all tiers against clean object trajectories. The Industry Innovation Award explicitly includes economic viability and complexity reduction. A valid submission, technical report and code-release commitment remain necessary for that award. These requirements directly support investigating computational efficiency. [Challenge page](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)

The FAQ requires all **30 episodes per tier**, learned-policy rollouts, and whole-body loco-manipulation; pretrained controllers are allowed. The supported-pelvis setup is therefore a manipulation development experiment, not a complete challenge submission. For Tier 3, the FAQ permits Tier 1's first frame for alignment, not the rest of its clean trajectory as a reconstruction input. [Official FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)

The current submission kit describes participant-side training and evaluation, an episode-to-checkpoint mapping, recorded trajectories, and one CSV per tier. Its evaluator is FlashCHORD `evaluate_policy.py`; the website still mentions an older entrypoint. Per-episode specialist policies are supported; a universal policy is not required. Provenance validation fixes the checkpoint hash within an episode, without requiring different hashes across episodes. The metric code compares fixed timestamps in the released world frame, without initial-pose alignment. Consequently, accomplishing the task with a substantially different object path can score poorly. Use the supplied frame grid, not an approximate resampling. [Submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip)

Tier 3 reconstruction results are **not currently published in the dataset**: its README instructs participants to reconstruct the shared exocentric videos. A `tier_3_cari4d` directory convention in the kit does not mean those reconstructed episodes are available for download. This dependency needs its own implementation and cost estimate. [Dataset README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/track_2/README.md)

The official full-sequence evaluator disables object assistance and automatic resets. Our output must eventually be the object's executed motion under those conditions. An optimized desired trajectory is not a substitute. [Evaluation harness](https://github.com/nvidia-isaac/video_to_data/blob/0593d80de5d39ab6f56f73ed16ca464ba5053499/robotic_grounding/flash_chord/src/flash_chord/evaluation/harness.py)

**Planning consequence:** move action-interface, data-schema and evaluator compatibility checks to the beginning. They do not require purchasing a full CHORD training run. Whole-body integration and Tier 3 cannot be left as unknowns until after the manipulation method is perfected. The present evidence does not justify promising a competitive complete entry within the remaining 16 days.

## 3. What CHORD does, and what remains our question

CHORD compares human and robot contact-wrench **capabilities**, constructed from unit friction-cone primitive wrenches and their directional supports. It does not recover a unique demonstrated per-finger force trace. Its contact representation is not an actuator-bounded certificate that a particular robot can acquire contacts and deliver a prescribed net wrench. Physics and RL realize execution, alongside tracking, imitation, curriculum and resets. For noisy whole-body reconstructions, guidance emphasizes force closure. Its reported approximately two hours on one L40S concerns per-sequence training with 2,048 environments, not a verified 4090 budget. [CHORD v2, Sections 3.1–3.2 and Appendices B/E](https://arxiv.org/html/2607.00033v2)

For our reasoning, separate three questions:

| Question | Meaning in our method | What it does not establish |
|---|---|---|
| Capability | Which wrench directions can these candidate surfaces support? | Whether the desired motion needs those directions |
| Demand | Which net wrenches are compatible with a plausible object motion and environment support? | Which fingers should supply them |
| Realization | Can the robot acquire those contacts and sustain their forces through its real command interface? | Robustness or reduced policy-training cost |

The third question is the missing bridge in our failed demonstration. Our proposed mechanics should make that bridge cheaper, then establish whether the learner benefits.

SONIC is a learned whole-body tracking controller with residual interfaces, not walking IK. The official ReconBody environment combines body/object tracking and force-closure guidance. We should preserve that eventual interface while isolating upper-body manipulation locally. [Whole-body environment documentation](https://github.com/nvidia-isaac/video_to_data/blob/0593d80de5d39ab6f56f73ed16ca464ba5053499/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/README.md)

Current FlashCHORD documentation validates full training on **48 GiB, with at least 46,000 MiB free**. It does not establish that 24 GB cannot work, or that it already does. [FlashCHORD README](https://github.com/nvidia-isaac/video_to_data/blob/0593d80de5d39ab6f56f73ed16ca464ba5053499/robotic_grounding/flash_chord/README.md) Its G1 recipe uses 2,048 worlds, approximately 350 million environment steps and ten million replay transitions; reducing world count alone may not remove replay-buffer memory pressure. [G1 recipe](https://github.com/nvidia-isaac/video_to_data/blob/0593d80de5d39ab6f56f73ed16ca464ba5053499/robotic_grounding/flash_chord/src/flash_chord/configs/experiment/g1_recon_body_flash_sac.yaml)

Our laptop can support bounded mechanics experiments; whether the complete learner fits the offered 24 GB 4090 is a separate measurement. At the quoted $0.32/hour, 10 GPU-hours cost $3.20 and 100 cost $32 before storage or other fees. These are price calculations, not runtime forecasts. The server's 30 GB disk allowance also needs checking before installing dependencies and staging assets.

FlashCHORD uses MuJoCo-Warp through Newton's `SolverMuJoCo`. Its `use_mujoco_contacts=true` path preserves MuJoCo contact parameters such as `solref`/`solimp`; the alternate collision path ignores those parameters. Contact backend, gains and solver settings therefore belong in our compatibility check. A successful local CPU MuJoCo replay would not by itself establish identical behavior upstream. [Runtime source](https://github.com/nvidia-isaac/video_to_data/blob/0593d80de5d39ab6f56f73ed16ca464ba5053499/robotic_grounding/flash_chord/src/flash_chord/runtime/sim.py)

## 4. Closest literature and what to borrow

The costs below belong to each paper's actual experiment. They are not estimates for our complete G1 episodes.

| Work | Verified scope and useful idea | Limitation and compute evidence |
|---|---|---|
| [PhysicsGen](https://arxiv.org/html/2502.20382v2), [full appendix](https://arxiv.org/pdf/2502.20382v2) | Kinematic initialization followed by simulation-based optimization of executable command knots; generated data trains diffusion policies. Borrow object-centric command optimization. | Inputs are VR demonstrations already interacting with simulated objects. CEM uses six knots, 50 samples and five elites, at 200 Hz over 1.25–2 s. No optimizer wall-clock/hardware estimate was located. The reported seven minutes concerns human demonstration collection. |
| [SPIDER v3](https://arxiv.org/html/2511.09484v3) | Sparse command optimization through parallel physics, receding horizons, optional contact guidance and robustness treatment; includes downstream learning. This is the closest direct baseline. | Large-scale median approximately 220 s/task includes preprocessing for short floating-hand contact-establishment/lift clips. L40S throughput is approximately 6 generated frames/s without guidance versus 0.4–0.5 with guidance under different configurations. Separate grasp PPO takes 3–4 h/L40. These are not full G1 episode costs. |
| [ReForce v1](https://arxiv.org/html/2608.15560v1) | Behavior-cloned per-finger feedback tracks motion and normal-force targets using bounded accumulated joint-command deltas. Borrow measured-versus-target force feedback. | It is supervised learning, not RL. Human force targets come from FSR sensors and robot feedback from tactile sensing; ordinary V2D poses do not supply those labels. Classical admittance outperforms it in the reported tongs force-safe-success comparison, 90% versus 54.5%. Do not assume a learned force controller is necessary. |
| [C2Dex v2](https://arxiv.org/html/2608.07045v2) | Aggregates contact evidence in object coordinates using temporal consistency and normal compatibility; retargets interaction geometry. Useful representation for noisy human contact evidence. | Holds the estimated object trajectory fixed and still uses residual PPO for physical execution. G1 hardware has Inspire hands, not Dex3. No compute budget was located. Stabilizing contact points must not erase genuine sliding. |
| [Grasp-and-Lift v1](https://arxiv.org/html/2601.18121v1) | Sparse spline commands, low-dimensional MANO coordinates and CMA-ES; the object's motion emerges from physics. A useful compact optimization baseline. | Human MANO embodiment, not G1. Clips have 60–70 frames at 30 Hz with the first ten excluded. Average optimization is 556.95 s on an A6000/i7 workstation; CPU/GPU division is unspecified. No demonstrated downstream learning saving. |
| [Direct Dynamic Retargeting v1](https://arxiv.org/html/2605.23762v1) | Optimizes simulator commands against video-derived task-space targets without making an intermediate IK trajectory compulsory. Borrow observation-based soft targets. | H1-2 body-motion experiments do not establish dexterous object transfer. Learning speed is measured toward a fraction of each method's own final reward, not a common quality threshold or total cost. |
| [Pang et al., quasi-dynamic contact planning](https://arxiv.org/html/2206.10787) | Includes actuator stiffness and optimizes position setpoints in a smoothed reduced contact model. Attractive for supported dragging. | Analytic-smoothed CPU planning reports 3.25–117.16 s for 1,000 iterations across tasks. The paper reports missed grasps/drift in second-order replay when damping is inadequate. Use as a planning approximation with feedback and physical validation. |
| [STOCS 2023](https://arxiv.org/html/2306.06465) | Sparse selection of object–environment support contacts; hardware uses pressing position offsets and pose feedback between modes. | Robot contact modes are prescribed or selected by an outer search; quasi-static regrasp assumes environment support. Complete multimodal examples take median 425–3,360 s on one CPU core. It is not an automatic inexpensive G1 grasp planner. |
| [STOCS 3D extension](https://arxiv.org/html/2407.16976) | Selects a small active contact set while retaining more detailed geometry. Borrow this active-set principle. | Still assumes sticking robot contacts. Examples report 24.84–676.01 s, generally for ten 0.1 s steps. Fewer candidate contacts do not automatically mean lower solve time. |
| [Contact-area retargeting, author PDF](https://www.andrew.cmu.edu/user/aslakshm/pdfs/TOG2025KinematicMotionRetargeting.pdf) | Surface charts, contact areas, IK and temporal cleanup transfer interactions across different hand morphology. Borrow regions instead of manufacturer fingertip sites. | Kinematic, with human annotation; no dynamic execution certificate. Exact human-contact matching can be inappropriate when morphology or grasp mechanics differ strongly. |
| [Learning from contact-rich planners](https://arxiv.org/html/2412.09743) | Demonstration consistency and action entropy affect downstream imitation learning. Physically feasible trajectories can still be poor supervision. | Some validation uses chunks reset to planned starting states. That is not uninterrupted episode execution. Borrow consistency-aware selection and direct learner evaluation, while keeping reset-assisted results separately labeled. |

The combined lesson is not to assemble every module from every paper. The smallest useful combination is **temporally stable interaction evidence, a few robot-surface hypotheses, and optimization through a known controller**. The unresolved scientific question is whether mechanics makes that combination cheaper and more reliable than a compact direct simulation search under equal budgets.

## 5. What our current code and saved experiments actually establish

There are two different implementation paths. They must not be conflated.

### 5.1 Historical full-episode correction

[full_episode.py](../../correction/src/optimization/full_episode.py) constructs contact targets, repairs robot joints with local IK/collision terms, and separately allocates forces from an estimated object wrench. It uses a fixed observed object path, derivative-based wrench estimates and short force-allocation windows. The hand target construction contains a fixed site-to-surface offset. This is the older separated design whose limitations we already discussed.

The full bin controller comparison replays this historical candidate; it is **not** a successful full-episode solve by the newer coupled optimizer. Its original planning report contains **zero force-feasible frames** under that planner's acceptance test. A QP returning a numerical solution did not make its residuals acceptable. [Historical report](../../correction/outputs/random_three_full_20261019/episode_000018/report.json)

Saved full-episode results, 858 frames / 28.57 s:

| Diagnostic | Original | Corrected + selected feedback | Interpretation |
|---|---:|---:|---|
| Supported G1 object-position RMS | 449.0 mm | 444.1 mm | Both fail |
| Supported G1 joint-acceleration RMS | 25.94 rad/s² | 100.19 rad/s² | Correction substantially worsens motion |
| Independent pads object-position RMS | 559.7 mm | 566.6 mm | Correction worsens tracking |
| Independent pads frames with contact | 65 | 63 | Planned sustained interaction is not delivered |

Sources: [G1 report](../../correction/outputs/g1_controller_full_20261019_v2/report.json), [independent-pad report](../../correction/outputs/support_points_full_20261019/report.json). Both physical comparisons preserve the initial state, run without episode resets and apply no direct object assistance.

A read-only inspection of the saved pad arrays localizes the failure:

- At 8.33 s, object error is approximately 5.7 mm and physical contact exists.
- By 9.33 s, there is no physical contact, although maximum pad-command tracking error is approximately 1.7 mm.
- At 10 s, object error is approximately 186 mm. The plan requests about 15–16 N on two pads; actual contact force is zero.
- At 13.33 s, object error is approximately 736 mm while pad-command tracking is about 0.1 mm.

This means the pads track their world-space commands while the object departs from the planned interaction. The contact gate suppresses force guidance after contact loss, while the world-space tracking controller has no adequate acquisition/recovery strategy. The simulator already delivers object forces through physical contacts. This failure does not prove that every alternative grasp or force profile is infeasible.

Another read-only audit compares saved nontrivial force vectors with the execution cone. Among 753 tip/frame vectors above 1 N, **517 exceed the controller's tangential limit** of 0.45 times normal force, allowing 0.01 N numerical tolerance. The allocator uses coefficient 0.6; the execution controller uses 0.45. Even at the intended orientation, projection can therefore change the planned wrench. This mismatch is actionable evidence, not proof that it alone explains the failed rollout. [Allocator](../../correction/src/dynamics/sparse_forces.py), [controller](../../correction/src/dynamics/contact_controller.py)

The servo adapter also limits its force-related joint offset, not the entire command. The saved candidate already violates joint limits; clipping and rate constraints conflict. The report records an offset-rate contract failure. Merely lowering the offset rate cannot repair an invalid nominal motion.

### 5.2 Newer coupled horizon implementation

[coupled_horizon.py](../../correction/src/optimization/coupled_horizon.py) does incorporate important parts of the revised reasoning: latent object motion, integrated dynamics, contact forces, robot splines and optional servo/torque constraints. It does not require raw observed frame acceleration as a target.

However, its current real-data adapter is book-specific: manually prescribed pickup timing, planar object patches, a few named contact hypotheses, and restricted initial environment support. It does not yet represent the bin's complete sliding/support/contact-transition problem. [Adapter](../../correction/src/preprocessing/g1_horizon.py)

The manufactured two-pad example is a useful mechanics test. The real G1 book candidate remains rejected for contact/geometry violations. Neither establishes a successful real-data manipulation reference. The relevant saved experimental scope and rejection reasons remain documented in [coupled horizon notes](COUPLED_HORIZON.md) and [controller execution notes](CONTROLLER_EXECUTION.md).

### 5.3 My implementation self-assessment

I progressed from force balance to robot replay without first establishing a physically attainable, controller-compatible contact strategy. I then added controller adaptations around a reference that was already rejected. This made the simulation more elaborate without resolving its central failure.

Specifically, I should have required:

1. Actual robot collision surfaces and a continuous approach to them, before treating hand sites as force application points.
2. One shared force/friction/actuator contract between planning and execution.
3. Checks on the complete applied command, including nominal motion, clipping, interpolation and feedback.
4. Separate labels for algebraic feasibility, physical replay success and learning benefit.

The existing diagnostics and rejection gates are worth keeping. The present saved full-episode correction is not an accepted reference, and the core research method is not finished.

## 6. Revised compact representation

Use a small number of interaction phases. For phase j, store:

| Variable | Meaning | Demonstration contribution |
|---|---|---|
| H_j | Contact-mode hypothesis: approach, push/slide, pinch, supported pivot, release | Temporal ordering and approximate intent |
| P_j | Candidate regions on the object | Object-frame hand proximity over an interval |
| S_j | Corresponding robot collision surfaces | Chosen using actual Dex3 geometry and reachability |
| I_j | Interval for onset/release, rather than an exact noisy frame | Motion and proximity evidence |
| z_j(t) | Few wrist/finger or joint command knots | Initialized near retargeted motion |
| X(t) | Latent object pose over time | Fit to observations within stated tolerances |
| f_j(t) | Robot-on-object contact-force plan | Chosen by mechanics and controller limits, not claimed to be measured human force |

What survives imperfect human motion is usually **relational and temporal**: which side the hand approaches, which region it remains near, whether two hands oppose each other, when support changes, and the overall object displacement/rotation. These are uncertain cues, not guaranteed labels. Repeated evidence in object coordinates can be stronger than one frame's nearest point.

The G1 may use a different surface or fewer contacts than the human. We preserve an interaction's object effect and demonstrated neighborhood, not a mandatory six-fingertip correspondence. For supported dragging, a push or hooked contact may be sufficient; lifting imposes different requirements. These alternatives need screening on the actual bin geometry before selecting one.

## 7. Optimization: couple observations, mechanics and executable commands

### 7.1 Observations are not force labels

Let observed object poses be O_k, human cues be Y_k, and retargeted joints be q_obs,k. Let latent object poses be X_k, robot configurations q_k, controller state c_k, and contact forces f_k. The controller state includes any filter or integrated-command memory.

Fit object motion over a horizon using a robust pose error:

$$
E_{obs}=\sum_k \rho\!\left(\left\|W_k\,\operatorname{Log}(O_k^{-1}X_k)\right\|^2\right).
$$

Here W_k scales translation and rotation by declared tolerances. These are not automatically calibrated statistical covariances. Use lower confidence for occlusion and inconsistent reconstruction; retain coherent long-term displacement and event evidence. In Tier 1 the clean reference can receive tighter tolerances. Tier 2/3 clean evaluation trajectories remain outside this optimization input.

The latent trajectory supplies dynamically consistent velocities and momentum through integration. Raw framewise second derivatives are not the demanded force trace. For an integration time step h, express velocity v, angular momentum L, forces and moments in the world frame. L is angular momentum about the COM; each r_i runs from the COM to its contact point:

$$
m(v_{k+1}-v_k)=h\left(mg+\sum_i f_{i,k}+f_{env,k}\right),
$$

$$
L_{k+1}-L_k=h\left(\sum_i r_{i,k}\times f_{i,k}+\tau_{env,k}\right).
$$

These are illustrative discrete balances; an implementation needs a consistent orientation/inertia integration and sufficient temporal resolution at contact transitions. Environment force must follow the selected support mode. It cannot be a free variable that supplies arbitrary assistance.

### 7.2 Choose a robot explanation, not a unique human force

For an active contact, require robot and object surfaces to meet, their normals to be compatible, and motion to obey the chosen sticking or sliding mode. For a unit inward object normal n_i:

$$
f_{n,i}=n_i^T f_i\ge 0,\qquad
\|f_i-f_{n,i}n_i\|\le\mu_i f_{n,i}.
$$

Add actual actuator limits, collision avoidance, and limits on applied commands. A force cone alone has no finite actuation bound. Sliding also requires a consistent friction/slip relation; allowing an arbitrary vector in the static cone is insufficient.

Multiple solutions remain possible. Select among them using demonstrational proximity, acquisition cost, smoothness, control effort and robustness margin. The objective is therefore not “recover the true human force”; it is “find a cheap executable robot explanation that preserves the observed task.”

A useful two-level design is:

1. Meet declared geometry, control and observation/goal tolerances, or return a diagnostic failure.
2. Within those tolerances, minimize command variation, effort and unnecessary contact changes.

This reduces dependence on one arbitrary weighted sum. Normalize soft errors by physical tolerances and test a small predeclared weight range. A low objective value is not an acceptance certificate.

### 7.3 Put the controller inside the problem

Let u_k be an executable position/residual command, s_k the physical robot/object state, and A the actual command adapter:

$$
(u^{applied}_k,c_{k+1})=A(u_k,s_k,c_k),\qquad
s_{k+1}=F(s_k,u^{applied}_k;\theta).
$$

Parameters theta contain geometry, mass/inertia, gains, friction, timestep and actuator limits. The adapter includes filters and saturation. Optimizing u and later executing a materially different A breaks the problem definition.

We should compare two small implementations on the same transition and budget:

- **Direct simulation baseline:** optimize a few smooth command knots through the actual simulator and adapter. Forces emerge from contact. This avoids pretending that a freely selected force is independently controllable, but can require many rollouts.
- **Mechanics-guided candidate:** use reduced dynamics, force allocation and geometry to initialize/rank a few candidates, then refine and validate through the same adapter and simulator. Its purpose is to reduce expensive search, not replace physical validation with a model certificate.

A quasi-static or quasi-dynamic approximation can initialize a slow supported phase. Lift, impact, slip transition and full replay still require appropriate dynamic checks. Start with short windows containing approach and contact transition; carry the resulting physical state forward and maintain overlap consistency. Do not reset each window to the desired object state in the acceptance demonstration.

### 7.4 Feedback needs both contact maintenance and task progress

After a small object error, maintaining contact requires reacting to the actual object. Continuing the original world-space hand target can abandon it, as in our bin replay. Conversely, rigidly attaching the target to the measured object can preserve contact while making no task progress.

The feedback law should therefore combine bounded object-relative contact maintenance with the desired object displacement/rotation over the next horizon. It must handle separation, saturation and release explicitly. This can begin with a classical local controller; no learned force module is required before testing that hypothesis.

For a PD joint controller, an equilibrium offset can produce contact torque. If f means robot-on-object force, the torque-equivalent increment relative to a compensated nominal servo target is:

$$
\Delta q=K_p^{-1}J^T f.
$$

Here J is evaluated at the actual contact configuration; multiple contacts use stacked Jacobians and forces. This relation assumes compatible gains and compensation, an unchanged nominal velocity target, and no lost torque from clipping. The same contact torque must not also be added separately. Differentiating the preload offset into the velocity target changes the damping contribution and breaks this simple equivalence. It is not a displacement to add unquestioningly to an invalid geometric reference, or a guarantee that the finger reaches the surface and delivers f. Position-command interfaces can exert forces through impedance; our problem is insufficient modeling/control of that interaction, not an inability of position control to generate force.

## 8. Delta actions and continuity

**Delta actions are useful, but “delta” has several distinct meanings.** They are not automatically better than a smooth absolute target.

| Interpretation | Formula | Consequence |
|---|---|---|
| Default-pose offset | q_cmd,k = q_default + S a_k | Still an absolute target; no temporal integration |
| Accumulated command increment | q_cmd,k = q_cmd,k-1 + h v_k | Bounded v limits command increments; drift and windup need handling |
| Offset from measured state | q_cmd,k = q_measured,k + S a_k | Bounds servo error, not command speed |
| Reference residual | q_cmd,k = q_ref,k + r_k | Preserves demonstration anchoring; smoothness depends on both terms |

Isaac Lab's `RelativeJointPositionAction` uses the **current measured joint position** at each application, not the preceding command. [Official action source](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py)

The V2D reference-residual adapter already uses a filtered residual added to a reference joint target. [Pinned adapter](https://github.com/nvidia-isaac/video_to_data/blob/0593d80de5d39ab6f56f73ed16ca464ba5053499/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/mdp/actions/reference_residual_action.py) SONIC's body residual is added in its action space before per-joint scaling and default-pose conversion; finger residuals use a different reference mapping. A body action scale is therefore not a uniform radian offset. [SONIC adapter](https://github.com/nvidia-isaac/video_to_data/blob/0593d80de5d39ab6f56f73ed16ca464ba5053499/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/mdp/actions/sonic_joint_residual_action.py)

Define the command increment as Delta q_k = q_cmd,k - q_cmd,k-1. Here h is the applied-command update period, which may differ from both video sampling and the physics timestep. Useful componentwise discrete bounds are:

$$
|\Delta q_k|\le h v_{max},\qquad
|\Delta q_k-\Delta q_{k-1}|\le h^2 a_{max}.
$$

A jerk bound similarly constrains the third finite difference by h cubed times the jerk limit. Appropriate interpolation and checks between samples are still required.

**Example:** at 50 Hz, alternating increments of +0.01 and -0.01 rad satisfy a small delta limit. Yet their interval velocities alternate between +0.5 and -0.5 rad/s, producing a finite-difference acceleration of 50 rad/s². Small deltas alone do not remove shaking.

For our method I recommend a smooth nominal reference plus a bounded residual, whose update rate is controlled. Then enforce limits on the **complete applied command**:

$$
\Delta q_{cmd,k}=\Delta q_{ref,k}+\Delta r_k.
$$

This is precisely why smoothing only our present force offset is insufficient. Include the filter in optimization/replay: its delay can remove preload at the moment contact is needed. Integrated commands should use the post-saturation applied state and anti-windup; filter/integrator state must be available to a later policy or its history.

For wrist SE(3) increments, specify whether the twist is body- or world-expressed and use the corresponding exponential-map composition. Small wrist increments do not prevent IK branch switches or joint collisions.

[CAPS](https://ai.bu.edu/caps/) supports explicit temporal/spatial policy regularization, while [Ruckig](https://arxiv.org/abs/2105.04830) provides useful derivative-constrained trajectory-generation ideas. Neither makes arbitrary contact timing safe to retime independently. Smooth command signals also cannot eliminate every physical acceleration discontinuity caused by impact or slip.

Preserve the official external action interface initially. Compare absolute residuals against accumulated residual updates later with identical gains, limits, observations and filtering. Otherwise a purported delta-action benefit may actually be a controller or rate-limiter change.

## 9. Revised milestones and acceptance evidence

These are the complete milestones, not a promise that all fit the remaining challenge schedule. The first implementation batch should be bounded to M0–M3 plus the early compatibility check.

| Milestone | Purpose and work | Evidence required before promoting it |
|---|---|---|
| **Early parallel compatibility check** | Pin a compatible official runtime/evaluator pair; inspect one adapter and perform a bounded pretrained/SONIC physical replay; measure memory/disk before training. | Correct joint order, timing, world frame, contact settings and action semantics; known evaluator route and body-controller interaction. No full baseline training required. |
| **M0: failure and model contract** | Audit first contact loss; unify mesh surfaces, force frames, friction, mass/COM/inertia, gains, limits and rates. | Planned-versus-delivered force/contact timeline; reproducible failure; known model differences. |
| **M1: compact interaction evidence** | Extract uncertain object regions, normal/approach directions and phase intervals from the demonstration. | Human evidence explains each region and phase; no clean-tier leakage. |
| **M2: robot contact hypotheses** | Build a few Dex3-compatible configurations near those regions, including approach and arm collision checks. | Actual collision surfaces meet correctly; joint continuity/reachability checked; no fabricated initial grasp. |
| **M3: task-wrench screening** | Rank candidates using uncertain horizon demand, environment support and actuator limits. | Necessary feasibility/margins stated with assumptions. No hard rejection based on noisy observed acceleration alone. |
| **M4: acquisition and persistence** | Execute a short approach/contact/hold-or-slide transition with the real hand and chosen nominal feedback controller. | Physical contact is acquired and sustained from the actual prior state; no direct object force. |
| **M5: controller-aware correction** | Compare compact forward-simulation search with mechanics-guided search under the same budget. | Better delivered object motion and valid complete commands; all search/build/replay time counted. |
| **M6: feedback robustness and recovery** | Extend the controller already used in M4–M5 to handle deviations, saturation and separation while preserving task progress. | Small declared perturbation tests; no concealed reset or oracle correction. |
| **M7: complete bin episode** | Carry the physical state through every phase. | Full uninterrupted original-versus-method comparison, intended trajectory visible, no assistance. Both robot and pad diagnostics remain clearly labeled. |
| **M7.5: whole-body controller compatibility** | Replay original/corrected references through the actual pretrained SONIC/action interface; remove the fixture, restore feet, measure delivered contact and saturation. Joint boundary owned by either correction or training agent. | Explicit joint/control ownership, complete unassisted zero-residual replay, separate planned/measured forces, failure feedback into correction. See [handoff](WHOLE_BODY_HANDOFF.md). |
| **M8: batch/generalization cost** | Freeze settings, then test several episodes and input qualities. | Success/failure coverage, per-episode cost distribution, p90 cost and manual interventions. |
| **M9: policy-cost experiment** | Same learner, matched controller and training budget; compare references and controller ablations. | Cost to the same tracking threshold, learning curves and paired seeds. **Start a small pilot after a useful M4/M5 improvement, in parallel with M6–M8.** |
| **M10: Tier 3 uncertainty** | Integrate reconstruction outputs and quantify geometry/pose uncertainty. | Correct schema/world frame; no prohibited Tier 1 information; reconstruction cost included. Preparation can proceed in parallel. |
| **M11: official whole-body evaluation** | Integrate the chosen reference/control outputs with learned policy and whole-body controller; pack each tier. | All required executed trajectories pass official validation and metrics. Supported-pelvis results do not substitute for this. |

M3 is a screening step, not a claim that static feasibility proves dynamic execution. If M2 cannot find a plausible approach, adding a larger force NLP is unlikely to help. If M5 does not beat the compact simulation baseline, reduce model complexity or change the hypothesis rather than accumulating more penalties.

**20 October ownership update:** the user assigns M8–M11 to another agent.
This chat retains correction and the M7.5 interface handoff, plus the two or
three additional exploratory episode comparisons explicitly requested.
Those comparisons are not a completed generalization or policy-cost benchmark.

## 10. Experiments that would establish or reject the contribution

The primary comparison remains original retargeting versus our method. Start with original, geometry-only and mechanics-corrected references under one controller. Stage the other comparisons where needed to explain an improvement; they are not all prerequisites for the first useful result and do not require more viewer channels.

| Variant | What it isolates |
|---|---|
| Original reference + common controller | Practical baseline |
| Geometry/continuity cleanup + common controller | Benefit from ordinary smoothing/contact geometry |
| Original reference + proposed feedback | Benefit from controller change alone |
| Geometry/continuity cleanup + proposed feedback | Geometry comparator with the full method's controller |
| Mechanics correction + proposed feedback | Full proposed method |
| Compact direct simulation optimization + same feedback | Whether the mechanics actually saves search cost |

Hold learner, reward functional form and weights, physical model, reset distribution, observations, training/evaluation controller and quality threshold fixed where appropriate. The reference is an intentional experimental variable; clean evaluation targets stay fixed and separate. Additional force rewards or changed termination rules need their own ablation. A force-guided controller added only to one reference is a confounded test of trajectory correction alone.

Measure:

$$
C_{total}=C_{preprocess}+C_{search}+C_{correction}+C_{validation}+C_{training}+C_{evaluation}.
$$

Report CPU-hours, GPU-hours, elapsed time, memory and monetary cost separately; unlike resources should not be added as if interchangeable. Include failed candidate searches, solver construction and cache misses. Report both first-run cost and amortized cost when geometry/contact caches are reused.

Learning cost means time/steps to a **common** object-tracking quality, not a fraction of each run's own final reward. Report runs that never reach the target as censored failures within the budget. Pair seeds when practical and distinguish a one-seed pilot from evidence across seeds/tasks.

Keep tier inputs, optimized latent targets, predicted physical trajectories, executed trajectories and clean scoring references separate. Correcting the observation model must not quietly change what counts as successful tracking.

Evidence against our hypothesis would include:

- A geometry-only or direct simulation baseline achieves the same quality at lower total cost.
- Mechanical margins do not predict acquisition, persistence or learning benefit.
- Correction saves training steps but consumes more total compute than it saves.
- The method works only with privileged clean poses or hand-tuned contact labels unavailable at deployment.
- Conservative screening rejects most episodes, or the fixed template set cannot cover their interactions.

These are useful outcomes for deciding where to invest; they should not be hidden behind unit-test counts.

## 11. Implementation handoff and cost boundaries

Every candidate should carry immutable input/version IDs; units and frame conventions; object COM/inertia; selected robot/object surface IDs; phase timing/uncertainty; controller gains, rates, filters and limits; and explicit accepted/rejected status with reasons. Store desired, commanded and measured motion/force separately.

Use a small candidate set and a hard total wall-clock budget that includes model construction and all retries. Reuse robot geometry and object contact-region caches. Do not expand immediately into an unrestricted whole-episode contact-implicit optimizer. Numerical solve time is not the entire optimization cost.

No CPU-seconds or 4090-hours success guarantee is justified yet. The existing full pad replay is inexpensive enough for local diagnosis, but it failed. Existing short nonlinear solves demonstrate that reduced variables alone do not guarantee cheap convergence.

**The next concrete implementation objective:** for the bin's first lost-contact interval, produce a few demonstration-supported Dex3 contact hypotheses, screen them under a shared model/controller contract, and test whether one can be acquired and maintained. Then evaluate compact command optimization against mechanics-guided correction on that same interval. Preserve the full failed episode and fallback outputs for comparison.

This is the preparation boundary. The next code change should implement that experiment, not claim that the final scalable policy pipeline already exists.

## 12. Version record and unresolved external dependencies

- Official repository snapshot: `0593d80de5d39ab6f56f73ed16ca464ba5053499`.
- Submission kit SHA-256: `8d2bce64ecab6da4eee938fe2859bd60e767535af30068b386e2dc20a72cf43e`.
- Kit evaluator manifest commit: `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.
- These two recorded commits are not a compatibility certificate. The early integration check must establish that the chosen runtime and kit evaluator work together.
- Detailed read-only working notes: [challenge/CHORD](../../correction/outputs/review_20261019/challenge_chord_notes.md), [mechanics](../../correction/outputs/review_20261019/mechanics_notes.md), [actions](../../correction/outputs/review_20261019/delta_action_notes.md). These are local generated notes, not required dependencies of the method.

Still unverified: a working 24 GB training configuration; precise Tier 3 reconstruction cost; end-to-end official evaluation of our outputs; acceptance/interface details for custom nonstandard policies or analytic-only controllers. The plan should not depend on analytic-only submission eligibility without clarification. No organizer messages were sent during this review.
