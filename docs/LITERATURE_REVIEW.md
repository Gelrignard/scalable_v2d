# Trajectory correction for efficient contact rich manipulation learning

Literature review and research positioning for NVIDIA V2D Track 2. Sources checked through **17 October 2026**. The companion [annotated reading list](LITERATURE_REFERENCES.md) groups the papers by their role in the project.

For the current implementation audit, submission interface and revised milestones, see the [19 October research review](correction/RESEARCH_REVIEW_2026-10-19.md). The implementation description below is a historical snapshot.

**The project has a plausible research direction, but inserting contact-aware trajectory optimization before policy learning is already established.** The most promising contribution is a specific, testable connection between errors in reconstructed interaction, the repair needed by a particular robot/controller, and the total compute required to obtain a successful policy. Adding friction cones, object-relative contacts, a residual policy, or a generic uncertainty penalty would not by itself establish novelty.

My recommendation is to investigate **when correcting an uncertain contact sequence actually makes it cheaper to learn**. Start with a compact correction that couples attained robot contacts to object wrench demand and temporal contact behavior. Test whether its diagnostics predict the benefit of correction beyond geometry-only repair, simple smoothing, and simulation-based retargeting. If this relationship holds on held-out tasks, use it to allocate a limited correction budget. These are research hypotheses, not claims that the combination is unprecedented or that publication is assured.

## 1 What the challenge makes scientifically useful

Track 2 fixes the embodiment to G1 + Dex3 and provides three input-quality tiers: clean multi-view motion, synthetic corruption, and reconstruction-derived motion. All tiers are evaluated against clean object trajectories, and a valid entry requires submissions for all three. The challenge explicitly permits improving the estimated object trajectory for noisy tiers while excluding the clean Tier 1 captions as correction inputs. Its scoring uses AUC, SP-SR, MP-SR and MPPE, rather than training cost. [Official challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)

This gives the project two separate objectives:

| Objective | Appropriate evidence |
|---|---|
| A strong challenge entry | Official tracking metrics for all tiers under the prescribed evaluation |
| A convincing research paper | A reproducible mechanism that improves the performance–compute tradeoff, with controlled baselines and failure analysis |

A leaderboard improvement would not establish reduced training cost. Conversely, matching baseline tracking quality with substantially less total compute could be a valuable paper result even without winning the leaderboard.

The attractive experimental axis is **reconstruction quality × contact difficulty × learning cost**. It lets you ask which errors matter for learning, rather than treating every millimeter of pose error as equally important. A small contact-normal error can change the available force directions; a larger free-space wrist error may be easily absorbed by the learner. That is a mechanism to investigate, not an assumption to build into the conclusion.

## 2 What the repository currently establishes

The audit below uses source and existing saved outputs. No new simulation or training was run for this review.

The implemented chain is:

1. Detect candidate fingertip contacts using object-relative distance/speed hysteresis.
2. Smooth object motion and use Newton–Euler equations to estimate required object wrench.
3. Optimize planar contact locations over a selected phase, with an inner force-allocation QP and outer contact-position optimization.
4. Project the targets into arm/hand configurations using local IK, with larger weight on normal closure and weights derived from planned force magnitude.
5. Audit the contacts actually attained by robot forward kinematics.

The implementation is a useful reduced model. It currently assumes a manually selected sticking phase, provisional physical parameters, and a rigid object with book-specific planar patches. Force balance is penalized and checked afterward. The optimizer does not solve full robot dynamics or command the inferred forces. Its contact planner and robot repair are one-way stages. See [method](correction/METHOD.md), [contact optimizer](../correction/src/optimization/contact_optimizer.py), [robot repair](../correction/src/optimization/robot_trajectory.py), and [contact audit](../correction/src/dynamics/contact_audit.py).

The recorded 41-frame book phase shows:

| Quantity | Before | After | Interpretation |
|---|---:|---:|---|
| Mean absolute fingertip surface gap | 10.76 mm | 0.40 mm | Much better contact closure |
| Frames passing instantaneous attained-contact wrench check | 0/41 | 41/41 | A force allocation exists under the audit model |
| Mean object-relative fingertip speed | 13.5 mm/s | 45.2 mm/s | The intended temporal sticking behavior remains unresolved |
| Accepted as feasible manipulation | No | No | No execution certificate |

The speed comparison averages all six tips over consecutive frames of the selected phase, including tips that were not physically in contact before repair; it excludes the surrounding taper intervals. It is **not** a measured force-weighted slip speed. It nevertheless exposes the distinction between per-frame closure and persistent contact. The audit's acceptance flag is deliberately false because its checks are incomplete. [Recorded results](correction/METHOD.md), [audit implementation](../correction/src/dynamics/contact_audit.py)

There is an even more important causal issue: the saved target-level wrench plan was already within force/torque tolerance before moving the contact locations. The improvement from zero to 41 feasible frames occurs when auditing the attained robot geometry after repair. Thus this example does not demonstrate that moving contact anchors using inverse dynamics rescues an infeasible grasp. A geometry-only repair with identical closure, collision, and continuity settings is essential. The relevant local artifact is `correction/outputs/tier1_book/wrench_plan.json`; generated outputs are ignored by Git and may be absent in another checkout.

The repository also derives wrench demand from smoothed motion while preserving raw object poses in playback. A small wrench residual is therefore not a certificate for tracking every raw pose fluctuation. `training/` is an implementation assignment; the five-episode Tier 2 manifest is marked `planned_not_run`. There is currently no measured policy-training saving. [Training status](../training/README.md), [pilot manifest](../experiments/tier2_pilot.json)

These observations suggest a practical first question: **does mechanics contribute beyond closing contact gaps carefully?** Answering that question is more useful than adding a larger model before establishing the effect.

## 3 The literature map

### 3.1 Contact mechanics and trajectory optimization

The foundations already provide the core ingredients: grasp maps, friction-limited wrench sets, unilateral contact, and trajectory optimization with changing contact modes. Posa, Cantu and Tedrake formulate contact-rich trajectory optimization without prescribing every contact event. Mordatch and colleagues develop contact-invariant optimization for hand manipulation. These establish that optimizing motion and forces together, including making contact choices, is not a new contribution. [Posa et al.](https://doi.org/10.1177/0278364913506757), [Contact-Invariant Optimization for Hand Manipulation](https://homes.cs.washington.edu/~todorov/papers/MordatchSCA12.pdf)

Your reduced contact-patch formulation makes a reasonable computational tradeoff: use the demonstration to restrict the search rather than rediscover the entire manipulation. The question is whether the reduction retains the constraints that determine learning difficulty. A fixed sticking phase can make a small optimization tractable; it also removes the very transitions where pickup, sliding, regrasping, and placement often fail.

Robust contact optimization is also established. Robustness to mass or friction uncertainty, worst-case objectives, and contact-mode planning each have their own literature. They are useful components, but the paper must show why a particular uncertainty model and repair mechanism help with reconstructed demonstrations. See the control references in the [reading list](LITERATURE_REFERENCES.md).

### 3.2 Optimized trajectories as supervision for learning

Guided Policy Search established a broad route from trajectory optimization to policy learning. DAPG showed the value of demonstrations for dexterous RL. Opt2Skill combines dynamics-aware trajectory optimization with RL for humanoid loco-manipulation. These are precedents for using model-based structure to reduce the burden on learning. [Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html), [DAPG](https://www.roboticsproceedings.org/rss14/p49.html), [Opt2Skill](https://opt2skill.github.io/)

A useful distinction is what the optimization produces. A state trajectory can guide rewards or initialization; an action trajectory can supervise behavior cloning; a local feedback controller can generate perturbed rollouts. Those are different learning interfaces. Your current exporter produces references, so a policy-data-generation claim would require additional controller-driven execution and action logging.

DemoStart uses demonstrations to create a learning curriculum. Scaffolding Dexterous Manipulation reduces the difficulty of learning using VLM-generated coarse motion scaffolds. These matter because a stronger learner or initialization scheme may remove the apparent benefit of preprocessing. Your correction should first be compared using the same learner, then checked with at least one meaningful alternative learning setup. [DemoStart](https://arxiv.org/abs/2409.06613), [Scaffolding Dexterous Manipulation](https://arxiv.org/abs/2506.19212)

### 3.3 Human interaction transferred through physics

DexMV, D-Grasp, ManipTrans, DexMachina, DexTrack, PhysHOI and InterMimic form a closely related family, with different combinations of human references, contact guidance, residual control, tracking policies, curricula and distillation. They make human-to-robot interaction transfer an established problem with strong baselines. The [annotated list](LITERATURE_REFERENCES.md) separates their assumptions and relevance.

QuasiSim explicitly addresses the tension between an easy-to-optimize contact model and faithful physics, gradually moving through a simulator curriculum. This is important for your small QP: a cheap approximation can be valuable, but success in that approximation must survive the target simulator. [QuasiSim](https://meowuu7.github.io/QuasiSim/)

Do not directly compare reported speedups across these papers. They can refer to per-sequence trajectory generation, one policy's training, amortized generalist training, or data collection. A fast replay generator and a robust closed-loop policy solve related but different problems.

### 3.4 Contact correction and interaction reconstruction

ContactOpt refines hand poses to match expected contact. TOCH learns an object-centric temporal representation to refine erroneous interaction sequences. Thus neither contact correction nor using object coordinates to suppress interaction artifacts is new. [ContactOpt](https://openaccess.thecvf.com/content/CVPR2021/html/Grady_ContactOpt_Optimizing_Contact_To_Improve_Grasps_CVPR_2021_paper.html), [TOCH](https://arxiv.org/abs/2205.07982)

Lakshmipathy, Hodgins and Pollard also transfer contact-rich anthropomorphic manipulation through contact-area correspondence, shape matching and IK. This is a particularly relevant geometric predecessor: a weak fingertip-only baseline would miss stronger existing ways to preserve interaction through pickup, reorientation and release. [Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations](https://doi.org/10.1145/3723872)

An earlier IROS paper already used approximate differentiable physics to reconstruct object manipulation from video and obtain robot motions without costly RL. Its simplified setting differs from G1 + Dex3, but it directly limits a broad claim that physics between vision and execution is novel. [Learning Object Manipulation Skills from Video via Approximate Differentiable Physics](https://www.cs.toronto.edu/~makarand/papers/IROS2022_PhysReal2Sim.pdf)

Grasp-and-Lift uses a low-dimensional spline representation and simulator-in-the-loop black-box optimization to refine hand-object trajectories. C2Dex aggregates contacts in canonical object coordinates, refines human motion, retargets interaction geometry, and applies residual RL. In the inspected C2Dex formulation the object trajectory stays fixed; deployment uses generated trajectory replay. These are particularly close comparisons for a proposed correction layer. [Grasp-and-Lift](https://arxiv.org/html/2601.18121v1), [C2Dex](https://arxiv.org/html/2608.07045v2)

The remaining opportunity is not merely making an interaction look smoother or closer to contact. It is identifying which uncertain estimates should change, preserving the intended task, and showing the resulting improvement in downstream learning cost.

### 3.5 Whole-body retargeting before RL

CoRe already uses contact-aware motion refinement before RL. OmniRetarget preserves interactions with objects and the environment. Direct Dynamic Retargeting optimizes in simulation from video-derived task-space references and explicitly reports improved downstream RL convergence. These prevent a broad novelty claim based on pipeline order or on the general idea that physically feasible references facilitate learning. [CoRe](https://doi.org/10.1109/Humanoids65713.2025.11203055), [OmniRetarget](https://omniretarget.github.io/), [Direct Dynamic Retargeting](https://arxiv.org/html/2605.23762v1)

Their existence does not eliminate a dexterous-manipulation paper. Sustaining finger-object forces and resolving uncertain contact transitions differs from reproducing whole-body poses and ground contacts. However, the distinction has to appear in the algorithm and experiments; changing the robot or benchmark alone is a weak argument.

### 3.6 The closest direct competitors

| Work | What overlaps with this project | What a useful comparison must establish |
|---|---|---|
| CHORD | Human interaction, object-centric wrench guidance, RL | Additional value from correction using the same downstream learner |
| PhysicsGen | Retargeting, trajectory optimization, policy training data | Benefit of your correction on noisy references and its total cost |
| SPIDER | Physics-based retargeting, robust rollout optimization, contact handling | Better quality per correction budget or lower eventual policy cost |
| C2Dex | Stable object contacts, interaction correction, retargeting, residual RL | Benefit beyond canonical contact stabilization and geometric refinement |
| Direct Dynamic Retargeting | Physically viable video-derived references that help RL | Dexterous contact-specific mechanism and fair preprocessing accounting |
| Grasp-and-Lift | Demonstration-preserving physics-in-the-loop refinement | Advantage over compact black-box simulation optimization |
| CoRe and OmniRetarget | Contact-aware retargeting/refinement before learning | More than adding contact objectives to IK |

**CHORD is especially important to understand precisely.** Its guidance compares approximate wrench capabilities induced by human and robot contact configurations, using friction-cone primitives and directional supports; this is not an exact multi-contact feasible net-wrench polytope. It differs from estimating the particular wrench required by a trajectory's Newton–Euler dynamics. Its noise ablation finds corrupted contact guidance problematic, and it already reports an association between learned wrench reward and success. Your possible distinction is a *pre-learning* diagnostic of repair benefit or time to success, not merely another contact score correlated with final performance. [CHORD, Sections 3.1, 4.3 and Appendix A.1](https://arxiv.org/html/2607.00033v2)

**PhysicsGen is a direct architectural precedent.** It converts sparse human demonstrations through retargeting and trajectory optimization into physically consistent data, then trains diffusion policies. A compact correction front end needs an advantage in the noisy-input, fixed-embodiment regime, rather than claiming the pipeline itself. [PhysicsGen, RSS 2025](https://www.roboticsproceedings.org/rss21/p053.pdf)

**SPIDER is the strongest warning against an overly broad extension.** Its inspected September 2026 revision includes simulation-based retargeting, contact guidance/filtering, robustness treatment and increased optimization effort on difficult segments. Its limitations discuss perception errors, gaps and semantically incorrect contact modes. Consequently, neither filtering contacts, randomizing physics nor retrying difficult segments is a sufficient novelty claim. Its headline speed advantage concerns trajectory generation; it should not be reused as an end-to-end policy-training speedup. [SPIDER v3](https://arxiv.org/html/2511.09484v3)

DexAgent is a further recent precedent for a video-to-simulation-to-robot pipeline that includes trajectory optimization and verification. Adding an agent or an automated verification loop would introduce substantial engineering without clearly separating your core mechanics contribution. [DexAgent, September 2026 preprint](https://arxiv.org/abs/2609.35318)

## 4 Five distinctions that should shape the paper

**A contact point near a surface is not an executable contact.** The robot must approach it without collision, establish it through its controller, sustain an admissible force, and release or slide appropriately. Per-frame force allocation alone does not establish this sequence.

**Wrench capability is not wrench demand.** A grasp can support many directions without supporting the required magnitude under real actuation limits. Conversely, a push can accomplish a task without force closure. A task-specific correction should not require force closure during every phase.

**Physical uncertainty is not reconstruction uncertainty.** Randomizing friction and mass addresses uncertain dynamics. It does not resolve an incorrect object pose, a missing hand trajectory, ambiguous contact timing, or the wrong interaction mode. Treating all these errors as isotropic pose noise loses the structure of the problem.

**Feasibility is not ease of learning.** A trajectory may be physically possible but require a narrow basin of initial states, large preload, precise timing, or controller gains that the policy cannot reliably exploit. A slightly less accurate reference may yield a much easier learning problem. Establishing that tradeoff is an empirical research opportunity.

**Improved references are not automatically cheaper policies.** Preprocessing can move effort out of RL while increasing total cost. It can also overfit an assumed simulator. Measure end-to-end cost at matched evaluation quality, not just training iterations or the optimizer's residual.

## 5 Candidate research directions

### 5.1 Couple wrench demand to attained temporal contacts

This is the closest extension to the existing implementation. Replace the one-way contact-plan → IK projection with a small alternating or sequential optimization that uses the robot's attained contact geometry.

For an intended sticking material contact, let

$$s_{i,t}=R_t^\top\bigl(x_i(q_t)-p_t\bigr).$$

Then constrain or penalize changes in $s_{i,t}$ during that mode, allocate forces at the attained locations, and recompute feasibility after updating the robot trajectory. Use physical surface contacts or pad geometry: fingertip-center motion is only a proxy and can misclassify rolling.

An illustrative objective is

$$\min_{q,f,z,T}\; L_{\mathrm{observation}}(T,q)+\lambda_q L_{\mathrm{reference}}(q)
+\lambda_w\sum_t\|G(q_t,T_t,z_t)f_t-w_{\mathrm{req}}(T)_t\|_W^2
+\lambda_v L_{\mathrm{mode}}(q,T,z)+\lambda_r L_{\mathrm{regularity}}.$$

Here $z$ specifies contact modes, $T$ is object motion, and $W$ normalizes force and torque units. For supported phases, $Gf$ must include environmental contact wrenches, or known environmental wrenches must be subtracted from the demand. In the first implementation, keep $T$ fixed and select a small set of phase hypotheses. Add friction, unilateral forces, collision and joint limits. Match physical parameters to the simulator. Include actuation feasibility or an explicitly bounded approximation; arbitrary contact force caps do not establish realizable robot torques.

Use short controller-driven rollouts to check which planned corrections survive execution. Optimize controller targets or account for their tracking error where necessary. A nominal force solution is not automatically the force generated by position control.

**Possible scientific claim:** attained temporal contact constraints explain and reduce learning failures left by geometric repair. **Main risk:** this remains a conventional optimization refinement unless it produces a clear, repeatable learning benefit and compares well with physics-in-the-loop methods. This is the best first implementation experiment, not the strongest standalone novelty claim.

### 5.2 Jointly repair uncertain object motion and contact modes

This is the most natural Track 2 extension. The current method treats object motion as authoritative, although noise in its second derivative can create large fictitious wrench demand. A corrector can otherwise contort the hand to explain a trajectory that is itself wrong.

Represent a latent object trajectory on SE(3), with observation weights derived from visibility and calibrated error estimates. Permit bounded changes to translation, orientation and contact timing. Distinguish approach, stick, slide, support and release; avoid imposing sticking through a genuine regrasp or tabletop slide. Add environmental support contacts for pickup and placement instead of assigning every wrench to the fingers.

Do not assume that the most physically convenient trajectory is the intended task. Preserve reliable observations, event order, object-relative relationships and task endpoints. For example, a repair that leaves an object on the table may be stable and cheap while completely failing the demonstrated lift. Keep evaluation timestamps fixed; a timing adjustment is not permission to change the evaluator's target or score after temporal realignment.

The inference is also underdetermined: motion with known physical parameters determines net wrench, not unique finger forces. Allowing mass, friction, object motion and all contact forces to change freely can produce many equally plausible explanations. Use known simulator parameters where available, constrain the remaining uncertainties, and describe inferred forces as model-dependent allocations rather than recovered human force measurements.

Compare a small collection of discrete contact hypotheses or use a mode inference method. A single Gaussian confidence ellipsoid cannot faithfully represent two plausible grasp modes. Sliding requires a mode-consistent friction law, not only a force cone with zero tangential velocity.

**Possible scientific claim:** jointly correcting reference uncertainty and contact mode avoids over-repair caused by noisy dynamics estimates, thereby reducing learning cost. **Closest threats:** TOCH, C2Dex, Grasp-and-Lift, SPIDER and physics-based reconstruction. The distinction must exceed temporal filtering or randomizing friction. It needs a better error model, a demonstrable inference mechanism and downstream evidence.

### 5.3 Predict when repair is worth its cost

This is the direction most tightly aligned with your stated motivation. Treat correction as a resource whose value depends on the episode and corruption, rather than assuming every reference should receive the same repair.

Construct inexpensive diagnostic features before policy training: normalized demanded-wrench deficit at attained contacts, persistence of intended modes, distance to friction/actuation limits, disagreement between plausible reconstructed trajectories, and failure in a short controller rollout. Test which features predict the *incremental benefit of correction* on held-out tasks.

A demand-based feature might be

$$d_t(\xi)=\min_{f\in\mathcal F(q_t,z_t;\xi)}
\left\|G(q_t,T_t,z_t;\xi)f-w_{\mathrm{req}}(T;\xi)_t\right\|_W,$$

where $\xi$ represents a stated reconstruction/physics hypothesis. Summarize its distribution across plausible hypotheses and time, rather than claiming that one nominal zero residual proves robustness. Near-zero deficit also does not measure available reserve; reserve and rollout sensitivity should be separate features.

The decision can be simple: accept, perform local geometric repair, perform coupled mechanics repair, or use a bounded simulation optimizer. Estimate whether expected downstream savings exceed the additional repair cost. Calibrate on separate development tasks; include calibration and probing cost in the accounting.

**Possible scientific claim:** a physically interpretable pre-learning diagnostic identifies which repair will improve the performance–compute tradeoff. **Main risk:** the diagnostic may not generalize or may simply correlate with task difficulty. Predicting that a hard task trains slowly is insufficient; it must distinguish which intervention helps. Compare with random allocation, constant budgets, geometric-error ranking and SPIDER-style difficult-segment retries.

This is a hypothesis about an experimentally testable gap in the reviewed work, not proof of an untouched literature niche. It is stronger when paired with one effective correction mechanism from 5.1 or 5.2.

### 5.4 Learn or amortize a correction operator

If optimization works across enough tasks, train a model to propose contact patches, mode hypotheses or a good optimizer initialization. Keep physical checks and allow rejection. Measure the number of new trajectories needed to repay the cost of generating training data and fitting the model.

This could improve scaling, but it is a poor first step while correction itself has not demonstrated a learning benefit. DexTrack, generalist tracking methods and recent force-aware retargeters already make amortization a competitive area. A learned corrector without a distinct target or useful guarantee is unlikely to be a strong addition.

### 5.5 Choose one central direction

| Direction | Fit with current code | What could make it a paper | Primary risk |
|---|---|---|---|
| Attained temporal contact correction | High | Causal improvement beyond matched geometry repair | Conventional optimization engineering |
| Joint reference and mode uncertainty | Medium | Better recovery under structured corruption and lower learning cost | Wrong task inferred from ambiguous data |
| Predictive allocation of repair compute | Medium after a working trainer | Held-out prediction of intervention benefit and net savings | Proxy does not generalize or probing erases savings |
| Amortized learned correction | Low initially | Favorable break-even at scale | Large data/training bill and strong existing competitors |

My preferred sequence is **5.1 as the minimum effective mechanism, 5.2 as the noise-handling extension, and 5.3 only after a measurable effect exists**. Do not promise all three as independent contributions. A focused paper about uncertain contact repair with convincing compute evidence is stronger than several weakly connected modules.

## 6 An experiment that can support or reject the idea

### 6.1 Isolate the correction before comparing complete systems

Use identical simulator, assets, controller, network, observations, reward terms, initialization distribution, training schedule, evaluation schedule and hardware for the primary comparison. Document exactly which reference channels change: robot configuration, object motion, contact targets, rewards derived from contacts, and reset states.

| Variant | Purpose |
|---|---|
| Official baseline as actually configured | Challenge reference; do not call a body-only adapter a full CHORD reproduction |
| Your contact-aware retargeting | Isolate the additional correction stage |
| Smoothing and interpolation only | Test whether denoising explains the improvement |
| Geometry-only repair | Same collision, continuity and normal-closure budget, without wrench guidance |
| Fixed anchors with force-based weights | Test whether force weighting alone matters |
| Optimized anchors with uniform weights | Test whether anchor relocation matters |
| Current full correction | Establish the existing method's effect |
| Proposed temporal/uncertainty extension | Isolate the new mechanism |

Run the detailed mechanistic ablations on a representative subset rather than multiplying every condition across every task immediately. Then compare the strongest variants on the larger suite. Include a strong simulation-based repair baseline, preferably SPIDER or a carefully identified adaptation, under a comparable correction budget. If full reproduction is infeasible, label the adaptation and narrow the claim accordingly.

A second comparison may allow each complete system its preferred learner. Keep that separate from the causal same-learner experiment; otherwise changes to optimization, learning algorithm and compute become inseparable.

### 6.2 Use tasks that expose the claimed mechanism

The existing five-episode manifest is a sensible integration pilot. It is not enough evidence for broad contact-rich manipulation claims by itself. Expand according to contact structure, not just object count:

- Sustained unilateral and bimanual grasps under changing load.
- Pickup and placement, including transfer of support between table and hand.
- Sliding or pushing, where force closure and zero-slip constraints would be wrong.
- Regrasp or contact exchange, where timing and mode inference matter.
- A demanding precision task if compatible assets are available; identify this as an extension beyond the initial challenge roster when appropriate.

Report long sequences and failures, not only manually selected stable phases. If the method supports only rigid objects, state that scope and avoid an articulated/deformable manipulation claim.

Start with one episode to validate real simulated execution. Use the five-task pilot to estimate effect size and failure modes. For the paper, choose a larger held-out suite and enough paired seeds to resolve the observed variation; 3–5 seeds is a practical starting proposal, not a statistical guarantee. Separate objects/task families used to tune correction or diagnostics from those used to evaluate generalization.

### 6.3 Stress the right errors

Evaluate the released Track 2 conditions, then add controlled stress tests on permitted development data:

| Error | Mechanistic question |
|---|---|
| Position/orientation jitter | Does differentiation produce spurious wrench demand? |
| Correlated drift | Does local contact evidence recover a coherent interaction? |
| Occlusion and dropout | Can the method preserve multiple hypotheses rather than invent certainty? |
| Contact normal/region errors | Does the attainable wrench set become misleading? |
| Contact onset/offset errors | Can a small timing error break the task despite low pose error? |
| Mass, COM, friction or geometry mismatch | Does a nominal repair survive plausible modeling errors? |

Separate reconstruction errors from physics parameter errors. Report error severity and temporal structure. Independent Gaussian jitter alone is not an adequate proxy for occlusion, drift or wrong contact modes.

Use no clean Tier 1 captions in Tier 2/3 preprocessing, tuning or training. Calibrate uncertainty on separately permitted data or procedural simulations. If an oracle experiment is useful for diagnosis, keep it explicitly outside the challenge pipeline and never count it as an eligible result. Hidden clean evaluation information must not select repairs or hyperparameters.

### 6.4 Measure total cost at matched quality

For each task and seed, predefine a target evaluation quality and measure

$$C_{\mathrm{total}}=C_{\mathrm{input\ processing}}+C_{\mathrm{repair}}+
C_{\mathrm{probe}}+C_{\mathrm{policy\ training}}+C_{\mathrm{evaluation}}.$$

Keep GPU-hours, CPU-hours, simulator transitions and wall-clock latency separately. If processes overlap, wall time is the critical path, not the sum of resource hours. For a monetary or energy comparison, state conversion assumptions instead of silently equating CPU and GPU time.

Report:

1. Official task metrics at fixed total budgets.
2. Environment transitions and total time to a prespecified quality threshold.
3. Fraction of runs that reach the threshold within budget.
4. Total resource cost divided by the number of successful policies over a fixed task pool, including failed corrections and failed training runs.
5. Full learning curves and preprocessing overhead.
6. Repair failure rate, task fidelity, and controller-driven execution metrics.

The official tracking AUC is distinct from area under a *training learning curve*. Label the latter separately. Require a stable evaluation criterion rather than selecting a lucky early checkpoint. Runs that never reach the threshold are censored/failures; do not average only successful runs and present the result as overall speedup. Pair seeds and corruption instances, report task-level variation, and use intervals that respect grouping by task.

If your method needs a trained diagnostic, count both its development cost and an explicitly labeled amortized per-task cost. Show the break-even number of downstream tasks.

### 6.5 Demonstrate the mechanism

The most informative figure would show, for the same corrupted interaction:

**contact error → demanded-wrench/temporal defect → corrected execution → learning curve.**

Across tasks, test whether the diagnostic predicts which intervention helps after controlling for geometric error and task length. Then intervene: repair only the nominated phase, repair an equally long random phase, or remove the nominated constraint. This is stronger causal evidence than a correlation between final feasibility and final success.

Measure actual load-bearing contact persistence, mode transitions, object tracking, controller saturation, penetration, and force/torque limits. Do not substitute a fingertip-distance plot or kinematic playback for execution.

## 7 A credible IROS positioning

A working title could be **Learning Efficient Manipulation from Noisy Human Motion through Contact Feasibility Repair**. The final title should reflect the mechanism that survives ablation, rather than promising a general solution in advance.

A potential contribution statement is:

> We study how errors in reconstructed human-object interaction affect the compute required to learn robot manipulation. We introduce a compact correction that couples uncertain object motion and contact modes with the contacts attained by the robot. Under a fixed downstream learner, we evaluate whether the resulting diagnostics predict repair benefit and whether correction lowers total cost at matched task quality.

Convert the last sentence to a quantitative result only after the experiment. If uncertainty is not implemented, remove it from the statement. If the diagnostic does not generalize, remove it rather than retaining an unsupported third contribution.

For a strong robotics submission, I would aim for one clear algorithmic idea, convincing simulation evidence across contact regimes, and a modest real-robot test of the claimed contact mechanism if hardware is available. Hardware is not a substitute for a compute study, and its absence is not automatically disqualifying; it changes how broadly you can claim practical transfer. This is research judgment, not an official IROS acceptance rule.

Likely reviewer objections should drive the design:

| Objection | Evidence that addresses it |
|---|---|
| This is just IK with friction constraints | Matched geometry-only baseline and causal force/contact ablations |
| PhysicsGen or SPIDER already does this | Exact algorithmic distinction and matched-budget comparison |
| Better references obviously help RL | Prediction of when they help, when they hurt, and why |
| Optimization cost was hidden | Complete resource accounting, including failures and calibration |
| You corrected toward privileged ground truth | Explicit data provenance and tier separation |
| It works only on a hand-picked carry phase | Automated phase processing and full-episode results |
| The references are feasible only in your approximation | Controller-driven dynamic rollout and model-mismatch tests |
| It is a benchmark-specific improvement | Held-out contact regimes, objects, or a second compatible learner |

Avoid claiming that Newton–Euler equations, friction cones, object-local contact anchors, residual RL, contact-aware retargeting or uncertainty randomization are new. Avoid claiming that existing methods do not scale: several reviewed systems already operate over large collections. Your contribution can be narrower and still useful if the evidence is strong.

## 8 What to do next

1. **Establish one working policy-training and evaluation path.** Freeze the actual learner and reference interface. Without it, the central compute hypothesis cannot be tested.
2. **Run the strongest simple counterexample to your own idea:** matched geometry-only repair versus the current full correction. Include the book and at least one case where nominal target wrench feasibility genuinely fails.
3. **Close the attained-contact loop.** Add temporal mode checks and short dynamic rollouts, using simulator-matched parameters. Expand from carry to support transitions.
4. **Measure the pilot before adding a learned module.** Determine whether savings survive preprocessing cost and whether geometry already explains them.
5. **Choose the paper based on the result.** If mechanics helps specifically under noisy contacts, develop joint uncertainty/mode repair. If a cheap diagnostic predicts intervention benefit, develop compute allocation. If the effect disappears against strong baselines, redirect the claim rather than adding unrelated complexity.

The practical opportunity is a careful account of **which interaction errors make robot learning expensive, and how little correction is needed to remove them**. The literature makes that question demanding, but the repository's explicit mechanics and independent attained-contact audit are useful starting points.
