# Contact sequences and command dynamics: implementation notes

This describes the bounded prototypes added during the 20 October overnight run. It does **not** replace the research plan or establish full-episode success. The experiment log and report contain the measured outcomes.

## What is now explicit

A contact phase stores:

- its demonstrated object region and the selected, actual robot/object collision-surface points;
- a complete finger configuration and object-relative palm poses;
- approach, closure, force ramp, carry, force fade, release and free-hold times;
- the same object mass/inertia, friction, actuator limits and controller convention used in replay.

`PhaseTiming` uses indices in the original episode. No phase owns a new simulation clock or initial state. `ContactPhase` and `build_sequence` compose robot commands, while a single `SupportedRobotScene.rollout` carries the physical state through every phase.

The current bin prototype has two interactions. Their timings are still declared experiment parameters, informed by the human contact intervals. They are not outputs of an automatic general contact-mode optimizer.

## Regrasping without resetting physics

One force/sensor channel corresponds to one physical robot body. If a finger uses a different surface point in a later phase, the reference stores time-dependent body-local points

$$
p_i^{B}(t),\qquad c_i^{O}(t),\qquad n_i^{O}(t),\qquad a_i(t)\in\{0,1\}.
$$

Here $p_i^{B}$ is on the robot collision surface, $c_i^{O}$ is the object anchor, $n_i^{O}$ is the inward object normal, and $a_i$ enables the phase. The controller rejects a changing robot patch if either adjacent sample is active. Interpolation between patches during an inactive interval is a channel transition, **not a model of rolling contact**.

The world-space point used for its Jacobian and physical force sensor is

$$
p_i^{W}=x_{B_i}^{W}+R_{B_i}^{W}p_i^{B}.
$$

The same point is used by the torque map and contact matching. Multiple force channels cannot silently double-count one physical body.

The new optional capture gate measures distance to the selected object patch:

$$
d_i=\left\|p_i^W-\left(x_O^W+R_O^Wc_i^O\right)\right\|.
$$

Robot preload is enabled when the channel is active and either a matching physical contact is sensed or $d_i$ is below the declared capture radius. This preload is robot torque. It is not an assertion that the requested force is already acting on the object. Measured contact forces remain separate.

The legacy gate measures distance to the object's bounding box. It can return zero inside a hollow bin where there is no nearby surface. Historical experiments retain that convention; newer patch-gate experiments explicitly record their capture radius. The first acquisition was sensitive to that radius: 2mm and15mm failed, while25mm restored a useful first manipulation. This is a tuned local controller result, not a general robustness claim.

## Connecting phases

After release, returning immediately to the unmodified arm motion caused another physical collision with the bin. The sequence builder instead retracts the palms, retains an explicit hold, and solves a continuous palm transition to the next approach. The transition uses an upward waypoint and joint-continuity regularization. Every intermediate pose is still subject to collision and reachability audits; the waypoint alone is not a collision certificate.

The second interaction revealed another issue: the first physical putdown differs from its intended pose. An optional offline iteration reads that deviation from the previous **unassisted full simulation**:

$$
\Delta p=\widehat p_{\mathrm{sim}}(t_a)-p_{\mathrm{obs}}(t_a).
$$

It initializes the next planned robot/object-relative path near that predicted state and smoothly reduces the translation offset to zero by the next phase end:

$$
p_{\mathrm{plan}}(t)=p_{\mathrm{obs}}(t)+\beta(t)\Delta p.
$$

Only the planning target changes. The scoring trajectory stays $p_{\mathrm{obs}}$, and the simulator never writes this planned pose into the object. This is one offline simulation-informed refinement, not an implemented online recovery policy. Its source result hash and120mm trust radius are recorded.

## Why command continuity was insufficient

The earlier robot reference used a C1 PCHIP spline. Although position and velocity were continuous, acceleration could change sharply and demand excessive inertial feedforward. A nominal audit of `sequence_v1` found first-phase commanded joint inertia alone exceeding an actuator cap by3.61×. Applied torque was clipped, so the intended command was not exactly realized.

`command_smoothing.py` tests a more explicit reduced problem. Robot joints are represented by a cubic B-spline with coefficient vector $z$:

$$
q=Bz,\qquad \dot q=B_1z,\qquad \ddot q=B_2z.
$$

The objective is

$$
\min_z\ \|Bz-q_{\mathrm{ref}}\|_2^2+
\lambda_a\|B_2z\|_2^2.
$$

It includes joint/velocity limits, a local trust region, unchanged initial command state, and linearized preservation of active surface points:

$$
\left|J_{i,k}\left((Bz)_k-q_{\mathrm{ref},k}\right)\right|
\leq \epsilon_p.
$$

Using mass matrices and Jacobians frozen at the reference, the nominal actuator constraint is

$$
\left|
M_{aa,k}(B_2z)_k+
M_{ar,k}\ddot q_{r,k}+
h_{a,k}-\tau_{\mathrm{passive},k}+
\sum_i J_{i,k}^{T}f_{i,k}
\right|
\leq \rho\,\tau_{\max}.
$$

The subscripts $a,r$ denote actuated upper-body joints and the externally prescribed root. $h$ includes gravity/Coriolis at the nominal command state. The object observations are not differentiated. Contact forces are held fixed in this first test, and contact tracking feedback is not magically absorbed into the torque bound.

The first full sequence solve exhausted its15s budget with `primal infeasible inaccurate`; it was rejected before replay. Constraint isolation subsequently showed that **even position-only fitting** was infeasible within the0.06rad trust region: the release reference changes right shoulder pitch by approximately0.55rad over two frames near506–508. This did not establish that fixed forces left insufficient actuator reserve. Widening only inactive-motion trust to0.35rad gives approximate solutions, but15s/30s runs still exceed the numerical residual gate and are rejected. This is not proof that no feasible manipulation exists.

## Regrasp paths and physically acquired contact surfaces

A static endpoint check missed a10.8mm thumb/index intersection during interpolation between hand shapes. The robot could not execute it, leaving about0.45rad thumb tracking error. The new inactive-path repair opens and reshapes unused fingers, then adjusts all upper-body joint samples through bounded sequential convex steps:

$$
\min_{Q,s\geq0}\quad
\|Q-Q_0\|_W^2+0.005\|D_1Q\|^2+0.0001\|D_2Q\|^2+10^5\|s\|^2,
$$

$$
d_j(Q^{(r)})+\nabla d_j(Q^{(r)})^T(Q-Q^{(r)})+s_j\geq d_{\min}.
$$

Here $Q$ contains inactive upper-body joint samples, $D_1,D_2$ are time-scaled differences, and $d_j$ is signed distance between actual collision geometries. Joint and speed bounds are hard constraints; each step stays within0.15rad of the previous iterate. Previously encountered pairs remain in the distance calculation even after leaving the simulator's active-contact margin. Dropping them immediately caused oscillation. The initial inherited0.25mm overlap is recorded and allowed only during a0.3s exit; a150Hz nonlinear audit requires no penetration afterward. This is a geometric repair; robot dynamics still requires physical replay.

Another failure was a mismatch between planned and acquired surfaces. The proximal left index link touched the object before the intended distal link. Added static candidates explicitly consider that proximal link, while requiring2mm clearance from encountered unwanted pairs. The accepted left-proximal candidate has0.079mm static contact mismatch and0.260mm maximum across six transport samples. A candidate using both proximal index links fails the geometry check.

Actual force logging now uses the solver contact point, force and contact couple:

$$
w_{\mathrm{object}}=
\begin{bmatrix}
f\\ (p_{\mathrm{contact}}-c_{\mathrm{COM}})\times f+\tau_{\mathrm{contact}}
\end{bmatrix}.
$$

All quantities are world-frame and the moment is about the **actual object COM**. Selected-patch, all-robot and environment wrenches are recorded separately. A selected robot patch can touch a different object location from the planned anchor; the logged point mismatch makes that visible.

The full replay with the proximal candidate improves object position/rotation RMS to69.8mm/8.46degrees, versus520.0mm/66.0degrees for the original matched retargeting. It keeps the bin upright after the second manipulation, but release displaces it and final position error is about149mm. Torque requests and joint-limit deviations remain unacceptable. **This is progress, not an accepted manipulation or evidence of a policy-training speedup.**

## Initial conditions and execution diagnostics

### Subsequent boundary and numerical correction (20 October,02:38)

The command boundary must use the **physical** initial state:

$$
B(0)c=q_{\mathrm{init}},\qquad
\dot B(0)c=\dot q_{\mathrm{init}}.
$$

Preserving an old command derivative instead leaves an initial damping impulse even when the initial positions agree. The previous common physical velocity came from the raw retargeted spline, including a12.31rad/s finger velocity above its12rad/s limit. The fitter now rejects this inconsistent boundary explicitly. A separate benchmark starts all three robot cases at zero joint velocity, from the original pre-approach posture. It retains the original object initialization and entire episode clock. This change requires fresh baseline replay; comparisons with the old initial state are not matched.

For numerical conditioning, contact constraint rows and their bounds are multiplied by100, and the QP is initialized by a least-squares fit of the joint samples to the spline basis. Neither changes the feasible physical set or objective. Acceptance residuals are converted back to the original units. The resulting rest-start command solve takes5.16s and satisfies the frozen nominal torque bound at0.8953of the actuator cap. Actual execution still requests up to2.31times the cap because realized contacts and tracking feedback differ from that nominal model.

The new matched full-episode comparison gives445.0/325.3/48.5mm object position RMS for original/geometry/mechanics, and final method error42.7mm/3.78deg. These results remain physically rejected under the existing gates. The full independent-pad replay is also rejected (218.9mm corrected RMS); changing robot surface contact into independent Cartesian spheres is not a proven equivalent execution contract.

### Impulse and contact-surface semantics

Video-rate force samples can miss a contact impact. The physical replay now records

$$
I_k^{\mathrm{robot}}=\sum_{s\in[k,k+1)}\Delta t\,w_s^{\mathrm{robot}},
\qquad
I_k^{\mathrm{environment}}=\sum_{s\in[k,k+1)}\Delta t\,w_s^{\mathrm{environment}}.
$$

For the source model's zero local COM offset, the measured force impulses satisfy

$$
m(v_{k+1}-v_k)
=mg\Delta t_k+I_{k,F}^{\mathrm{robot}}+I_{k,F}^{\mathrm{environment}}.
$$

An instrumentation-only replay is exactly equal in robot/object trajectories to the previous run, and its maximum linear momentum residual is2.44e-14Ns. This audit found74N withdrawal impacts between otherwise zero video-rate force samples. It does not infer force from observed video acceleration.

The independent-pad adapter preserves actual robot link/patch identities, including inactive-only patch changes. A selected robot surface point $p$ with outward normal $n$ becomes a sphere-center target $p-rn$, with radius $r=6$mm. Object-frame planned anchors, robot surface points, pad centers and measured contact centroids are separate arrays. Planned force arrows originate at intended object anchors; actual arrows use measured contact centroids. Explicit object mass, COM and inertia are shared, but pad actuators and contact solver settings differ from G1. Failure of this relaxation cannot be hidden behind a good robot result.

### Actual tracking torque and missing opposing contacts (20 October,03:30)

The nominal spline bound is not the executed torque bound. For the direct torque adapter,

$$
\tau_{\mathrm{req}} =
K_p(q_{\mathrm{cmd}}-q_{\mathrm{act}})
+K_d(\dot q_{\mathrm{cmd}}-\dot q_{\mathrm{act}})
+\tau_{\mathrm{comp}}+\tau_{\mathrm{contact}}.
$$

We now retain the four signed terms at each joint's actual1ms peak request, instead of explaining that peak with a nearby30Hz sample. At10.687s, left wrist pitch requests

$$
9.630+3.550-0.766-0.877=11.537\ \mathrm{Nm},
$$

against a5Nm cap. The large positive terms are tracking feedback. The allocated force is not, by itself, the cause of this peak. An instrumentation replay reproduces all prior robot/object states and torques exactly.

A controller-only diagnostic filters the contact-eligibility gain:

$$
\gamma_{s+1}=\gamma_s+
\frac{\Delta t}{T+\Delta t}(e_s-\gamma_s),
\qquad f_{\mathrm{guide},s}=\gamma_{s+1}\bar f_s.
$$

Here $e_s$ indicates an active scheduled contact that is touching or within the declared capture distance; $T=0.03$seconds in the diagnostic. An inactive planned mode clears the gain. Following temporary contact loss, guidance decays on the **robot joints**; the object still receives only physical contact. This changes the execution contract and is saved as an explicit ablation. It reduces the largest guidance-torque jump6.47→0.70Nm but leaves full tracking essentially unchanged (48.5→48.3mm RMS), worsens rotation RMS, and still requests2.05times capacity. It is not promoted.

Every-step signed contact impulses reveal a more fundamental representation gap. During airborne frames300–440:

| Contact set | Mean vertical force on object | Mean moment about actual COM |
|---|---:|---|
| Selected patches |+15.63N|(-1.344,+1.744,+0.116)Nm|
| Other robot patches |−5.89N|(+1.344,−1.744,−0.116)Nm|
| All robot contacts |+9.74N|approximately zero|
| Environment |0N|zero|

The opposing contact is predominantly the right index distal link, which is not an active planned channel in the first grasp. This is measured attribution, not a causal experiment removing that finger. Nevertheless, the executed grasp clearly contains an internal force/moment balance absent from the selected-point plan. The planned15N normal cap limits the guidance/allocator, not every actual contact force generated by position tracking, impact, or unselected robot surfaces.

The independent pads cannot be assumed equivalent to the full hand. Their original contact-only acquisition also differed from the robot's25mm patch preload gate. An explicit sphere-surface proximity version makes this distinction testable but **worsens** pad RMS218.9→383.3mm and overturns the bin. The original pad trial is preserved. Fixing that acquisition asymmetry alone does not resolve the missing grasp representation.

The next correction should first complete the active contact set: use the extra right-index contact as a candidate within the same demonstration-supported hand region, or explicitly constrain it clear of the object. Then jointly reconsider commanded hand shape/compliance and force allocation. Adding force to the existing incomplete set, increasing force limits, or claiming the sparse point reference is already sufficient would not follow from these results. This is still an inexpensive local candidate/refinement hypothesis; it needs an uninterrupted replay and a matched cost comparison.

### Exact current acceptance semantics

The local strict gate requires maximum object error below10mm/5degrees, penetration below1mm, initial penetration below0.05mm, no object assistance, joint-limit excursion below10mrad, and requested torque no greater than the cap. The best replay fails object tracking and requested torque. Its9.05mrad joint excursion is below that particular limit. No absolute adjacent-frame continuity gate is implemented in `assessed`; a0.209rad sample step is not itself proof of a discontinuity. The maximum is left wrist roll619→620, during the free transition. Joint acceleration RMS3.99rad/s² improves over the matched original25.66, while maximum EEFstep60.49mm is slightly above the original60.10mm. Report these separately rather than inventing a pass/fail threshold.

### Contact completion and continuous clearance (20 October,04:34)

The measured opposing right-index contact supplies a **hypothesis**, not an oracle force target. At a saved executed pose, let $x_c$ be a geometric contact position. Transform it into object coordinates, $r_c=R_o^\top(x_c-p_o)$, and retain it only if it lies within the declared100mm human-hand evidence region. Fit robot and object triangle coordinates and nearby joint angles using the same pose/evidence objective and actuator-bounded static capacity test. No measured force is invented from this recomputed geometry. The observation at frame400 yields a patch about46mm from its nearest human anchor.

Convex decomposition introduces a representation detail: one physical patch can span neighboring object geoms. The optional patch-neighbor rule requires the same robot link, both closest surface points within30mm of the corresponding selected anchors, and inward-normal cosine at least0.5. Only those compatible pairs receive the selected-contact penetration allowance. This matches the controller's finite-patch interpretation; it does not make every contact on that link acceptable or establish the accuracy of a single point-wrench approximation.

The five-contact static candidate passes geometry/capacity, but removing the unacquired left thumb from that candidate leaves a2.573N gravity-force deficit. The static five-contact LP can also allocate zero force to the new index. Therefore neither result proves that the intended internal force distribution will occur under the servo controller. The full unassisted replay remains the decisive check.

Another gap appeared in the free transition: collision-free video frames do not imply collision-free interpolation. Both neighboring frames had almost2mm clearance while their Hermite path penetrated54.5micrometers. Merely discovering the pair between frames and constraining the endpoints did not fix it.

For adjacent command knots $q_i,q_{i+1}$ separated by $h$, the Hermite midpoint is

$$
q_m=\tfrac12(q_i+q_{i+1})+\tfrac h8(v_i-v_{i+1}).
$$

At each local SQP iteration, recompute $v_i$ from the current PCHIP command, then hold these velocities fixed while linearizing the midpoint distance $d(q_m)$. Add

$$
d(q_m)+\tfrac12J_d(q_m)(\Delta q_i+\Delta q_{i+1})
+\xi_m \ge d_{\mathrm{clear}},\qquad \xi_m\ge0.
$$

The existing objective penalizes clearance slack; joint, speed, endpoint and trust constraints remain in force. This is a local approximation because velocities change between iterations. The independently evaluated150Hz nonlinear path audit remains mandatory, with the previous acceptance thresholds. The repaired path passes that audit, as does its later C2 command. This check covers the inactive525–600 interval, not every contact and collision throughout the complete physical episode.

**Physical result at04:40:** this completed-contact candidate fails its full replay (286mm RMS versus the retained48.5mm). In airborne300–440, selected patches now account for9.718of9.741N upward force and the net moment, but their force allocation is wrong: planned left-thumb vertical force−5.21N is actually−0.025N; right-index planned−2.06N is actually−10.13N; right-middle planned+6.51N is actually+13.78N. The thumb appears in only4of141video contact samples. The net object wrench is mechanically reasonable while the individual actuator/contact load distribution is different and torque saturates. Contact-set completeness, force-capacity feasibility, and controller realization are three distinct checks.

The next M5 step must optimize executable command parameters through the actual controller, or include a verified local compliance model, instead of treating allocated forces as independently realized decisions. A compact direct simulation search is still a required comparator. The new full failure also exposes phase coupling: changing first-grasp execution changes the object state presented to the second approach. Any prediction-based transition update must use a preceding unassisted rollout, retain the original task targets and physical state, count the extra rollout cost, and be checked again over the complete episode. No favorable first-phase metric substitutes for that.

### Bounded controller-aware command proposal (20 October,05:23)

The next local experiment keeps the controller and allocated force sequence fixed and changes only a few smooth servo targets. Let $B\in\mathbb R^{n_q\times4}$ contain two finger-shape directions and two wrist-pitch directions. Each unit parameter changes fingers by at most0.08rad or a wrist by at most0.02rad. These directions use the seed posture, not measured forces. Both search methods use exactly this same parameterization:

$$
q_{\mathrm{cmd}}(t;\alpha)
=q_{\mathrm{seed}}(t)+s(t)B\alpha,\qquad \alpha\in[-1,1]^4.
$$

The quintic envelope $s(t)$ and its first two derivatives vanish at the original-frame boundaries192 and525; it is one over270–480. Thus initial state, pelvis path, second-phase commands and complete episode clock remain unchanged. The simulator executes the resulting Hermite command, including the corresponding velocity. Old global spline coefficients are removed because they no longer describe this composite command.

A small mechanics proposal averages the measured generalized force discrepancy over airborne300–440:

$$
\bar e_\tau =
\frac1{T}\sum_k\Delta t_k
\sum_i J_i(q_k)^\top
\left(f^{\mathrm{plan}}_{i,k}-f^{\mathrm{meas}}_{i,k}\right),
\qquad
\alpha_{\mathrm{mech}}
\approx\arg\min_{\alpha\in[-1,1]^4}
\left\|D_\tau^{-1}(K_pB\alpha-\bar e_\tau)\right\|^2 .
$$

Here $D_\tau$ contains the unchanged actuator torque limits. The implementation computes an unconstrained least-squares solution and clips coordinates to the box; in this four-column construction the directions have disjoint joint support, so this clipping also solves the separable box problem. Measured forces come from integrated body impulses, divided by each video interval; no object acceleration is estimated. Jacobians use the saved actual state at the interval start.

This proposal assumes a fixed contact configuration and fixed other servo terms. It approximates a finite patch by force at its selected point, ignores independent patch couples, and averages transient loading. Its validity is explicitly tested, not assumed. A slider/box test verifies the sign: a positive target displacement away from a surface unloads an excessive contact force. It does not validate that approximation for the whole G1.

The mechanics arm tests scales0.25,0.5,1 of this proposal. The uninformed arm tests three predeclared random points in the same box, seed20261020. Both use the same bound backtracking and fixed controller/force sequence; candidates are interleaved under a720s budget and run full uninterrupted episodes. This is a local comparison from a mechanics-derived seed, **not** a comparison against starting from raw retargeting.

The saved ranking objective is

$$
J =
\operatorname{mean}_t\left[
\left(\frac{\|p_o(t)-p_o^\star(t)\|}{0.05\,\mathrm m}\right)^2+
\left(\frac{\theta_o(t)}{10^\circ}\right)^2
\right]
+0.1\,\operatorname{mean}_{t,j}\left[(|\tau^{\mathrm{req}}_j|/\tau_j^{\max}-1)_+^2\right]
+100\left(\frac{(\Delta q_{\mathrm{limit,max}}-0.01\,\mathrm{rad})_+}{0.01\,\mathrm{rad}}\right)^2 .
$$

Each term is reported separately. The torque term uses each joint's peak request within each video interval, before actuator clipping. A lower ranking score is not physical acceptance; object tracking, penetration, torque, initial state and joint-excursion gates remain unchanged. Changing the first grasp changes the second approach's actual initial object pose, so phase-specific and full-episode results must both be reported.

The first proposed search revealed an inherited spline defect before any physical candidate ran: its zero-offset seed exceeded a joint limit by1.396mrad between video frames. Bounding cubic B-spline coefficients and derivative coefficients now provides sufficient continuous position/speed constraints. The new seed passes the dense audit but still fails physical replay (259mm RMS), so it remains only an optimizer starting point. This is distinct from the retained48.5mm viewer fallback.

**Completed local result,05:32:** all six full trials fail. The best-ranked mechanics proposal has275.8mm/48.5deg RMS and2.42cap request; the best-ranked uninformed proposal has271.1mm/45.8deg and1.50cap. Their lower objective values come partly from smaller joint-limit penalties; neither improves tracking over the259.3mm seed. Another uninformed proposal reaches244.8mm but requests7.99cap and exceeds joint limits by32.9mrad. All three mechanics proposals worsen position tracking. The466.36s process is preserved in `servo_search_v2`, with a verified shared-input comparison.

This limited comparison does not establish a general method ranking. It does reject promotion of this frozen point-load proposal on this seed. In particular, one allocated force distribution is not a unique desired solution: the missing thumb and internal-force alternatives require updating contact availability and latent forces with executable commands and the long-term object objective. A force-error heuristic alone does not close that gap.

### Frozen-prefix second-approach prediction (20 October,06:22)

A preceding full unassisted simulation supplies the object pose at frame600,
$\hat T_b=(\hat R_b,\hat p_b)$. The original object observation remains
$T^\star(t)$ and remains the scoring target. The predicted pose is used only
to construct the next command; it is never written into the physical object.
The observed error is

$$
\delta p=\hat p_b-p_b^\star,\qquad
\delta\theta=\log(\hat R_b R_b^{\star\top}).
$$

The first trial blended these residuals into the previous planning path.
Its numerical solve and prefix checks pass, but the approaching thumb
intersects the bin; the trial is rejected before physics. A further audit
shows why hand targets and the approach obstacle must be distinct: the settled
object cannot be assumed to follow the hand-target blend.

The revised local hypothesis holds the planning object at $\hat T_b$ through
acquisition at645, then blends toward the transported task path by660.
Residual position and orientation decay smoothly to zero over660–725.
This is an explicit stationary-object approximation during approach, supported
only by the preceding simulation's2.56micrometer translation over600–610.
It is not a general state estimator or a dynamic feasibility certificate.

Hand targets use the previously screened object-relative palm configuration,
with an outside-and-above approach. A5mm final approach margin closes while
force ramps645–660. Original unused-finger shapes remain fixed. The old palm
command blends into these targets over600–612. All changes require nonlinear
collision checks against the approach obstacle and a new full episode.

To keep the prediction meaningful, the prefix is preserved exactly. Write the
cubic spline as $q(t)=\sum_i B_i(t)c_i$. Every coefficient whose basis support
starts before $t_b$ is fixed to its saved value. Remaining bases begin at or
after that simple interior knot; their value and first two derivatives vanish
there. Substitute fixed coefficients out of the QP:

$$
c=c_{\mathrm{fixed}}+S z,\qquad
A S z\in[\ell-Ac_{\mathrm{fixed}},u-Ac_{\mathrm{fixed}}].
$$

Only tail dynamics are relinearized. Continuous joint/speed bounds, contact
trust and the unchanged initial state remain explicit. This preserves the
whole earlier continuous command, not merely its sampled endpoint. The
simulator's q/dq arrays and force/contact prefix are also copied bitwise after
verifying the spline representation. The independent replay audit must verify
that the resulting *physical* robot and object prefix agrees exactly too.

This tests phase consistency. It does not solve the missing-thumb or internal
force allocation problem, and a numerical tail solve is not physical success.

The second attempt also fails the independent approach check after smoothing.
The bounded repair now places the actual signed-distance constraint in the same
spline decision variables. At dense sample time $t_j$, let $E_jc$ denote the
joint command and let $d_j,\nabla d_j$ come from the robot/object collision
surfaces at the current iterate $c_0$. Require

$$
d_j+\nabla d_j^\top E_j(c-c_0)\ge 0.002\,\mathrm m.
$$

These constraints have no slack and apply only to the declared inactive
approach600–644. They coexist with the exact prefix, continuous joint/speed
bounds and nominal robot torque constraints. Two local iterations discover
and retain newly encountered surface pairs. Independent150Hz geometry finds
zero penetration after the second iteration, against1.748mm before repair.
The complete repair/audit process takes17.27s. This demonstrates a bounded
geometric correction, pending physical replay; a local signed-distance model
still cannot certify all possible contact dynamics.

**Physical result,06:30:** the full replay has bitwise-identical robot q/dq and
object SE3 through600. The new second phase improves302.8→64.2mm position RMS
and57.55→13.01deg rotation RMS; full-episode position RMS improves259.3→153.6mm.
It still fails: final interval460.5mm/89.1deg, maximum error462.2mm/89.45deg,
requested torque2.639times capacity and penetration1.472mm. The unchanged first
phase retains its old torque failure. Actual torque remains clipped.

The impulse timeline prevents a misleading release diagnosis. At700 the object
already has21.15deg orientation error. Most useful hand contact is gone by700;
robot contact impulse is exactly zero from720 onward, whereas planned release
begins725. The bin tips later without another hand impact. Environment support
averages7.26N during680–700, versus2.46N from the robot. Updating the approach
helps, but contact persistence and transport are still unsolved. This candidate
does not replace the better48.5mm fallback, and no new pad/viewer result is
claimed.

### Contact persistence and bounded surface feedback (20 October,07:24)

The saved second-phase execution admits an exact geometric decomposition. At
the current robot patch and object anchor, write

$$
p_{\mathrm{actual}}-a_{\mathrm{actual}}
=(p_{\mathrm{actual}}-p_{\mathrm{command}})
+(p_{\mathrm{command}}-a_{\mathrm{plan}})
+(a_{\mathrm{plan}}-a_{\mathrm{actual}}).
$$

The three terms respectively measure robot tracking, nominal geometric fit,
and object prediction error. Over645–720 their mean magnitudes are about
3.3–4.1mm,0.7–1.0mm and35.8–47.1mm. These averages include the already-failed
motion. The timeline matters: gaps are3–4mm through675, but the left index
loses its last sampled contact at682, right index683 and right middle684.
By690 all four selected channels have lost sampled contact. This supports
testing adaptation to actual object motion before adding more force.

The bounded controller-only experiment keeps the optimized command and force
plan fixed. It uses current simulated object SE3 as a perfect bench sensor,
the current selected patch coordinates, and the temporary planning path.
For active patches within30mm, compute

$$
\delta p_i=a_i^{\mathrm{actual}}-a_i^{\mathrm{plan}},\qquad
\delta q_A\approx\arg\min_z
\|J_Az-\tfrac12 w(t)\operatorname{clip}_{15\mathrm{mm}}(\delta p)\|^2
+0.025^2\|z\|^2.
$$

The damped least-squares solution is clipped to0.08rad; it is not asserted to
be the exact box-constrained optimum. Only arm/waist columns $A$ are used.
Finger commands remain fixed. The quintic envelope rises660–675 and falls
725–746. Fifty-Hz target updates drive an integrated critically damped offset
state at1ms, with declared0.4rad/s and2rad/s² limits. The simulator consumes
both the modified position and its matching velocity. Robot torque is still
clipped at the actual actuator caps. No state or force is applied to the object.

This adapter can follow an undesirable object displacement; it does not add
the missing long-horizon task-wrench optimization. It is an explicitly changed
feedback controller, not an isolated improvement to the offline reference.
The first trial improves full-episode RMS153.6→57.1mm and finishes upright,
but fails task tracking and has seven offset-acceleration conflicts caused by
hard clipping at nominal joint limits. Maximum offset acceleration49.17rad/s²
violates the declared2rad/s², so that result is rejected.

The repair selects only joints with at least0.0801rad margin over the complete
remaining nominal trajectory. A cubic Hermite segment is bounded by the
convex hull of its four Bezier control points

$$
\left(q_k,\ q_k+\tfrac h3\dot q_k,\
q_{k+1}-\tfrac h3\dot q_{k+1},\ q_{k+1}\right).
$$

Using these bounds leaves room for the entire permitted offset without
crossing absolute joint limits. The full replay must still check all derivative
and physical bounds; this margin does not certify the physical robot state.

The completed `surface_follow_v2` replay uses 14 arm/waist joints and passes
the added offset contract: maximum offset 0.03304 rad, speed 0.2241 rad/s,
acceleration 2 rad/s² within numerical tolerance, and zero bound conflicts.
Its physical robot/object prefix through frame 660 is bitwise identical to
`predicted_replay_v1`, and every saved reference array is unchanged. Full
858-frame RMS error improves from 153.61 mm / 30.34 degrees to
57.72 mm / 7.66 degrees. This is a controller ablation with a perfect
simulated-pose sensor, not an isolated offline trajectory improvement.

Physical acceptance still fails. Final error is 142.60 mm / 9.04 degrees;
the unchanged first interaction requests 2.639 times actuator capacity.
The second interval requests up to 1.013 times capacity. Actual torques
remain clipped to the real caps. Dense impulses show that during frames
680–700 robot support averages 5.334 N vertically and environment support
4.140 N. Sampled contact lasts longer for several selected fingers, but the
object is not continuously carried by them. After planned release, other
finger surfaces still contact the object: during 770–857 the left middle
proximal link contributes 0.878 N·s unsigned impulse, with a 68.45 N peak.
These are physically simulated contacts, not the planned force allocation.
The linear momentum-balance residual is 2.44e-14 N·s.

The experiment therefore supports a limited role for bounded surface
feedback while exposing its limitation: following an observed object
deviation does not impose task progress or guarantee a clean withdrawal.
The best 48.48 mm fallback remains unchanged. New evidence:
[matched comparison](../../correction/outputs/overnight_20261020/surface_follow_comparison_v1/report.json),
[physical replay](../../correction/outputs/overnight_20261020/surface_follow_v2/report.json),
and [dense impulse audit](../../correction/outputs/overnight_20261020/surface_follow_v2/release_impulse_audit.json).

A separate read-only LP diagnostic reproduces the no-thumb first-grasp
deficit2.573N. Changing friction, compression limit or torque capacity alone
does not remove it. Friction1 plus24.86N compression reaches static balance
at the real actuator limit; at friction0.8, a relaxed case uses32.62N and
2.53times real actuator capacity. These are counterfactual diagnostics only,
not executable force references or permission to raise limits. They show
interacting constraints for this local grasp, not impossibility of all grasps.

## Files and entry points

| File | Responsibility |
|---|---|
| `src/optimization/manipulation_sequence.py` | Contact phases, continuous free transition and shared force-channel assembly |
| `src/optimization/surface_reference.py` | Per-phase geometric command construction |
| `src/optimization/command_smoothing.py` | Bounded C2 command/torque QP prototype |
| `scripts/screen_contact_transport.py` | Cheap sparse trajectory-wide grasp rejection before physical replay |
| `scripts/rollout_manipulation_sequence.py` | Build and physically test a two-phase command |
| `scripts/replay_frozen_sequence.py` | Controller ablations on an immutable saved command |
| `scripts/refine_sequence_commands.py` | Command-QP trial, rejection or uninterrupted replay |
| `scripts/audit_command_dynamics.py` | Separate nominal joint inertia, root inertia and gravity demand |
| `src/optimization/finger_transition.py` | Bounded inactive-path search and collision/smoothness refinement |
| `src/dynamics/contact_measurement.py` | Actual solver wrench at object COM |
| `scripts/diagnose_contact_delivery.py` | Requested versus measured wrench and tracking torque audit |
| `scripts/screen_executed_contact_geometry.py` | Static capacity of geometry at saved executed poses |
| `scripts/audit_release_impulses.py` | Every-step contact impulses and momentum-balance audit |
| `src/dynamics/torque_audit.py` | Per-joint signed decomposition at actual torque peaks |
| `scripts/audit_torque_peaks.py` | Explain sub-frame peaks and verify unchanged instrumentation replay |
| `scripts/audit_contact_realization.py` | All-body versus selected-patch impulses and pad acquisition comparison |
| `src/optimization/contact_completion.py` | Recover plausible additional robot/object mesh patches from saved execution |
| `scripts/complete_grasp_contacts.py` | Screen completed contact hypotheses within human evidence regions |
| `scripts/audit_frozen_command.py` | Recheck interpolated free-path clearance and shared baseline contract |
| `src/optimization/servo_preload.py` | Four-parameter smooth command space, load-error proposal and fixed ranking objective |
| `scripts/search_servo_preload.py` | Bounded full-episode local mechanics-versus-uninformed search comparison |
| `scripts/summarize_servo_search.py` | Verify shared inputs/initial states and separate objective ranking from physical acceptance |
| `src/optimization/phase_prediction.py` | Separate temporary SE3 planning hypotheses from fixed task observations |
| `scripts/update_second_approach.py` | Rebuild the second approach, freeze the entire first prefix and audit before export |
| `scripts/audit_prediction_replay.py` | Verify exact earlier physical execution and unchanged task/controller after replay |
| `src/optimization/spline_clearance.py` | Actual collision-surface constraints in the executed spline variables |
| `scripts/refine_predicted_approach.py` | Bounded local tail SQP and independent dense collision/limit gates |
| `scripts/diagnose_contact_persistence.py` | Separate actual contact separation into tracking, nominal geometry and object prediction errors |
| `src/dynamics/surface_follow.py` | Optional bounded arm feedback using current object-surface displacement |
| `scripts/replay_surface_follow.py` | Full-episode controller ablation with exact unchanged-prefix verification |
| `scripts/summarize_surface_follow.py` | Verify identical references/base contract and physical prefix before comparing feedback trials |
| `scripts/diagnose_capacity_constraints.py` | Read-only friction/compression/actuator constraint isolation; no executable export |
| `scripts/repair_release_path.py` | Bounded inactive withdrawal repair; rejects overlapping fixed endpoints |
| `src/preprocessing/surface_pad_reference.py` | Shared surface channels and radius-correct independent-pad targets |

Paths in this table are relative to `correction/`. All runners require a new output directory and preserve source snapshots. Saved references, simulated results, requested forces and physically measured forces must remain distinct.

The physical acceptance gate still fails. In particular, a good geometric contact fit and a useful first carry do not establish a feasible second manipulation, a lower total policy-learning cost, or an official challenge submission.
