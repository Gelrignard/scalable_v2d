# Demonstration-preserving contact correction before policy learning

**Method note and implementation audit — 12 October 2026**

**Latest manipulation-only experiment, 19 October:** see
[Supported EEF manipulation](SUPPORTED_MANIPULATION.md). A freely simulated
book is carried by six force-limited contact pads, isolating the contact
correction from locomotion and arm IK. Frames 520–565 track within 1.66 mm and
2.07 degrees after correction, while the uncorrected contact replay loses
orientation. This is reduced-model physical evidence, not full G1 validation.
The earlier derivation and results below retain their original scope.

**Implementation update, 19 October:** the [bounded correction experiment](BOUNDED_CORRECTION.md)
adds sparse force checks, contact alternatives, temporal constraints, full-scene
collision diagnostics and CPU physical replay. It processed frames 400–600
within a 120-second correction budget, but failed physical acceptance. The
derivation and short-phase results below describe the earlier implementation.

**Operational update, 17 October:** see [COMPUTE_AND_IMPLEMENTATION_HANDOFF.md](../COMPUTE_AND_IMPLEMENTATION_HANDOFF.md) for the quoted 4090 server budget, reconstruction requirements, current submission-kit contract, and work packages for the next implementation agent. The method results below remain the 12 October trial.

**Research question.** Can a small, CPU-based mechanics optimization repair the contact inconsistencies in a retargeted human demonstration sufficiently to reduce the total compute needed to learn a successful robot policy?

The proposed contribution is a bridge between retargeting and policy learning. We preserve the demonstrated object motion and the human's approximate contact choices, infer the forces that would be needed, and adjust the robot reference toward contact configurations that can support those forces. Policy learning would then start from a more useful reference.

**Current status:** an object-level force/contact optimizer and a local robot trajectory correction are implemented. One short phase of a real Tier 1 episode has been processed on CPU. This demonstrates improved contact geometry and instantaneous object-wrench admissibility under stated assumptions. It does **not** yet demonstrate physically successful manipulation, faster policy learning, or scalability across the dataset.

For an advisor discussion, Sections 1–3 give the motivation and inputs; Sections 4–8 derive the implemented method; Sections 9–13 distinguish the evidence, remaining work, and proposed experiment.

## 1. What we are trying to preserve

Your starting point is the right one:

> We already have a demonstration of how the object should move, together with approximately correct human motion. Our job is to make that demonstrated interaction executable by the robot.

We treat the object trajectory as the task reference. We treat human motion as evidence about contact timing, participating fingers, contact regions, and overall body configuration. We allow local changes to the retargeted robot motion because human and robot hands have different geometry.

In the current experiment:

- The object poses and timestamps in the replay are unchanged.
- The original retargeted trajectory supplies the robot reference.
- Only arm and hand joints near one selected carry phase are corrected.
- The floating base, legs, and waist stay at their original reference values.

Keeping those coordinates fixed is a scope decision; it does not establish whole-body balance.

The policy eventually has to move a dynamic object through real simulated contacts. A visualization that prescribes both robot and object poses is useful for inspection, but does not establish that the robot can cause the motion.

### The claim we should make

“No existing method scales” is too broad to defend from our experiment. The useful hypothesis is:

> Repairing contact inconsistency before learning can reduce the cost of producing successful policies, even after accounting for the cost and failure rate of the repair.

This connects directly to a mechanics group's expertise: inverse dynamics, constrained optimization, contact modeling, and the separation of geometric feasibility from dynamic execution. The standard mechanics equations are not themselves a novelty claim. The research contribution would be the representation, correction procedure, integration with learning, and measured reduction in total cost.

## 2. What information is actually needed?

Let frame index be \(k=1,\ldots,N\), and contact index be \(i=1,\ldots,K\).

| Information | Purpose | Status in our trial |
|---|---|---|
| Object pose \(T_k=(R_k,p_k)\), timestamps, visibility | Motion reference; velocity and acceleration estimation | Provided Tier 1 data |
| Object surface geometry and metric scale | Contact locations, normals, collision checks | Provided mesh; planar approximation for optimization |
| Human body and hand motion in the same world frame | Contact hypotheses and retargeting | Decoded from SOMA-X |
| Human shape, scale, and flexible-bone parameters | Turn joint rotations into metric joint/fingertip positions | Used by the official body-model decoder |
| Retargeted robot configuration \(q_k^0\) | Starting robot trajectory and morphology prior | Existing contact-aware retargeter |
| Robot model, joint limits, velocity limits | FK, Jacobians, and local robot correction | G1 + Dex3 model |
| Object mass \(m\), COM \(c\), inertia \(I\) | Convert motion into force and torque demand | Provisional assumptions, not matched to evaluation physics |
| Friction and contact force bounds | Decide whether candidate forces are admissible | Provisional \(\mu=0.6\), 15 N normal cap per contact |
| Ground/support geometry and gravity | Distinguish hand support from environmental support | Supplied scene data, transformed consistently |
| Contact phase and candidate finger-to-surface associations | Restrict the search to the demonstrated interaction | Inferred associations; phase selected manually |

Thus, **object SE(3) and human joint angles alone are insufficient for a dynamics calculation**. Geometry and physical parameters are also needed. These can come from assets and simulator configuration; we do not need to estimate everything from video.

The supplied pose schema uses SOMA-X, rather than “SAMO.” Object pose is stored as position followed by a scalar-first quaternion. Human local rotations require the additional body parameters to recover metric fingertips. Visibility flags matter: a missing, zero-filled object quaternion is not a valid orientation. The released pose columns do not supply measured contact forces or ground-truth fingertip contacts. See the [Tier 1 data schema](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/track_2/tier_1_multiview_caption/README.md).

We convert human joints, object poses, and scene assets to a consistent metric, Z-up frame. A correct equation evaluated with mixed coordinate conventions is still wrong.

### Which tiers already provide reconstructed motion?

As checked on 12 October 2026, the [specific Track 2 release README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/5f68335f3acc802033d1e80728c1633197521de8/track_2/README.md) describes released Tier 1 and Tier 2 poses. For the reconstruction-based condition, it states:

> Reconstruction results are not published; participants generate them with their own method before retargeting and grounding.

Consequently, we can develop and test the correction directly on Tier 1 and Tier 2. For Tier 3, we must currently budget for generating and integrating reconstructions. The [challenge page](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) offers CARI4D as an off-the-shelf option. Developing a new CV estimator is not necessary, but assuming that Tier 3 poses are already downloadable would be incorrect.

## 3. What survives imperfect human reconstruction?

Suppose the world position of human fingertip \(i\) is \(x_i^H(t)\). Its position in object coordinates is

$$
h_i(t)=R(t)^\top\bigl(x_i^H(t)-p(t)\bigr).
\tag{1}
$$

During a sticking contact, the true surface contact point should be approximately constant in these coordinates. World-space hand motion can be large while object-relative motion remains small.

This representation has a useful invariance. If both the human and object are transformed by the same rigid change of world frame \((Q,b)\), then

$$
R'=QR,\qquad p'=Qp+b,\qquad x_i^{H\prime}=Qx_i^H+b
\quad\Longrightarrow\quad h_i'=h_i.
\tag{2}
$$

It cancels a common rigid coordinate change. It does not cancel independent hand/object reconstruction errors, scale errors, occlusion, or bad timing.

The information we hope survives is therefore coarse and relational:

1. Which hand and fingers participate.
2. Which side or surface neighborhood of the object they approach.
3. Approximately when contact starts and ends.
4. Whether a finger stays near one object-local region while the object moves.
5. The demonstrated arm and body posture that places the robot near a useful solution.

We do not assume the reconstructed fingertip is already on the true surface. We use it to select a **bounded candidate patch**, then let geometry and force balance refine the position.

For the book, the six candidates correspond to left/right thumb, index, and middle fingertips. Existing proximity/hysteresis heuristics identify contact candidates. Those heuristics are not validated contact labels, and the current patch construction is specific to the book's broad faces.

This is the bridge from uncertain human data to mechanics: human motion restricts the contact hypotheses; mechanics evaluates and adjusts those hypotheses.

## 4. Object motion determines a required net wrench

A wrench is a force and a torque considered together:

$$
w=\begin{bmatrix}F\\ \tau\end{bmatrix}\in\mathbb R^6.
$$

Torque must always be stated about a particular point. We use the object's center of mass, with all six components expressed in object coordinates.

### 4.1 Derivation, including an offset center of mass

Let:

- \(R(t)\) map object coordinates into world coordinates.
- \(p(t)\) be the world position of the object's mesh origin.
- \(c\) be its COM in object coordinates.
- \(I\) be its constant inertia matrix about the COM, in object coordinates.
- \(g\) be the world gravity vector.
- \(\omega\) be body-frame angular velocity, defined by \(\dot R=R[\omega]_\times\).
- \(\alpha=\dot\omega\), where \([a]_\times b=a\times b\).

The COM position is \(p_C=p+Rc\). Differentiating twice gives

$$
a_C=\ddot p+
R\left(\alpha\times c+\omega\times(\omega\times c)\right).
\tag{3}
$$

Newton's law, with gravity separated from contact forces, is

$$
m a_C=mg+\sum_i Rf_i+F_{\mathrm{env}}^W.
\tag{4}
$$

Euler's equation about the COM is

$$
I\alpha+\omega\times(I\omega)
=\sum_i(s_i-c)\times f_i+\tau_{\mathrm{env}}.
\tag{5}
$$

Here \(f_i\) is the force the finger applies **to the object**, in object axes, and \(s_i\) is its object-local contact point.

The required nongravitational wrench is consequently

$$
w^\star(t)=
\begin{bmatrix}
mR^\top(a_C-g)\\
I\alpha+\omega\times(I\omega)
\end{bmatrix}.
\tag{6}
$$

If an environmental support wrench is known, the hands must supply

$$
w_{\mathrm{hand}}^\star=w^\star-w_{\mathrm{env}}.
\tag{7}
$$

Our selected carry phase is separated from the table and floor, so the model sets \(w_{\mathrm{env}}=0\). That assumption cannot be carried over to pickup, placement, leaning, or supported sliding. Unknown support forces must instead be optimization variables with their own contact constraints.

**What is determined and what is not?** Given motion and physical parameters, Equation (6) determines the net wrench. It does not uniquely determine its allocation among fingers. Geometry, unilateral contact, friction, force limits, and demonstration priors restrict that allocation.

Even a motionless object requires support against gravity. Kinematics alone cannot tell us whether that support can be supplied.

### 4.2 Why noisy poses require care

Acceleration involves second derivatives. Position noise of amplitude \(\epsilon\) can create apparent acceleration on the order of \(\epsilon/\Delta t^2\). At 30 Hz, millimeter-scale jitter can substantially change inferred force demand.

The implementation applies an 11-frame local cubic Savitzky–Golay smoother to translation and sign-continuous quaternion coefficients, normalizes the quaternions, and evaluates translation/rotation spline derivatives. It excludes derivative estimates near missing data and segment boundaries.

This is an approximate derivative estimator, not recovery of ground-truth motion. In particular, **our wrench demand comes from smoothed motion, while the comparison replay preserves the raw object poses**. A small wrench residual therefore does not certify exact dynamic tracking of every raw pose fluctuation.

For noisier tiers, a defensible extension is to optimize a latent smooth object trajectory within confidence bounds. That extension is not implemented here.

## 5. A compact interaction representation

For one phase with fixed contact identities, represent contact \(i\) by

$$
s_i(u_i)=a_i+U_i u_i,\qquad
U_i=[t_{i1}\ t_{i2}],\qquad u_i\in\mathbb R^2.
\tag{8}
$$

\(a_i\) is the center of a planar surface patch, \(t_{i1},t_{i2}\) are orthonormal tangent directions, and \(n_i\) is the inward normal. Define \(B_i=[n_i\ t_{i1}\ t_{i2}]\).

The geometry variables are the two coordinates \(u_i\), held constant during this phase. The corresponding force at sample \(k\) is

$$
\lambda_{ik}=
\begin{bmatrix}\lambda_{n,ik}\\\lambda_{t1,ik}\\\lambda_{t2,ik}\end{bmatrix},
\qquad f_{ik}=B_i\lambda_{ik}.
\tag{9}
$$

Positive normal force pushes into the object. We use

$$
0\le\lambda_{n,ik}\le\bar f_i,\qquad
|\lambda_{t1,ik}|+|\lambda_{t2,ik}|
\le\eta\mu_i\lambda_{n,ik}.
\tag{10}
$$

The second inequality is a conservative diamond inside the circular Coulomb cone. We use \(\eta=0.85\) to leave a nominal friction reserve. This is not a guarantee against uncertain friction.

For six contacts, there are **12 geometric variables**. There are also \(3KN\) force variables: 738 in the 41-frame trial. “Compact” refers to the contact geometry and rigid-body description, not to a claim that the entire optimization has only 12 unknowns.

The reduced object model is mass, COM, inertia, and local surface patches. The mesh remains useful for constructing patches and checking the resulting robot geometry. This reduction can make offline reasoning cheap; it does not eliminate the dynamic object from the simulator used to train and evaluate the policy.

### The connection to your support-polygon intuition

Define the grasp matrix

$$
G(u)=
\begin{bmatrix}
B_1&\cdots&B_K\\
[s_1(u_1)-c]_\times B_1&\cdots&
[s_K(u_K)-c]_\times B_K
\end{bmatrix}.
\tag{11}
$$

Then \(G(u)\lambda_k\) is the net hand wrench. The admissible wrench set is

$$
\mathcal W(u)=
\{G(u)\lambda:\lambda\text{ satisfies Equation (10)}\}.
\tag{12}
$$

The desired test is \(w^\star_{\mathrm{hand},k}\in\mathcal W(u)\). This is the useful analogue of a support-region test: the question is whether these contacts can supply the required force **and** torque. It lives in six-dimensional wrench space and includes friction and force caps.

We do not explicitly construct the whole set. A small constrained force solve asks whether the particular demonstrated wrench can be produced. That is narrower than requiring a grasp to resist every possible disturbance.

### A numerical example: why contact position matters

Consider an illustrative stationary 1 kg box with COM at the origin and two contacts

$$
s_1=(a,d,0),\qquad s_2=(-a,d,0).
$$

The fingers squeeze along opposite \(x\) directions. To support the weight symmetrically, each needs an upward tangential force of \(mg/2=4.905\) N.

With \(\mu=0.6\), each needs at least \(4.905/0.6=8.175\) N of normal force. With our reserve, this becomes \(4.905/(0.85\cdot0.6)\approx9.62\) N.

But both contacts are displaced by \(d\) in the \(y\) direction. Their upward forces create

$$
\tau_x=d(f_{1z}+f_{2z})=dmg.
\tag{13}
$$

For \(d=0.02\) m, this is \(0.1962\) N m, whereas a stationary object requires zero net torque. Increasing the opposing squeeze forces cannot remove that \(x\)-moment in this point-contact geometry. Moving the demonstrated contacts toward \(d=0\) can.

That is contact-wrench guidance: motion specifies the demand; force balance exposes an inconsistency; a bounded change in demonstrated contact position can remove it. This example explains the mechanism; it is not a measurement from the book episode.

## 6. The implemented contact optimization

We use a nested optimization. At fixed contact locations, force allocation is a convex quadratic program (QP). A small outer optimization adjusts the patch coordinates.

### 6.1 Preserve human evidence while accommodating robot morphology

Let \(u_i^H\) be the robust human-derived patch coordinate and \(u_i^R\) the projection of the retargeted robot's median fingertip position. The book trial uses the soft preference

$$
\bar u_i=0.25u_i^H+0.75u_i^R.
\tag{14}
$$

These weights are a heuristic, not calibrated confidence estimates. Hard bounds are constructed from \(u_i^H\) before this preference is substituted:

$$
u_i\in\text{patch bounds},\qquad
\|u_i-u_i^H\|_\infty\le0.03\ {\rm m}.
\tag{15}
$$

This is a 3 cm bound per tangential coordinate, not a 3 cm Euclidean-distance guarantee. We exclude a 10 mm edge margin on the broad-face patches and reject initial candidates requiring more than 6 mm of normal projection.

### 6.2 Normalize force and torque before adding their errors

Force error is measured in N and torque error in N m. Simply adding their squared numerical values would make the tradeoff depend on arbitrary units.

Choose force scale \(f_0=10\) N and length scale \(\ell_0=0.1\) m:

$$
S_w=\operatorname{diag}(f_0,f_0,f_0,
f_0\ell_0,f_0\ell_0,f_0\ell_0),\qquad
z_k=\lambda_k/f_0,\qquad
b_k=S_w^{-1}w^\star_{\mathrm{hand},k}.
\tag{16}
$$

The normalized grasp block is

$$
\widehat G_i(u_i)=
\begin{bmatrix}
B_i\\
[(s_i(u_i)-c)/\ell_0]_\times B_i
\end{bmatrix}.
\tag{17}
$$

Stack forces and demands over time into \(z,b\). Let \(A(u)\) contain one copy of \(\widehat G(u)\) per frame, and let \(D_t\) compute force differences divided by the actual time interval.

### 6.3 Inner force QP

For a fixed \(u\), solve

$$
\Phi(u)=\min_z\
\frac{\alpha}{N}\|A(u)z-b\|^2
+\frac{\beta}{NK}\|z\|^2
+\frac{\gamma}{(N-1)K}\|D_tz\|^2
\quad\text{subject to Equation (10), in normalized variables}.
\tag{18}
$$

The three terms penalize wrench mismatch, force magnitude, and force variation. The book trial uses \(\alpha=10^4,\ \beta=0.05,\ \gamma=10^{-4}\). The last weight implicitly carries the time units associated with the derivative penalty.

Writing the objective as \(\tfrac12z^\top Hz+h^\top z+\mathrm{const}\),

$$
H=2\left[
\frac{\alpha}{N}A^\top A+
\frac{\beta}{NK}I+
\frac{\gamma}{(N-1)K}D_t^\top D_t
\right],\qquad
h=-2\frac{\alpha}{N}A^\top b.
\tag{19}
$$

Since \(\beta>0\), \(H\) is positive definite. The fixed-geometry force optimum is unique. We solve it with DAQP through qpsolvers.

Wrench balance is a **soft penalty**, not a hard equality. A solver can successfully return a large mismatch if the demonstrated wrench is impossible. We therefore inspect force and torque residuals afterward, in physical units.

### 6.4 Outer contact correction

The outer problem is

$$
\min_{u\text{ satisfying (15)}}\
J(u)=\Phi(u)+
\frac1K\sum_i\left\|\frac{u_i-\bar u_i}{\sigma_u}\right\|^2,
\qquad \sigma_u=0.02\ {\rm m}.
\tag{20}
$$

We use SciPy L-BFGS-B with an analytic gradient. Changing a contact point changes its torque contribution:

$$
\frac{\partial}{\partial u_{ij}}
\left[(s_i-c)\times f_{ik}\right]
=t_{ij}\times f_{ik}
\quad\text{when holding the inner optimum fixed}.
\tag{21}
$$

Because the force constraints do not depend on \(u\) for these fixed planar patches, the envelope theorem supplies the outer derivative without differentiating an entire learning procedure. In normalized variables, if \(r_k=\widehat Gz_k-b_k\),

$$
\frac{\partial\Phi}{\partial u_{ij}}
=\frac{2\alpha}{N\ell_0}
\sum_k r_{\tau,k}^\top
\left(t_{ij}\times B_i z_{ik}^{\star}\right).
\tag{22}
$$

The proximity penalty contributes \(2(u_{ij}-\bar u_{ij})/(K\sigma_u^2)\).

The complete problem remains nonconvex because contact positions and forces interact through torque. This is a local correction around the demonstrated interaction, with no global-optimality guarantee.

Finally, optimized points are projected onto the actual mesh. That changes positions and potentially normals. A force solution for the planar proxy must not automatically be treated as valid for the projected geometry.

## 7. How the contact plan changes the robot trajectory

The contact optimizer produces desired object-local contact points and nominal forces. It does not yet know whether the robot can attain them.

Let \(x_i(q)\) be robot fingertip FK. A target fingertip center in world coordinates is

$$
d_{ik}=p_k+R_k\left(s_i^{\rm mesh}+\rho n_{i,\mathrm{out}}^{\rm mesh}\right),
\qquad \rho=0.002\ {\rm m}.
\tag{23}
$$

The outward offset approximates a 2 mm fingertip radius. We apply sequential differential IK with Pinocchio/Pink, correcting 28 arm/hand joints.

To express the objective, let \(\delta q_{a,k}=q_{a,k}-q_{a,k}^0\) denote the actuated-joint correction. The residual structure is approximately

$$
\begin{aligned}
\min_{q_{a,k}}\quad&
\sum_i\|W_{ik}(x_i(q_k)-d_{ik})\|^2\\
&+0.025^2\|\delta q_{a,k}\|^2
+\|\delta q_{a,k}-\delta q_{a,k-1}\|^2\\
&+400^2\sum_j
\left[\max(\rho_j-d_{\rm hull,j}(q_k),0)\right]^2 .
\end{aligned}
\tag{24}
$$

The actual solver operates on linearized task residuals, with damping, constraints, repeated updates, and nonlinear backtracking. Equation (24) describes those residuals; it does not claim a jointly optimized global trajectory minimum.

The last term discourages hand-probe penetration into an object convex hull. It is a collision proxy, not complete self-collision or mesh collision checking.

Let \(N_{ik}\) be the world surface normal. We weight normal closure more strongly than tangential placement:

$$
W_{ik}=a_i\left[12I+(180-12)N_{ik}N_{ik}^{\top}\right],
\tag{25}
$$

where \(a_i\) is the normalized square root of the mean planned force magnitude, with a small floor. Thus, the force plan affects both target placement and the relative importance of contacts. **We do not command these inferred forces or convert them to joint torques in the current implementation.**

Hard constraints enforce joint limits, saved-frame velocity limits with a 0.9 margin, and bounds around the original joints. The latter allow up to 0.9 rad for arm joints and 1.5 rad for finger joints, tapered through 15 transition frames at each end. A backward reachability calculation helps ensure the correction can return to the original reference under the velocity limits.

The selected contact phase is frames 135–175; transitions extend the modified interval to 120–190. The largest actual joint change is 1.14 rad. Small Cartesian contact adjustments can require substantial finger joint changes.

**This is a time-coupled kinematic trajectory repair guided by object dynamics. It is not yet a full robot optimal-control solve.**

## 8. Audit the achieved contacts, not just the desired ones

After correction, recompute the actual robot fingertips in object coordinates:

$$
s_{ik}^{\rm act}=R_k^\top(x_i(q_k)-p_k).
\tag{26}
$$

Find the closest actual mesh point and normal. Define the signed gap relative to the fingertip-center offset:

$$
g_{ik}=
(s_{ik}^{\rm act}-s_{ik}^{\rm nearest})^\top n_{ik,\mathrm{out}}
-\rho.
\tag{27}
$$

A contact receives force authority only if \(|g_{ik}|\le3\) mm. Otherwise its force cap is set to zero. For each frame, solve another force allocation problem using the attained mesh locations and normals.

Acceptance of this **instantaneous object check** requires force residual at most 0.02 N and torque residual at most 0.002 N m. We separately check whether maximum sampled hand-probe penetration is at most 5 mm.

These thresholds are model tolerances, not measured physical accuracy. The post-repair audit solves forces independently at each frame. It does not enforce time-consistent contact forces, sticking, sliding-friction direction, actuator feasibility, or whole-body balance.

For example, a finger can remain near the surface while moving tangentially across it. The friction cone might contain a balancing force that a real sliding contact cannot generate because sliding friction must oppose relative motion. That is why instantaneous wrench existence is only one necessary test.

## 9. What the real Tier 1 trial actually shows

The input is episode 11, light-blue-book pickup and placement, from dataset revision 5f68335f3acc802033d1e80728c1633197521de8. There are 901 source frames; the final invisible frame is excluded. Retargeting covers all 900 visible frames. Correction covers one manually selected 41-frame airborne phase, approximately 4.50–5.83 s.

We assumed a 1 kg object, COM at the mesh bounding-box center, and a uniform-box inertia. Neither those inertial parameters nor the 15 N per-contact caps are certified properties of the evaluation robot/object. Our friction coefficient is also provisional.

| Metric, selected phase only | Original retargeting | Corrected |
|---|---:|---:|
| Mean absolute fingertip surface gap | 10.76 mm | 0.40 mm |
| Maximum absolute surface gap | 28.10 mm | 3.23 mm |
| Contact-eligible fingertip/frame pairs | 82 / 246 | 244 / 246 |
| Frames admitting the required instantaneous wrench | 0 / 41 | 41 / 41 |
| Maximum force residual | 10.50 N | \(2.89\times10^{-6}\) N |
| Maximum torque residual | 0.0782 N m | \(4.15\times10^{-7}\) N m |
| Maximum sampled hand-probe penetration | 26.84 mm | 0.99 mm |
| Mean 3D fingertip-to-planned-anchor error | 19.93 mm | 18.12 mm |
| Mean object-relative fingertip speed | 13.5 mm/s | **45.2 mm/s** |

No joint-position or saved-frame actuated-velocity violations were detected. Object poses, timing, and root pose are unchanged.

The important mixed result is that surface closure improved greatly, while the mean object-relative fingertip speed increased. The latter averages all six tips over the phase transitions; it is not a measured, force-weighted sliding speed, and some original tips were not in contact. Nevertheless, the corrected reference does not satisfy the intended sticking model.

The recorded output explicitly sets **accepted_as_feasible_manipulation = false**. The rest of the episode, including pickup and placement transitions, is not repaired by this experiment.

Measured CPU times were approximately 36 s for body decoding, 117 s for full retargeting, 6.25 s for phase contact optimization, and 12.16 s for the local repair routine through projection. The latter excludes subsequent final geometry evaluation and the separate force audit. These are individual-stage measurements, not a complete end-to-end benchmark or development-cost estimate.

The underlying records are [wrench_plan.json](../../correction/outputs/tier1_book/wrench_plan.json), [robot_repair_metrics.json](../../correction/outputs/tier1_book/robot_repair_metrics.json), and [fk_force_audit.json](../../correction/outputs/tier1_book/fk_force_audit.json).

**What we can claim now:** CPU preprocessing can substantially improve the attained contact geometry of this reference.

**What remains unproven:** that dynamics guidance is better than equally careful geometry-only repair, that the reference is physically executable, that it improves policy learning, and that the procedure works automatically across episodes.

## 10. What remains before calling the core method complete?

The architecture is concrete, but several gaps are inside the correction method itself.

### 10.1 Couple contact geometry to the attained robot motion over time

For an ideal sticking point contact, define

$$
r_i(q,t)=R^\top(x_i(q)-p)-s_i.
\tag{28}
$$

Ignoring the fingertip-radius offset for this idealized equation, sticking requires \(r_i=0\) and

$$
\dot r_i=
R^\top(J_i(q)v-\dot p)
-\omega\times R^\top(x_i(q)-p)=0.
\tag{29}
$$

The current robot projection prioritizes normal closure and allows tangential error. The next correction should enforce or strongly regulate object-relative tangential motion for contacts designated as sticking, while retaining force feasibility at the **attained** geometry.

A practical next implementation is a short-horizon sequential optimization that alternates robot trajectory updates with force allocation, followed by a joint contact/force audit. This reuses the existing reference and contact hypotheses. It need not search for a grasp from scratch.

For rolling fingertips, contact-point motion and fingertip-center motion differ; an improved pad model must represent that distinction. For intentional sliding, use a sliding contact model rather than forcing Equation (29) to zero.

### 10.2 Match the simulator and validate physical execution

Use the evaluator's object mass, inertia, collision geometry, friction, controller semantics, and timing. Replace generic force caps with actuator-aware constraints or explicit conservative estimates.

For context, a full robot dynamics condition would have the form

$$
M(q)\dot v+h(q,v)
=S^\top\tau+J_{\rm env}^\top f_{\rm env}^{W}
-\sum_iJ_i^\top Rf_i,
\tag{30}
$$

with joint-torque bounds and ground contact constraints. The minus sign appears because \(f_i\) was defined as the force on the object; its reaction acts on the robot.

Together with configuration evolution, object Newton–Euler equations, and contact constraints, this would define a genuine robot/object optimal-control problem. We currently do not solve it.

We need not build that entire solver before a small learning experiment. An existing whole-body controller and a short, controller-driven physical rollout can expose failures cheaply. Such a rollout must let the object respond to contact; prescribing its pose cannot validate manipulation.

### 10.3 Handle phases, supports, and uncertainty

The current manually chosen airborne phase avoids the most difficult transitions. Full episodes require contact activation/deactivation, table/floor support, release, and potentially sliding or impacts. A method that needs manual phase selection for every clip has not met our scaling objective.

For Tier 2/3, an optional latent trajectory could be written

$$
\widetilde T_k=T_k^{\rm obs}\exp(\widehat{\delta\xi_k}),
\qquad
J_{\rm obs}=\sum_k\delta\xi_k^\top\Sigma_k^{-1}\delta\xi_k,
\tag{31}
$$

with temporal regularization and contact consistency. Here \(\Sigma_k\) expresses pose uncertainty in a stated tangent-frame convention. This is a proposed extension, not part of the current trial. We must not use Tier 1 clean trajectories to repair Tier 2/3 inputs.

## 11. How this becomes a policy, and how to test the cost claim

A corrected trajectory is a reference, not a feedback policy. A downstream policy might take robot/object state and reference phase as input and produce controller-compatible actions:

$$
a_k=\pi_\theta(o_k,\phi_k;\ q^{\rm ref},T^{\rm ref}).
\tag{32}
$$

For a compatible controller, it could instead predict a residual around a reference action. The exact action interface must match the selected training backend. Joint trajectories do not automatically provide torque or residual-action labels.

The intended benefit is that policy learning spends less effort discovering contact closure and compensating for an internally inconsistent reference. This is a hypothesis. A more aggressively corrected reference could also be harder for the controller to track.

### Minimum informative comparison

Use the same simulator, controller, network, rewards, seeds, evaluation protocol, and training budget for:

| Variant | Reference preprocessing |
|---|---|
| A | Original retargeted motion |
| B | Geometry-only correction, with comparable collision, continuity, and IK settings |
| C | Dynamics-guided contact correction plus robot repair |

Variant B is essential. Much of the book improvement might come from stronger normal closure and collision handling. An A-versus-C comparison alone would not isolate the contribution of force reasoning.

Start with one episode to establish integration and estimate costs. If promising, use several episodes and paired random seeds; three episodes and three seeds would be a useful next small study, not proof of industrial scale. Report failures and full wall time, rather than only successful solver runs.

Measure:

- Official object-tracking performance at equal compute.
- Environment steps, wall time, and GPU-hours to a prespecified success level.
- Peak VRAM separately from training duration.
- CPU preprocessing, evaluation, reconstruction, and failed-run costs.
- Sensitivity to pose corruption and physical-parameter mismatch.

Let the total cost per attempted policy be

$$
C_{\rm total}=
C_{\rm reconstruction}+C_{\rm retarget}
+C_{\rm correction}+C_{\rm train}+C_{\rm eval}.
\tag{33}
$$

Costs should be reported in hardware hours and, where useful, a stated monetary conversion. Across a batch of demonstrations, the stronger scaling measure is

$$
\text{cost per successful policy}
=\frac{\text{total cost of all attempts}}
{\text{number meeting the evaluation criterion}}.
\tag{34}
$$

A useful correction must save enough training cost and/or rescue enough failures to compensate for its own cost. Faster optimization on one clip alone does not establish that.

### What must change to scale the optimizer?

The existing force QP uses dense matrices. With \(3KN\) force variables, matrix memory grows quadratically in phase length; current timings cannot be extrapolated to arbitrarily long videos.

Its structure is favorable for improvement: wrench constraints are per-frame, while force-rate penalties couple neighboring frames. Sparse/block solvers, cached geometry, adaptive force knots with dense validation, and parallel processing of independent phases are plausible engineering steps. Automatic phase/patch inference and useful failure rejection are equally important.

The compact representation helps make those steps possible; it is not evidence that they are already completed.

## 12. Compute requirements and the role of CHORD

| Stage | What we can currently say |
|---|---|
| SOMA-X decoding, retargeting, current correction | Already run on CPU locally |
| One/few-environment controller-driven playback | Must profile the chosen backend; not yet validated by our trial |
| Small learning integration experiment | Needs a working simulator/controller and a measured memory budget |
| Unmodified FlashCHORD full training recipe | Officially validated on a 48 GiB GPU with at least 46,000 MiB free |

The last specification comes from the [FlashCHORD README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/flash_chord/README.md). It also distinguishes lighter replay/viewing from full training. This is not a 48 GiB requirement for our optimizer or a universal minimum for every CHORD configuration.

The published [G1 FlashSAC configuration](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/flash_chord/src/flash_chord/configs/experiment/g1_recon_body_flash_sac.yaml) uses 2,048 simulated worlds and a replay capacity of ten million transitions. Such resident workloads help explain why memory can be large even when the policy network is modest. Reducing these settings may reduce memory, but changes throughput and potentially learning behavior; we have not established a working minimum for the laptop GPU.

**Fewer RL steps do not automatically mean less peak VRAM.** A policy that converges sooner can still use the same number of parallel worlds, replay capacity, and batches throughout training.

CHORD/FlashCHORD is a possible downstream training and comparison backend. We do not need a complete expensive reproduction before developing our correction. We do eventually need an uncorrected control run under the same training setup to support a speedup claim.

The cost-conscious order is: local correction, a short physical playback, a tiny training/evaluation integration check, then a bounded paired learning experiment. The present evidence justifies requesting that limited experiment, not claiming that a full competition submission already fits within a small cloud credit.

## 13. Challenge scope and an advisor-facing proposal

Track 2 evaluates learned-policy object tracking on G1 + Dex3. Tier 1 uses clean references, Tier 2 synthetic corruption, and Tier 3 reconstructed inputs. All are scored against clean object trajectories; each tier needs a separate submission and contributes equally. Thus, merely completing a similar semantic task does not guarantee a good score: object-motion fidelity matters. See the [official challenge description](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).

The [official FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf) requires learned-policy outputs for all 30 reference trajectories and whole-body control, while allowing pretrained controllers. Our 41-frame kinematic correction is an intermediate research result, not a submission.

**Submission-kit clarification checked 17 October:** the FAQ alone does not specify distinct neural networks, but the current [Track 2 kit guide](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip) directs participants to train per scored episode and supply one checkpoint per episode to its batch evaluator. We should implement and budget that workflow first. A shared, reference-conditioned policy would require additional design and verification of its compatibility with the submission interface; it is not the currently integrated path.

### A proposal that the current evidence supports

> I propose a bounded study of dynamics-guided reference correction for learning manipulation from human demonstrations. Human motion will supply contact hypotheses and a robot motion prior; object motion will supply force and torque requirements. We will use small constrained optimizations to repair contact inconsistencies before policy learning. A CPU prototype has improved contact geometry on a real challenge sequence, but execution and learning benefits remain to be tested. The next milestone is a controlled comparison against both original retargeting and geometry-only repair, measuring total compute per successful policy. This fits our group's work on dynamics and trajectory optimization while producing a clear go/no-go result for further GPU investment.

The first milestone is to repair the remaining sticking inconsistency and test the short sequence with the actual controller and object physics. The second is a matched learning comparison. Only if those are promising should we expand to automatic phase handling, additional episodes, and Tier 3 reconstruction.

## 14. Implementation map and reproducibility

| Component | File |
|---|---|
| Official SOMA-X decoding and coordinate conversion | [data_tools/decode_tier1.py](../../data_tools/decode_tier1.py) |
| Object inverse dynamics and smoothing | [object_wrench.py](../../correction/src/dynamics/object_wrench.py) |
| Rigid-body and contact-patch definitions | [models.py](../../correction/src/dynamics/models.py) |
| Inner force QP and outer contact optimization | [contact_optimizer.py](../../correction/src/optimization/contact_optimizer.py) |
| Real-book phase, patches, and assumptions | [book_wrench.py](../../correction/src/optimization/book_wrench.py) |
| Local robot trajectory correction | [robot_trajectory.py](../../correction/src/optimization/robot_trajectory.py) |
| Force reallocation at attained FK geometry | [contact_audit.py](../../correction/src/dynamics/contact_audit.py) |
| Corrected robot-reference output | [robot_reference.py](../../correction/src/pass_traj/robot_reference.py) |
| Commands and output description | [README.md](../../correction/README.md) |

The implementation uses NumPy/SciPy, DAQP/qpsolvers, and the existing Pinocchio/Pink robot tooling. Human decoding uses the official py-soma-x 0.2.1 package with CPU Torch in a separate environment. No neural-network training is part of the correction.

The sample dataset revision is 5f68335f3acc802033d1e80728c1633197521de8; SOMA assets are pinned to 466879a83d57eabf3d875ded2d869f2075f90348. The local upstream checkout inspected for this note is 193382a184308f6800331f3db829f76dc4294cfe. Challenge release and hardware statements were checked against the linked official sources on 12 October 2026; they can change after that date.
