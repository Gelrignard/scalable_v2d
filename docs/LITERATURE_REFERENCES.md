# Annotated literature on contact correction and policy learning

Companion to the [research review](LITERATURE_REVIEW.md). This is a targeted, broad review of **46 papers**, with sources checked through **17 October 2026**, rather than an exhaustive systematic review. Publication status is stated conservatively: a preprint or submission is not treated as an accepted paper. Years refer to the stated publication or initial preprint; revision dates are supplied where they materially affect the comparison.

**Evidence labels:** **M** means relevant method/results text was inspected in the paper or a detailed official method page. **S** means the annotation is limited to a primary abstract or official project summary; it does not support fine-grained claims about unreported ablations. The annotations identify relevance to this repository, not independent reproductions of the papers' results.

## Read these first

Start with **CHORD → SPIDER → PhysicsGen → C2Dex → Direct Dynamic Retargeting → Grasp-and-Lift**. Then read **Opt2Skill** for the optimization-to-RL interface and **Should We Learn Contact-Rich Manipulation Policies from Sampling-Based Planners?** for the distinction between executable data and useful learning data.

For each paper, record: what is observed; what is held fixed; what is optimized; how contacts are inferred; whether forces are targets, decision variables or simulator outcomes; whether the output is replay or a feedback policy; and exactly what the reported compute excludes. These questions reveal stronger distinctions than grouping every method as “physics-based.”

## Closest comparisons

**1. Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration — CHORD.** Zhu et al., 2026 preprint. **M.** [Paper, v2](https://arxiv.org/html/2607.00033v2), [official project](https://nvidia-isaac.github.io/video_to_data/chord/). Read the wrench formulation, noise ablation, learning setup and baseline-reproduction appendix. This is the essential downstream baseline and the reason to distinguish required wrench from contact capability.

**2. SPIDER: Scalable Physics-Informed Dexterous Retargeting.** Pan et al., 2025 preprint; v3 dated 26 September 2026. **M.** [Paper, v3](https://arxiv.org/html/2511.09484v3). The strongest broad comparison for efficient physical retargeting. Inspect its contact filtering, optimization restarts and robustness before proposing similar modules. Keep trajectory-generation cost separate from closed-loop policy cost.

**3. Physics-Driven Data Generation for Contact-Rich Manipulation via Trajectory Optimization — PhysicsGen.** Yang et al., RSS 2025. **M.** [Proceedings paper](https://www.roboticsproceedings.org/rss21/p053.pdf), [project](https://lujieyang.github.io/physicsgen/). A direct precedent for the overall architecture. Particularly useful for studying how retargeted demonstrations become dynamically consistent training data and how their diversity is generated.

**4. Direct Dynamic Retargeting for Humanoid Imitation Learning from Videos.** Roux et al., May 2026 preprint. **M.** [Paper](https://arxiv.org/html/2605.23762v1). Uses task-space simulation optimization and evaluates subsequent RL. Its agile/balancing examples differ from dexterous manipulation. Its convergence measure includes reaching a fraction of each method's own final reward; your experiment should instead use a common task-quality threshold.

**5. C2Dex: Contact-Consistent Reconstruction and Retargeting for Dexterous Manipulation from Monocular Video.** Ren et al., August 2026 preprint; September v2 lists a RA-L submission. **M.** [Paper](https://arxiv.org/html/2608.07045v2). Close prior for stable object-side contact reconstruction and transfer. Particularly relevant to your fixed-anchor representation. Compare its fixed object-reference assumption with any proposed joint object/contact repair.

**6. Grasp-and-Lift: Executable 3D Hand-Object Interaction Reconstruction via Physics-in-the-Loop Optimization.** Choi, Oh and Lim, January 2026 preprint. **M.** [Paper](https://arxiv.org/html/2601.18121v1). Compact spline-based motion parameters and CMA-ES permit black-box physical refinement while retaining demonstration fidelity. A useful simple competitor to elaborate differentiable or learned correctors; its grasp/lift scope should be distinguished from full manipulation sequences.

**7. CoRe: A Hybrid Approach of Contact-Aware Optimization and Learning for Humanoid Robot Motions.** Humanoids 2025. **M.** [Publisher](https://doi.org/10.1109/Humanoids65713.2025.11203055), [official method page](https://tmjeong1103.github.io/CoRe/). Contact-segment detection, motion refinement and subsequent contact-aware RL are already combined. Its foot-contact emphasis leaves different mechanics for dexterous objects, but rules out novelty based on the order of stages.

**8. OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction.** ICRA 2026 according to the official project; 2025 preprint. **M.** [Project](https://omniretarget.github.io/), [paper](https://arxiv.org/html/2509.26633v2). Uses an interaction representation and constrained optimization to retain agent–object–environment relations. Important for geometric baselines and scene-aware retargeting; distinguishes relational fidelity from force feasibility.

**9. ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning.** CVPR 2025. **M.** [Project and paper links](https://maniptrans.github.io/). A general hand imitator supplies a starting point for task-specific residual learning. Useful both as a competing transfer system and as an example of a downstream learner whose behavior could be held fixed while testing preprocessing.

**10. DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation.** 2025 preprint; ICML 2026 listing. **M.** [Paper](https://arxiv.org/html/2505.24853v1), [conference listing](https://icml.cc/Downloads/2026). Contact and motion guidance combine with a virtual object-control curriculum. Functionally appropriate robot contacts need not reproduce every human contact. Relevant to rejecting overly rigid demonstration-preservation constraints.

**11. DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References.** Liu et al., ICLR 2025. **M.** [Proceedings paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/d5a1f97d2b922da92e880d13b7d2bf02-Paper-Conference.pdf), [project](https://meowuu7.github.io/DexTrack/). Iteratively combines successful tracking data, RL/IL and homotopy-based per-trajectory optimization. A key precedent for amortizing correction across many interactions; account for the cost of creating its general controller when comparing deployment-time efficiency.

**12. Grasp to Act: Dexterous Grasping for Tool Use in Dynamic Settings.** Gupta, Mirzaee and Yuan, RA-L 2026. **M.** [Paper](https://arxiv.org/html/2602.20466v1), [publisher](https://ieeexplore.ieee.org/document/11455929/). Demonstration-informed grasp synthesis uses disturbance-based grasp evaluation, followed by residual adaptation. A direct warning that adding grasp-wrench robustness to a reference-following policy is already established. Time-varying uncertain contact modes would need additional justification.

## Human interaction and efficient control

**13. DexMV: Imitation Learning for Dexterous Manipulation from Human Videos.** Qin et al., 2021 preprint. **S.** [Primary record](https://arxiv.org/abs/2108.05877), [project](https://yzqin.github.io/dexmv/). Early video-to-dexterous-imitation pipeline. Read for historical positioning: reconstructing and retargeting human demonstrations for manipulation learning predates the current humanoid systems. This annotation uses the preprint date rather than inferring venue status from secondary indexes.

**14. D-Grasp: Physically Plausible Dynamic Grasp Synthesis for Hand-Object Interactions.** Christen et al., CVPR 2022. **M.** [Proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Christen_D-Grasp_Physically_Plausible_Dynamic_Grasp_Synthesis_for_Hand-Object_Interactions_CVPR_2022_paper.html). Hierarchical physics/RL synthesizes grasp-and-move interactions from a grasp reference. Its input is much less than a full dense trajectory, illustrating that exact frame-by-frame imitation is not always necessary for useful physical interaction.

**15. PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction.** 2023 technical report. **M.** [Official project](https://wyhuai.github.io/physhoi-page/). Contact-graph guidance helps physics-based imitation avoid failures of kinematics-only matching. Useful for explaining the importance of interaction structure, while recognizing that the physical character-control setting differs from real robot actuator and sensing constraints.

**16. InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions.** Xu et al., CVPR 2025. **M.** [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_InterMimic_Towards_Universal_Whole-Body_Control_for_Physics-Based_Human-Object_Interactions_CVPR_2025_paper.pdf), [project](https://sirui-xu.github.io/InterMimic/). Teacher policies refine and track imperfect interactions, followed by distillation and further learning. A precedent for using corrected physical behavior to train a scalable controller rather than treating every raw reference as equally suitable.

**17. QuasiSim: Parameterized Quasi-Physical Simulators for Dexterous Manipulations Transfer.** Liu et al., ECCV 2024. **M.** [Project](https://meowuu7.github.io/QuasiSim/), [proceedings paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10032.pdf). A curriculum of simulator models trades optimizability against fidelity. Useful when deciding how much contact physics belongs in your CPU optimizer, and why final execution must be checked in the target simulator.

**18. Visual Imitation Enables Contextual Humanoid Control — VideoMimic.** May 2025 preprint. **M.** [Primary record](https://arxiv.org/abs/2505.03729), [paper](https://www.videomimic.net/VideoMimic.pdf). Combines video reconstruction, environmental context, retargeting and learned humanoid control. Adjacent whole-body context rather than the most direct dexterous baseline; useful for the assumptions inherited from video reconstruction.

**19. DexSynRefine: Synthesizing and Refining Human-Object Interaction Motion for Physically Feasible Dexterous Robot Actions.** May 2026 preprint; revised June 2026. **M.** [Primary record](https://arxiv.org/abs/2605.05925), [paper](https://arxiv.org/html/2605.05925v1). Combines interaction motion synthesis, task-space residual refinement and dynamics adaptation. Relevant to action-space choices and a warning against presenting residual RL plus contact adaptation as a new combination.

**20. Scaffolding Dexterous Manipulation with Vision-Language Models.** de Bakker et al., NeurIPS 2025. **M.** [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/862644b156e51c35dea5a7446d640b14-Abstract-Conference.html), [paper](https://arxiv.org/html/2506.19212v2). Coarse generated motion scaffolds guide residual learning. Important conceptual counterpoint: the goal may be useful structure for learning rather than the most accurate possible reconstruction.

**21. DemoStart: Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots.** Bauza et al., 2024 preprint. **S.** [Primary record](https://arxiv.org/abs/2409.06613). Uses a handful of simulated demonstrations and sparse reward to automate a manipulation curriculum, with real transfer. Relevant to reset-state selection and sample-efficiency baselines. Its reported demonstration-efficiency result should not be recast as a training-compute speedup.

**22. DiffMimic: Efficient Motion Mimicking with Differentiable Physics.** Ren et al., ICLR 2023. **M.** [Paper](https://openreview.net/pdf?id=06mk-epSwZ), [official code](https://github.com/jiawei-ren/diffmimic). Differentiable simulation and demonstration replay improve motion-mimicking efficiency. An adjacent alternative to model-free RL, with different assumptions about gradients and simulation; character-animation results do not directly establish dexterous robot performance.

## Correcting reconstructed interaction

**23. ContactOpt: Optimizing Contact To Improve Grasps.** Grady et al., CVPR 2021. **M.** [Proceedings](https://openaccess.thecvf.com/content/CVPR2021/html/Grady_ContactOpt_Optimizing_Contact_To_Improve_Grasps_CVPR_2021_paper.html). Predicts desirable contact and optimizes the hand pose to realize it. Establishes contact-based pose correction. Its soft-tissue contact model can permit penetration; transferring that assumption to rigid robot fingers requires care.

**24. TOCH: Spatio-Temporal Object-to-Hand Correspondence for Motion Refinement.** Zhou et al., ECCV 2022. **M.** [Proceedings paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630001.pdf), [primary record](https://arxiv.org/abs/2205.07982). Object-centric temporal fields and a learned prior refine erroneous interaction sequences. A direct precedent for denoising in object coordinates; it does not itself establish lower robot-policy training cost.

**25. Learning Object Manipulation Skills from Video via Approximate Differentiable Physics.** Petrík, Qureshi, Sivic and Tapaswi, IROS 2022. **M.** [Author paper](https://www.cs.toronto.edu/~makarand/papers/IROS2022_PhysReal2Sim.pdf). Physics-regularized reconstruction produces simplified interaction trajectories that can transfer to a robot without the same reliance on RL. Valuable historical precedent for placing a mechanics stage between perception and robot execution.

## Contact planning and uncertainty

**26. Contact-Invariant Optimization for Hand Manipulation.** Mordatch, Popović and Todorov, SCA 2012. **M.** [Author paper](https://homes.cs.washington.edu/~todorov/papers/MordatchSCA12.pdf). Optimizes hand motion and auxiliary contact variables from object-level objectives. Establishes longstanding work on joint motion/contact selection. Particularly relevant to arguments that the human demonstration restricts an otherwise difficult contact search.

**27. A direct method for trajectory optimization of rigid bodies through contact.** Posa, Cantu and Tedrake, IJRR 2014, online 2013. **M.** [Publisher](https://doi.org/10.1177/0278364913506757), [author paper](https://groups.csail.mit.edu/robotics-center/public_papers/Posa13.pdf). Contact-implicit optimization uses complementarity to capture unilateral contact and friction. Read for the conditions omitted by instantaneous wrench feasibility, and for the computational consequences of including them.

**28. Global Planning for Contact-Rich Manipulation via Local Smoothing of Quasi-dynamic Contact Models.** Pang, Suh, Yang and Tedrake, 2022 preprint, revised 2023. **M.** [Primary record](https://arxiv.org/abs/2206.10787), [author paper](https://groups.csail.mit.edu/robotics-center/public_papers/Pang22.pdf). Combines smoothed contact models with global sampling-based planning. An alternative to using RL to discover manipulation transitions; also a foundation for understanding when quasi-dynamic approximations are computationally attractive.

**29. Simultaneous Trajectory Optimization and Contact Selection for Multi-Modal Manipulation Planning — STOCS.** Zhang, Jha, Raghunathan and Hauser, RSS 2023. **M.** [Proceedings paper](https://www.roboticsproceedings.org/rss19/p044.pdf). Jointly selects contacts and optimizes trajectories, including environmental support and different manipulation primitives. Important when extending the current carry-only model to sliding, pivoting and transitions.

**30. Mediating Between Contact Feasibility and Robustness of Trajectory Optimization Through Chance Complementarity Constraints.** Drnach, Zhang and Zhao, Frontiers in Robotics and AI, 2022. **M.** [Publisher](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.785925/full). Studies risk-sensitive objectives and uncertain contact feasibility. Its examples illustrate that adding a chance constraint is not equivalent to achieving robust behavior; the robustness objective and contact model matter.

**31. Chance-Constrained Optimization in Contact-Rich Systems for Robust Manipulation.** Shirai, Jha, Raghunathan and Romeres, 2022 preprint. **S.** [Primary record](https://arxiv.org/abs/2203.02616). Uses stochastic complementarity-system models and joint chance constraints. A theoretical precedent for probabilistic contact feasibility, with modeling assumptions that must be revisited before applying it to noisy high-dimensional human motion.

**32. Trajectory Optimization under Contact Timing Uncertainties.** Zhao and Khadiv, 2024 preprint. **S.** [Primary record](https://arxiv.org/abs/2407.11478). Addresses uncertain event timing through optimization over an uncertainty set. Relevant prior for contact-onset or release-time correction: timing uncertainty is a substantive research topic, not an unexplored add-on.

## From optimized demonstrations to policies

**33. Guided Policy Search.** Levine and Koltun, ICML 2013. **M.** [Proceedings](https://proceedings.mlr.press/v28/levine13.html). Uses trajectory optimization to guide policy search. Foundational precedent for the general optimization-to-learning argument; the distinct problem here is noisy, cross-embodiment contact interaction and total pipeline cost.

**34. Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations — DAPG.** Rajeswaran et al., RSS 2018. **M.** [Paper](https://m.roboticsproceedings.org/rss14/p49.pdf), [primary record](https://arxiv.org/abs/1709.10087). Demonstration initialization and demonstration-augmented policy updates enable challenging dexterous tasks. Useful foundation for evaluating whether correction adds value beyond providing demonstrations at all.

**35. Residual Reinforcement Learning for Robot Control.** Johannink et al., ICRA 2019; 2018 preprint. **M.** [Paper](https://arxiv.org/abs/1812.03201). Combines conventional control with learned residual actions, including contact-rich assembly. A standard mechanism for retaining useful model-based structure while learning the missing behavior; novelty must lie in a more specific interface or capability.

**36. Opt2Skill: Imitating Dynamically-feasible Whole-Body Trajectories for Versatile Humanoid Loco-Manipulation.** Liu et al., 2024 preprint; official project lists RA-L 2025. **M.** [Project](https://opt2skill.github.io/), [primary record](https://arxiv.org/abs/2409.20514). Dynamics-aware optimized motions, including torque considerations, become references for RL and hardware execution. One of the clearest precursors to the argument for better reference feasibility.

**37. Guided Reinforcement Learning for Robust Multi-Contact Loco-Manipulation.** Sleiman, Mittal and Hutter, CoRL 2024; proceedings 2025. **M.** [Proceedings](https://proceedings.mlr.press/v270/sleiman25a.html). Uses optimized demonstrations with adaptive phase behavior to obtain robust multi-contact skills. Useful for contact recovery and temporal alignment; adding adaptive phase tracking alone would not establish a new method.

**38. Should We Learn Contact-Rich Manipulation Policies from Sampling-Based Planners?** Zhu et al., 2024 preprint. **M.** [Primary record](https://arxiv.org/abs/2412.09743). Examines why planner trajectories can be poor imitation data and improves their consistency. Particularly important for your thesis: physical validity and planning success do not by themselves imply that the resulting policy is easy to learn. Its behavior-cloning setting differs from reference-guided RL.

## Force and tactile extensions

**39. MimicTouch: Leveraging Multi-modal Human Tactile Demonstrations for Contact-rich Manipulation.** CoRL 2024; proceedings 2025. **M.** [Proceedings](https://proceedings.mlr.press/v270/yu25c.html). Transfers tactile strategies through imitation and online residual learning. Relevant if you later target assembly or force-sensitive manipulation, but it uses information absent from ordinary video-based Track 2 inputs.

**40. ReForce: Learning Force-aware Retargeting for Dexterous Manipulation.** Wu et al., August 2026 preprint. **M.** [Paper](https://arxiv.org/html/2608.15560v1). Learns force-tracking residuals over kinematic retargeting, supporting online and offline use. Distinguish sensed force supervision from your inferred net object wrench: Newton–Euler demand does not identify the demonstrator's per-finger forces.

**41. DexTaG: Tactile-as-Guidance in Reinforcement Learning for Dexterous Manipulation.** Yang et al., September 2026 preprint. **M.** [Paper](https://arxiv.org/html/2609.33882v1). Human tactile contact guidance supports RL and trajectory-conditioned transfer. Relevant to amortized retargeting and contact-pattern supervision, but comparison must acknowledge its extra sensing and training information.

## Recent work to check before freezing the novelty claim

**42. VICAR: Video Imitation with Contact-Aware Retargeting for Humanoid Interactive Skills.** October 2026 preprint. **M.** [Paper](https://arxiv.org/html/2610.13079v1). Contact-aware retargeting emphasizes global contact poses, motion speed and collision clearance before learned motion generation. A very recent precedent for geometrically consistent interactive skills; inspect before claiming contact timing or world-frame anchors are novel.

**43. OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport.** September 2026 preprint. **M.** [Paper](https://arxiv.org/html/2609.36602v1). Uses surface relations and jointly adjusts robot/object motion. A close warning for the proposed latent-object extension: changing both trajectories is already studied. Preserve task intent and evaluate against the challenge's clean target, not your own modified reference.

**44. DexAgent: An Agentic Human2Sim2Robot Framework for Dexterous Manipulation with Self-Evolving Tool Library.** Wang et al., September 2026 preprint; revised October. **S.** [Primary record](https://arxiv.org/abs/2609.35318). Video-conditioned simulation reconstruction, trajectory optimization, verification and robot-data generation are assembled into an adaptive workflow. Relevant to broad pipeline claims; it is not the most controlled comparison for a fixed reconstruction input and unchanged RL learner.

**45. Weave: Learning Whole-Body Dexterous Loco-Manipulation from Human-Object Interactions.** September 2026 preprint. **S, abstract-level screening.** [Primary record](https://arxiv.org/abs/2609.16683). Describes contact-aware retargeting and coordinated whole-body/dexterous control. A recent neighboring system to inspect in full before submission; no detailed performance or compute claim is inferred here from its abstract.

## Additional geometric baseline

**46. Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations.** Lakshmipathy, Hodgins and Pollard, ACM Transactions on Graphics 44(2), 2025. **S, abstract/introduction screening.** [Publisher](https://doi.org/10.1145/3723872), [author paper](https://www.andrew.cmu.edu/user/aslakshm/pdfs/TOG2025KinematicMotionRetargeting.pdf). Transfers contact areas through shape matching and IK across different hands, including pickup, reorientation and release. A strong geometric predecessor; use it to define a meaningful contact-preserving baseline without assuming that kinematic consistency establishes dynamic execution.

## How to turn this into the paper bibliography

For the final manuscript, cite the primary works that explain the actual algorithm and comparisons, rather than all 46 indiscriminately. A focused related-work section can have three threads: physical interaction transfer; correction under uncertain contact; and optimized demonstrations for efficient learning. Use older papers to establish the foundations and recent papers to identify the exact experimental competitors.

Freeze the versions used for reproduction. In particular, do not combine a result from an early preprint with the methods or limitations of a later revision without identifying the difference. Recheck acceptance status and newly released code before submission. Broad terms such as “first,” “novel contact correction” and “existing methods do not scale” require stronger evidence than this targeted search provides.
