# Two parallel implementation workstreams

The baseline source lives on main. Use separate checkouts/worktrees and branches:

| Branch | Scope | Primary files |
|---|---|---|
| work/tier2-rl | Five-episode Tier 2 trainer, adapters, profiling, evaluation | training/, experiments/ |
| work/tier3-reconstruction | Video-to-human/object motion and canonical input adapter | reconstruction/ |

Each agent should read correction/METHOD.md,
correction/COMPUTE_AND_IMPLEMENTATION_HANDOFF.md, and its workstream README.
These are implementation assignments; neither pipeline is already implemented.

Do not switch branches in another agent's active working directory. From a clone,
create separate worktrees if they do not already exist:

~~~text
git fetch origin
git worktree add .worktrees/tier2-rl work/tier2-rl
git worktree add .worktrees/tier3-reconstruction work/tier3-reconstruction
~~~

If a branch exists only remotely, use, for example:

~~~text
git worktree add -b work/tier2-rl .worktrees/tier2-rl origin/work/tier2-rl
git worktree add -b work/tier3-reconstruction .worktrees/tier3-reconstruction origin/work/tier3-reconstruction
~~~

Run dependency bootstrap in each worktree. Environments are also checkout-local;
do not overwrite another worker's upstream checkout or dependency environment.

Keep checkpoints, datasets, model weights, and generated trajectories outside
Git. Share large immutable inputs using an explicitly configured external cache,
not tracked files or another worktree's mutable output directory.

Prefer workstream-local changes. Coordinate changes to retargeting/, correction/,
the canonical HumanMotion interface, or shared setup files before both branches
depend on them. Include those changes in a small separate commit for review.

Commit and push on the workstream branch. Integrate completed work into main by
a reviewed merge or PR; there is no reason to force-push main.

## Experiment definition

“Our retargeting + correction versus their retargeting” measures the combined
pipeline. It does not isolate the correction. Preserve three named variants:

1. nvidia_body_baseline: the pinned SOMA/G1 body-only retargeting path.
2. contact_aware: our retargeting without dynamics-guided correction.
3. contact_aware_corrected: our retargeting plus correction.

The first is a specific baseline implementation, not a reproduction of every
CHORD retargeting or training component. Keep the same downstream learner for
all variants. A later geometry-only correction ablation isolates wrench guidance.

The proposed five-episode roster is recorded in experiments/tier2_pilot.json.
It was selected before RL outcomes and covers boxes/book and smaller grasped
objects. Frame duration is not training time. Begin with one episode and one
seed, verify physical execution and exporter correctness, then expand the roster.
No GPU jobs or paid infrastructure are launched by creating these branches.
