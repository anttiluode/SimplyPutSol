# SimplyPutSol v1 — frozen benchmark receipt

Run: [GitHub Actions success, full 650-episode protocol](https://github.com/anttiluode/SimplyPutSol/actions/runs/40164587906).
The workflow uploaded a **simplyputsol-benchmark** artifact containing per-seed
results.json and an offline report.html. Six unit tests passed, including an
analytic cable tangent checked against numerical finite differences.

## Main results

Three seeds, four targets in order (bell -> switch -> light -> horn), 650
training episodes per target, 80 fresh postures per target per wind direction.
Mean first-attempt endpoint **distance** (lower is better). Each learner gets
the same *two scalar sensory probes* per update, not corrected joint angles.

| Model | Parameters | Bell just learned | Bell after other three | Final four-task distance | Final contact | Wind-conditioned action difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stateful plastic cable | 312 | 0.029 | 0.736 | 0.396 | 21.8% | 0.888 |
| Phase locked to zero | 312 | 0.033 | 0.611 | **0.390** | **32.1%** | 0.712 |
| Cable parameters frozen | 312 | 0.029 | 0.714 | 0.395 | 29.8% | 0.979 |
| No persistent cable state | 312 | 0.418 | 0.973 | 0.819 | 3.3% | 0.000 |
| Trainable tanh RNN | 620 | 0.030 | 0.714 | 0.665 | 13.4% | 2.161 |
| Static local experts | 624 | 0.395 | **0.392** | 0.460 | 2.7% | 0.000 |

Wind-conditioned difference is the norm of the change in predicted motor
command when earlier wind sign is swapped while final posture/task inputs are
identical. A nonzero number shows response, **not necessarily the correct
compensation**. The ideal compensation magnitude here is approximately 0.86.
The RNN's large response illustrates why action difference alone is not skill.

## Preregistered gates, unchanged after seeing results

- G1 **PASS**: stateful cable differs across wind histories; state-erased
  cable cannot.
- G2 **PASS (implementation)**: learning receives scalar probes, not direct
  optimal target/joint-angle labels or a 2-D error vector.
- G3 **PASS**: plastic cable learned the final horn target relative to its
  untrained initial behavior.
- G4 **PASS (weak evidence)**: cable final distance 0.396 versus RNN 0.665.
  Unequal parameter counts and untuned hyperparameters limit interpretation.
- G5 **FAIL**: the plastic cable does **not** beat its fixed-parameter version
  (0.396 versus 0.395). Switching off phase slightly *improves* scores
  (0.390). No evidence that learned phase/tuning earns its place.

## Interpretation: positive memory test, negative retention result

The first bell becomes extremely precise, then is overwritten. Plastic cable
bell distance rises from **0.029 to 0.736** (approximately 25x), the frozen
cable from 0.029 to 0.714, and the phase-off cable from 0.033 to 0.611.
The ordinary static route benchmark stays near 0.395 -> 0.392, though it
has worse final mean performance than the recurrent cable variants.

Thus, **state is useful for this deliberately history-dependent task, but
resonant cable plasticity did not prevent catastrophic forgetting or show a
benefit over freezing the cable.** All routes participate in the shared output
and its online updates, so they can interfere. This v1 supplies no evidence
for selective physical route insulation, no real dendritic demonstration,
and no AI architecture advantage beyond the tested toy baseline.

The outcome is not numerically comparable to SimplyPutClaude's original
corrections-per-reach study; this environment adds a hidden wind history
and replaces that study's successful-reach teacher with scalar probes.

## What a separate v2 would need

Keep v1 frozen. To test the original idea of selective branching, introduce
activity-dependent branch competition/plasticity and a capacity/skill-overlap
sweep, preserving fair sensor budgets and comparing with ordinary learned
context gating. A further step should replace reversible paired probes
and graded proximity with single-trajectory noisy, delayed or binary-contact
feedback. Preregister new gates rather than reinterpreting v1.
