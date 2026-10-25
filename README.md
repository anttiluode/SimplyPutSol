# SimplyPutSol — the returning probe changes the cable

A tiny experiment inspired by the bike bell hidden behind the rider's seat:
at first the hand fumbles; with experience a movement becomes easy.

[SimplyPutClaude](https://github.com/anttiluode/SimplyPutClaude) tested
whether existing local-expert routing prevents catastrophic forgetting,
with a correction algorithm that teaches successful joint displacement.
**SimplyPutSol asks a different question:** can a shared dynamical network
learn by making its *own exploratory actions* and comparing their returns?

## First result (frozen v1)

The [measured benchmark receipt](RESULTS.md) reports **4/5 predeclared gates passing**. The cable learns the first bell very well but **forgets it after later tasks** (mean distance 0.029 -> 0.736). It beats the tested RNN in final mean distance (0.396 vs 0.665), yet **frozen cables (0.395) and phase-off cables (0.390) do as well or better**. The key claimed extra mechanism therefore **fails its ablation**. Static local experts retained the bell more reliably. Do not read this as a discovery of a new learning mechanism.

### The apparatus

A two-joint arm reaches for bell, switch, light, or horn. Task cue and
proprioception are visible. A brief gust signal occurs between the first
and last input frames; the resulting wind deflection still affects the
reach. The gust is absent from the final observation, so a useful controller
has to retain some information about earlier inputs.

Each learning step tests two symmetric, randomized motor reaches from a
resettable posture. The environment returns scalar proximity intensity and
a contact flag, but **no desired joint angles and no 2-D error arrow**.
The learner uses the intensity difference as a stochastic estimate of
which change in motor output would help. All six controllers receive the
identical sensor interface and probe budget.

The main network has 24 *parallel complex leaky cable branches*:

    z[k,t+1] = rho[k] * exp(i*omega[k]) * z[k,t] + tanh(C[k] @ input[t])
    motor_command = B @ real(z_final)

Each branch carries a resident signal, frequency/phase preference,
leak and learned input coupling. It also accumulates **local tangents**
of its final contribution with respect to its own tuning and phase.
Paired probe feedback changes the participating branches. There is no
target-angle teacher or cross-branch BPTT. This is an established class
of finite-difference/three-factor learning, not an invented biological rule.

Controls: ordinary trainable RNN (three-step BPTT), ordinary static
local-expert routing, frozen cable parameters, zero phase, and cable state
wiped between frames. A successful control is a valid negative result
against the extra phase/plasticity machinery.

### Reproduce

Python 3.10+ with NumPy:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python simplyputsol.py

The full benchmark runs six models across three seeds and four sequential
tasks (650 training reaches each). A quick integration check:

    python simplyputsol.py --episodes 30 --seeds 1 --evaluation 10

Outputs: results/results.json (all per-seed records and the five gates)
and results/report.html (offline illustrated summary). GitHub Actions
also saves the full benchmark as a downloadable CI artifact.

**Important:** Learning uses *graded scalar proximity*, not just a
binary click. The simulator can reset before each paired probe and thus
is more permissive than an actual moving bicycle. Parameter counts are
reported but are not exactly matched; stronger baselines, noise stress,
realistic motor irreversibility and long sequences remain untested.

The central falsifiable question is whether locally changing cable
phase and tuning beats ordinary recurrent prediction/control and the
fixed-cable ablation on this synthetic history-dependent task. Merely
beating a no-memory controller is expected, not evidence of novelty.

See [PREREGISTRATION.md](PREREGISTRATION.md), then inspect recorded
results. Nearby projects:
[LoopHolonomyNeuron](https://github.com/anttiluode/LoopHolonomyNeuron),
[Operaattori](https://github.com/anttiluode/Operaattori),
[SimplyPutClaude](https://github.com/anttiluode/SimplyPutClaude).
