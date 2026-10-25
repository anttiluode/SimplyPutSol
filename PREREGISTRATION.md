# SimplyPutSol v1: predeclared protocol

The main sweep and comparison criteria were fixed before looking at the
benchmark outputs. Structural unit tests and syntax fixes are allowed;
hyperparameter tuning after the main sweep requires a separately labeled v2.

## Actual question

Can parallel stateful resonant cable operators, trained from their own
outgoing motor probes and scalar sensory returns, perform sequential skills
and remember a transient disturbance at least as well as ordinary recurrent
learning? Does locally changing cable phase and tuning earn its cost?

## Boundary, sensor, and comparison

- Four targets, sequential order bell -> switch -> light -> horn.
- 2-DOF planar arm, random start posture and a task cue.
- At frame 1, wind direction arrives as a brief pulse; at final frame 2
  only cue and posture remain. Plant wind deflects the final motor command.
- The environment has target coordinates but the *learner* cannot see them,
  obtain a desired joint angle, or read a 2-D target-error vector. It receives
  two *scalar graded proximity* readings at symmetric motor probes.
- The same two probes and number of sensor evaluations train every controller.
  This resettable, shaped-feedback simulator is easier than real click-only
  tactile exploration; no contact-only learning claim is made.

## Model conditions

1. cable: independent complex leaky resonators with learnable phase and
   input tuning, plus learned motor readout; local per-branch tangents and
   a global scalar improvement estimate; no across-branch BPTT.
2. phase0: phase locked at zero, other plasticity retained.
3. frozen: readout learns, but cable phase/tuning are held.
4. no_state: wipe resident state each frame, leaving final input only.
5. rnn: ordinary tanh RNN, three-step backpropagation, equal sensor probes.
6. route: static mixture of locally weighted linear experts, equal probes.

Three seeds, 650 learning episodes per task per seed, 80 held-out starting
postures per task per wind. No training on evaluation postures. Main criterion:
mean first-reach hand distance; also report contact rate, bell retention,
parameter count and behavior when wind history is reversed with the same
current input. All full per-seed phase records are retained.

## Yes/no decision rules

- G1: phase/state-bearing cable responds differently to opposing past wind
  signs while no-state cable responds identically. Constructed mechanism check.
- G2: target angles and direction-to-target teaching signal are absent
  from the training interface. Implementation audit.
- G3: the cable reduces held-out distance on the horn after learning the
  four tasks, compared with its own untrained horn performance.
- G4: final four-target mean distance for cable is lower than RNN.
- G5: final four-target mean distance for plastic cable is lower than
  frozen-cable version.

G1/G2 are not research discoveries, and even G4/G5 passing would not
establish novelty against known RNN, perturbation, e-prop, adaptive
filter and mixture-of-experts methods. Report failure without retuning.
No claims of neural realism, true physical impedance propagation or
general continual-learning performance follow.
