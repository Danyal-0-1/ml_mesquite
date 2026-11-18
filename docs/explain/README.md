# Learning the project: the explanation pipeline

These five files teach Steps 1 and 2 of Mesquite ML to a reader who knows the project's goal but not this code: Step 2
in full depth, Step 1 as far as Step 2 depends on it. Read them in order; each answers one question and links to the
next.

```text
MENTAL_PICTURE.md ─► PIPELINE_EXPLAINED.md ─► MATH_EXPLAINED.md ─► CODE_EXPLAINED.md ─► THEORY.md
 "what happens?"     "what happens to the     "why does it work?"   "how is it built?"     "where do the ideas
                      data, stage by stage?"                                                come from?"
```

| # | file | the question it answers | reading time |
|---|---|---|---|
| 1 | [MENTAL_PICTURE.md](MENTAL_PICTURE.md) | What happens, as a picture you can replay with your eyes closed? No equations. | 20 min |
| 2 | [PIPELINE_EXPLAINED.md](PIPELINE_EXPLAINED.md) | What happens to the data, stage by stage, from an AMASS file to `docs/RESULTS_STEP2.md`, with the real numbers of one session? | 45 min |
| 3 | [MATH_EXPLAINED.md](MATH_EXPLAINED.md) | Why does it work? Every formula derived step by step, with small worked examples that a test reproduces. | 90 min |
| 4 | [CODE_EXPLAINED.md](CODE_EXPLAINED.md) | How is it built? A map of the modules, one command traced line by line, the Step 2 code block by block, the tests, how to extend and debug. | 60 min |
| 5 | [THEORY.md](THEORY.md) | Where do the ideas come from? Rotations, inertial sensing, identifiability, joint constraints, directional statistics, estimation and evaluation theory, with full references. | 60 min |

Every file ends with self-check questions and their answers. Two tests keep the files honest:
`tests/test_explain_links.py` checks that every link and every `path:line` reference resolves to the current code, and
`tests/test_explain_examples.py` reproduces every worked example of MATH_EXPLAINED.md.

## Which file for which question?

| question | open |
|---|---|
| Why can nothing ever recover north or the body's facing? | [MENTAL_PICTURE §2](MENTAL_PICTURE.md#2-the-room-north-can-never-be-known), [MATH_EXPLAINED §2](MATH_EXPLAINED.md#2-the-measurement-model-and-what-cannot-be-known), [THEORY §3](THEORY.md#3-gauge-freedom-and-identifiability) |
| What is δ, and why does a sign error leave 14°? | [MATH_EXPLAINED §3](MATH_EXPLAINED.md#3-the-target-and-the-outward-correction) |
| What does Step 0 do to a postcard? | [PIPELINE_EXPLAINED stage 5](PIPELINE_EXPLAINED.md#stage-5-step-0-the-post-office), [MATH_EXPLAINED §4](MATH_EXPLAINED.md#4-step-0s-arithmetic) |
| How are two pods' clocks put on one time line? | [MATH_EXPLAINED §5](MATH_EXPLAINED.md#5-one-clock-for-parent-and-child), [CODE_EXPLAINED §3.2](CODE_EXPLAINED.md#32-the-clock-clockpy) |
| Why does the knee give δ directly, and when not? | [MENTAL_PICTURE §6.1](MENTAL_PICTURE.md#61-the-knee-is-a-door-hinge), [MATH_EXPLAINED §6](MATH_EXPLAINED.md#6-the-hinge-the-knees) |
| How is the knee axis found without being told? | [MATH_EXPLAINED §6.4](MATH_EXPLAINED.md#64-finding-the-axis-from-motion), [THEORY §4.1](THEORY.md#41-hinge-joints-and-axis-identification) |
| Why can a 2-DoF joint give two answers? | [MATH_EXPLAINED §7](MATH_EXPLAINED.md#7-the-2-dof-constraint-elbows-wrists-ankles) |
| Why is κ zero at the hip? | [MATH_EXPLAINED §8](MATH_EXPLAINED.md#8-the-range-of-motion-ball-joints), [MENTAL_PICTURE §6.5](MENTAL_PICTURE.md#65-ball-joints-walls-instead-of-pins) |
| What is κ, and why is the product of two beliefs a belief? | [MATH_EXPLAINED §9](MATH_EXPLAINED.md#9-the-von-mises-distribution-and-the-filter), [THEORY §5](THEORY.md#5-directional-statistics) |
| Why does a reboot make a joint forget? | [MENTAL_PICTURE §7](MENTAL_PICTURE.md#7-confidence-and-memory), [CODE_EXPLAINED §3.4](CODE_EXPLAINED.md#34-the-events-and-the-filter-geopy-vmfilterpy) |
| Why was κ recalibrated, and on which data? | [MATH_EXPLAINED §10](MATH_EXPLAINED.md#10-calibration), [THEORY §6.4](THEORY.md#64-correlated-residuals-overstate-confidence) |
| How is "GEO-ID is better than B1" decided? | [MATH_EXPLAINED §11](MATH_EXPLAINED.md#11-metrics-and-statistics), [THEORY §7](THEORY.md#7-evaluation-theory) |
| Which function implements a formula, and which test checks it? | [MATH_EXPLAINED](MATH_EXPLAINED.md) (every formula names both), [CODE_EXPLAINED](CODE_EXPLAINED.md) |
| How do I add a joint model, an estimator, a check or a dataset? | [CODE_EXPLAINED §6](CODE_EXPLAINED.md#6-how-to-extend-the-code) |
| Something looks wrong: sign flips, units, leaks? | [CODE_EXPLAINED §7](CODE_EXPLAINED.md#7-how-to-debug) |
| What do the Step 2 results say about H2? | [docs/RESULTS_STEP2.md](../RESULTS_STEP2.md), [PIPELINE_EXPLAINED stage 10](PIPELINE_EXPLAINED.md#stage-10-the-results) |

## Related documents

[README.md](../../README.md) (status and quickstart) · [MUST_KNOW.md](../MUST_KNOW.md) (what to understand before
trusting a number) · [MATH.md](../MATH.md) (the reference version of the mathematics) · [PIPELINE.md](../PIPELINE.md)
and [CODE.md](../CODE.md) (reference tables) · [VALIDATION.md](../VALIDATION.md) (every check, generated) ·
[RESULTS_STEP2.md](../RESULTS_STEP2.md) (generated) · the briefs in [brief/](../brief/).

*Start with [MENTAL_PICTURE.md](MENTAL_PICTURE.md).*
