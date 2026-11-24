# Evaluation protocol

How Shipwright decides that a new system is better, and the rules that keep that decision honest. The numbers this
protocol produced are in `submission/WRITEUP.md`; `python scripts/reproduce.py` recomputes them from the published runs.

## Task splits

tau2-bench retail, split once by exact task id (`configs/splits.yaml`, read by `src/shipwright/heal/splits.py`):

| Split | Tasks | Used for |
|---|---|---|
| train | the whole tau2 `train` split, 74 | everything the loop learns from: baselines, teacher replays, corrections |
| held-out | 20 of the 40 tau2 `test` tasks | the gate, and nothing else |
| hidden | the other 20 `test` tasks | one final report, after every decision was made |

The 40 test ids were shuffled with seed 300 and cut in two. They are listed explicitly, so a tau2 update cannot move
them. Enforcement is in code, not convention: `scripts/baseline.py` refuses a `test` run without explicit
`--task-ids`, refuses hidden ids unless `--open-hidden` is passed, and `build_sft.py` refuses any run that is not
from the train split.

Why three splits: the gate selects among candidates on the held-out tasks, so after many gates the held-out number is
optimistic. The hidden split is the check on that selection. It was opened once, on 2026-11-15, after all
decisions; it is now spent, and a further independent check needs new tasks (another tau2 domain), not a second look
at the same ones.

## The gate

`configs/gate.yaml`, decision logic in `src/shipwright/heal/gate.py` (pure functions), metrics from AuditKIT
(`src/shipwright/heal/eval_metrics.py`, `bfcl_eval.py`). The candidate and the deployed system both run the 20 held-out
tasks x 3 trials (60 runs each) on the same harness, in the same job, with the agent at temperature 0. The base is
re-run at every gate, never reused.

All eight checks must pass. Too little evidence is a refusal, never an approval.

| Check | Rule | Why |
|---|---|---|
| comparable | same 20 held-out task ids, same split, k = 3 | a gate never compares different tasks |
| valid_runs | each system >= 54 valid runs (90% of 60) and >= 18 tasks with all 3 trials valid | infra errors are excluded; too few means refuse |
| pass_1 | candidate pass^1 >= current pass^1 | the headline: task success may not fall |
| pass_k | candidate pass^3 >= current pass^3 | reliability may not fall |
| target | the failure labels the candidate targets fall by >= 5 points (3 of 60 runs) | the fix must fix what it was built for |
| non_target | every other failure label rises by <= 5 points | a fix may not create a new kind of failure |
| bfcl | BFCL v3 accuracy (simple, multiple, irrelevance; 100 each) falls by <= 2 points | narrow fine-tuning must not erase general tool use |
| agenttune_gate | AgentTune `DeploymentGate.evaluate_decision` approves on >= 54 paired runs | the library's own regression check, with a floor it lacks by default |

Failure labels come from one classifier, applied the same way to both systems: our rules, AgentTune's
`FailureDetector`, and AgentTune's `FailureClassifier` mapped into the label list of `src/shipwright/heal/classify.py`.
`gate_decision.json` records, per system, how many failures each source labelled and how many classifier calls failed
(`label_coverage`). The gate job only serves the models and runs the conversations; classification and the decision
run afterwards in Cloud Run, where the Vertex token can refresh, and a decision with any failed classifier call is
never reused.

## What is measured

- **Headline: task success on the held-out tasks**, pass^1 and pass^k (AuditKIT `AgentEvalRunner`,
  `FinalStateAssertion` on tau2's verified final state). The problem statement asks to gate redeployment on measured
  accuracy; this is that accuracy.
- **Supporting:** per-label failure rates, tool-call error rate, BFCL v3, AuditKIT `ToolCallF1` against tau2's expert
  actions, and trajectory efficiency (`scripts/trajectory_efficiency.py`: wasted calls, calls per expert call).
- **Teacher ceiling:** the teacher on the same held-out tasks (`shipwright stage ceiling`), so a student result is
  read as a share of the gap closed (`scripts/teacher_student_report.py`).

## Reading a result

- 60 runs per system with about 6 base passes: two tasks flipping moves pass^1 by 10 points. One gate's gain of 3
  passes is not a result on its own.
- Every rate is reported with a Wilson interval; base against candidate uses an exact McNemar test on the shared
  (task, trial) pairs; the gap to the teacher uses a task bootstrap.
- Gates on the same held-out tasks select on them; the hidden split is the number to quote for generalisation.

## Frozen values and their changes

Frozen before round 1 (2026-11-11): the failure label list, the gate thresholds, k, seeds, the teacher, user simulator
and judge models. A frozen value changes only with a dated entry here, written before the change is used:

| Date | Change | Reason |
|---|---|---|
| 2026-11-11 | splits and gate sized for 20 held-out tasks x k=3 (floors 54 runs, 18 tasks) | the gate had been sized for all 40 test tasks with no hidden set; the final number would have come from the tasks that picked the rounds |
| 2026-11-12 | the rule classifier also finds `premature_transfer` (a transfer before any attempt to authenticate); the label list is unchanged | the most common round 1 failure (37 of 59 valid runs) was counted by no rule, so no candidate could target it |
| 2026-11-13 | `classify.classifier_model`: AgentTune's `FailureClassifier` labels every failed run of both systems | with rules only, 28 of a candidate's 46 failures were unlabelled and counted as a new kind; the base had the same failure hidden under tool-error labels. Gates before this change are not comparable on `target` / `non_target` |
| 2026-11-14 | classification moved out of the GPU job into Cloud Run (no config change) | the Vertex token minted at job launch expired before classification in two gates, so their label checks were not measurements |
| 2026-11-25 | `advisory_checks: [target, pass_1]` (reported, no longer blocking); new blocking check `tool_errors`: tool-call error rate may rise by at most 5 points | on 20 tasks x 3 these two checks are mostly noise (round 16 scored pass^1 0.183 and 0.100 in two gates); a tool-calling regression (round 18: 34% vs 6% errors at equal pass rate) was caught only by `target` |

Every decision records the gate's `config_sha` (a hash of its parsed values): `c807805003a25bcc` for gates 3 to 8,
`9e811350bc846a6d` for gates 10 to 16 (after the classifier change). Comments in the file do not change it.
