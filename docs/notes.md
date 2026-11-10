# Lexsi hackathon notes (2026-11-07)

## Dates
- Submit by Nov 14, end of day AoE. The submission form comes by email a few days before.
- Demo Day shortlist on Nov 16 (10 to 20 teams). Demo Day is Nov 22.

## Judging (kickoff slides, hackathon.lexsi.ai/kickoff-slides/)
- Double-blind. More than one judge scores each team. Normalized averages pick the Demo Day teams.
- Rubric: Write-up clarity /10, Practical usefulness /10, Code/notebook quality /10, Overall review /100.
- First, an internal team checks validity: public artifacts, write-up, checklist, no duplicates.
  Then it tags each submission to a track.

## Hard rules
- Model must be one of: Tiny Aya Base/Global/Fire/Water/Earth, Aya Expanse 8B, Aya Expanse 32B.
  Command R7B is NOT on the list.
- Track 3: AgentTune + AuditKIT, used together as one connected system.
- Everything public on HF: models, adapters, logs, datasets, benchmarks (two HF collections, ungated).

## HF article checklist (sections)
1. Problem statement and approach, in brief
2. Hackathon repos used: which ones, for what, which features
3. Models used and their licenses
4. Datasets and benchmarks used and their licenses
5. Model artifacts public, as an HF collection
6. Data artifacts public, as an HF collection
7. Experiments run, findings, what else was tested
8. With more credits: your approach, before and after results
9. Teammates tagged (judges mask names for the double-blind round)
- A markdown write-up in the GitHub repo is accepted if HF article access is a problem.

## Discord facts
- Docs: agenttune.lexsi.ai, audit.lexsi.ai, curatorkit.lexsi.ai/latest/
- Office hours run daily: see the server Events tab.
- Off-list ideas or scope checks: open a ticket in #report_issues.
- Organizers (Timidan): not publishing harmful weights is OK, and tagging teammates in the article is OK.
- Translation tip from a participant: Gemini 2.5 Flash Lite for Hindi/Telugu prompt translation.

## Review of Ishank's repo (github.com/Ishankgiri/lexsi-agent-distill, commit 9c1dae8)
Problem statement: Track 3 "Agentic distillation". Teacher (GPT-4o/Claude) -> curate -> QLoRA SFT -> eval.

Blockers:
1. Student model is CohereLabs/c4ai-command-r7b-12-2024, which is NOT an allowed model.
   Fix: DEFAULT_STUDENT_MODEL_ID in src/business_agent/config.py -> CohereLabs/tiny-aya-global.
   Tiny Aya is also Cohere2ForCausalLM, so the MLX path should still work. Accept the gate on HF first.
2. The Lexsi libraries are optional fallbacks. On a Mac the MLX path uses neither AgentTune nor
   AuditKIT. Validity checks look for real use of the track repos. Make AgentTune
   create_distill_trainer (on a CUDA box) and AuditKIT the primary path.
3. CHANGELOG says auditkit.agent_eval is an empty placeholder. That is out of date: on main it
   now has episodes_from_agenttune, outcome oracles, pass^k reliability. AuditKIT also has
   ak.load_bfcl + ToolCallF1 for BFCL v3.

Weak spots:
- 4 mock tools with hash-based outputs. Almost the same as AgentTune's own bfsi_kyc example.
  This hurts "practical usefulness" and originality.
- Nothing trained or generated yet (data/raw and data/curated are empty).

Strengths to keep: clean staged pipeline, schemas, tests, the efficiency/ROI report.

Suggested upgrades (in order):
1. Swap the model and run end to end once with the mock teacher, then a real teacher.
2. AgentTune distill on Colab/Kaggle/HF credits; AuditKIT scoring + BFCL as a public benchmark.
3. Add harder cases: multi-step, irrelevance (no tool should be called), bad arguments.
4. Add a Hindi/Hinglish test split (Tiny Aya is multilingual; few teams will test this).
5. Optional layer: GRPO after distillation with a tool-call reward (AgentTune create_agentic_trainer).

## Tracks page (saved HTML, read 2026-11-07)
- "Self-healing agents" is Track 3 problem statement 8, in the same track as "Agentic distillation" (statement 3).
  It is not a separate track. Changing statements does not change the track or its competitors.
- Statement 8 text: detect failing runs, classify why they failed, generate corrective training examples,
  retrain, and gate redeployment on measured accuracy.
- The page lists only AgentTune + AuditKIT for Track 3. CuratorKIT and AlignTune are listed under Track 1.
  Confirm in a #report-issues ticket before you count CuratorKIT as a required Track 3 repo.
- One submission counts: the last one before the deadline. Organizers recommend one track.

## Self-healing research (2026-11-07, subagent web search; numbers marked "check" are from secondary sources)
- Taxonomy: MAST, arxiv 2503.13657 (14 failure modes, 3 categories). Add our own "environment/transient" label and exclude it from training.
- Corrections: Agent-R, arxiv 2501.11425 (splice a correct branch at the first wrong step; code bytedance/Agent-R).
  Implemented 2026-11-11 (Mitushi), `docs/pipeline/agent-r.md`. Values from the paper: c_uct 0.25, 4 children per
  expansion at temperature 1, depth 20, beta 0.2, alpha 0.5 / 0.7 / 1.0 per iteration, eta 0.2.
  Our changes, decided before any Agent-R round: 24 rollouts per task (paper 200 to 300, budget), the revision
  thought inline before the first good action (paper: an extra Assistant / "OK." turn pair, which our user
  simulator would have to answer), LoRA lr 2e-4 as in our other training (paper: full fine-tune, lr 2e-5).
  Paper and code differ on pairs: we pair each bad leaf with its adjacent good leaf (longest shared prefix),
  as the abstract says; the released code pairs any leaves with a reward gap.
  APIGen-MT, arxiv 2504.03601 (verified multi-turn tool data, keep only runs that reach the goal state; xLAM-2 beats GPT-4o on tau-bench per abstract).
  APIGen, arxiv 2406.18518: verify in 3 steps (format, real execution, semantic).
- No paper compares tool-layer fixes with weight fixes. That comparison is our own testable claim.
- Gate: use pass^k from tau-bench (arxiv 2406.12045, repo sierra-research/tau2-bench), not pass@1. Held-out tasks + regression set of past passes.
- Forgetting: replay old/general examples in LoRA SFT (arxiv 2504.17780, I-LoRA 2402.18865).
- Tool-description layer: GEPA, arxiv 2507.19457 (prompt evolution, 100 to 500 evals).
- Cite: Reflexion 2303.11366, Agent-R, APIGen-MT, tau-bench, MAST. Also FireAct 2310.05915, AgentRefine 2501.01702, survey 2507.21046.
- No mature open-source self-healing agent repo found.
- Tiny Aya license: CC-BY-NC + acceptable-use addendum. No commercial claims in the write-up without care.
- Stand-out ideas: one regression test per failure class; show that training on unfiltered env errors lowers pass^k;
  report gain per fix type per GPU hour; record conflicts when a tool fix and a weight fix both solve a task.

## AgentTune already has a self-healing loop (2026-11-07)
- Guide: github.com/Lexsi-Labs/AgentTune docs/user-guide/self-healing.md. Examples: examples/self_heal_*_real.py, closed_loop_real.py.
- Path 1: Project.heal() -> FailureDetector -> SelfHealLoop(FailureClassifier, TrainingExampleGenerator, trainer_factory).
- Path 2: FullClosedLoop on an audit.jsonl -> RetrainingTrigger (6 conditions, 2 safety gates, e.g. max_drop_rate) ->
  retrain -> DeploymentGate (task A/B + trajectory mean) -> deploy or keep old.
- Trap: if there are fewer than min_test_samples successful runs, the gate APPROVES by default.
- Also: IsolatedTool / isolate_tools() runs tools in a separate process.
- Meaning: build on this, do not rebuild it. Our additions: 3-way routing (tool fix / weight fix / no fix),
  AuditKIT as the gate (pass^k, BFCL), a real domain (tau2-bench), and measured results per round.

## Competition scan (2026-11-07, 18 GitHub repos found)
- Track 3 so far: 4 repos. Track 2: 7. Track 1: 1. Unclear: 6.
- Closest to us: UEddy/Kataa (Track 3, Tiny Aya + AgentTune, refuses unjustified financial actions).
  It tried AgentTune's DPO self-heal path; a gate written before the run rejected the repair
  (it escalated 99% of tasks). Strong write-up. It is not a failure-routing loop.
- Others in Track 3: ridakalam52-design/safetoolcall-tiny-aya (distillation + DPO), PRINCE2-AI/lexsi-agentic-hinglish
  (Hinglish expense agent), NSEVEjk0/permit-agent.
- Note: someone pushed agenttune-tau2-mock-grpo-smoke-data to HF, so tau2-bench is not unique.

## Hermes agent (NousResearch/hermes-agent, MIT) study (2026-11-07)
- Learns by editing memory and skills (text), never weights. No failure classifier for fixes, no eval gate.
- Error taxonomy only for runtime API errors (agent/error_classifier.py, FailoverReason, about 30 values).
- Post-turn review fork (agent/background_review.py) with a tool whitelist; writes only to memory/skills.
- Skill review prompt has a "do not capture" list: environment failures, transient errors. Same idea as our "no fix" class.
- tools/skill_ledger.py: JSONL audit of every edit with content-addressed before/after snapshots, one-step rollback.
- agent/curator.py: unused skills go stale, then archived, never deleted.
- Borrow for our tool-fix branch: the ledger + rollback for each tool patch, and the reviewer fork with a whitelist.
