# Shipwright

A self-healing support agent. Lexsi AI Safety and Alignment Hackathon, Track 3, problem statement 8.

A tool-calling support agent on Tiny Aya Global that finds its own failed runs, classifies why they
failed, fixes the tool or retrains the model, and redeploys only when AuditKIT shows it is better.

- Design: [docs/system-design.html](docs/system-design.html)
- Setup: [docs/setup/gcp-setup.md](docs/setup/gcp-setup.md), [docs/setup/hf-setup.md](docs/setup/hf-setup.md), [docs/setup/cloud-runs.md](docs/setup/cloud-runs.md)
- Running the loop: [docs/pipeline/loop.md](docs/pipeline/loop.md) (one command: `shipwright run --config configs/rounds/<name>.yaml`);
  the steps it runs: [baseline.md](docs/pipeline/baseline.md), [training.md](docs/pipeline/training.md), [gate.md](docs/pipeline/gate.md), [tool_fix.md](docs/pipeline/tool_fix.md)
- Debugging a run: [docs/ops/debugging.md](docs/ops/debugging.md), [docs/ops/observability.md](docs/ops/observability.md)
- Rules for coding agents: [AGENTS.md](AGENTS.md)

```sh
./scripts/setup_tau2.sh                                               # .venv with tau2, AgentTune, AuditKIT and `shipwright`
.venv/bin/shipwright doctor                                           # is this machine ready?
.venv/bin/shipwright run --config configs/rounds/default.yaml --dry-run   # the whole loop offline, free
```

Status and results: [WRITEUP.md](WRITEUP.md).
