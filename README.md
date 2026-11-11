# Shipwright

A self-healing support agent. Lexsi AI Safety and Alignment Hackathon, Track 3, problem statement 8.

A tool-calling support agent on Tiny Aya Global that finds its own failed runs, classifies why they
failed, fixes the tool or retrains the model, and redeploys only when AuditKIT shows it is better.

- Design: [docs/system-design.html](docs/system-design.html)
- Setup: [docs/setup/gcp-setup.md](docs/setup/gcp-setup.md), [docs/setup/hf-setup.md](docs/setup/hf-setup.md), [docs/setup/cloud-runs.md](docs/setup/cloud-runs.md)
- Running the loop: [docs/pipeline/baseline.md](docs/pipeline/baseline.md), [docs/pipeline/training.md](docs/pipeline/training.md), [docs/pipeline/gate.md](docs/pipeline/gate.md)
- Debugging a run: [docs/ops/debugging.md](docs/ops/debugging.md), [docs/ops/observability.md](docs/ops/observability.md)
- Rules for coding agents: [AGENTS.md](AGENTS.md)

Status: round 1 has run for real - baseline, classification, tool patches and teacher-generated
training data all exist. Training a LoRA adapter and running a real gate decision are next.
