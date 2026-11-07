# Self-healing support agent

Lexsi AI Safety and Alignment Hackathon, Track 3, problem statement 8.

A tool-calling support agent on Tiny Aya Global that finds its own failed runs, classifies why they
failed, fixes the tool or retrains the model, and redeploys only when AuditKIT shows it is better.

- Design: [docs/system-design.html](docs/system-design.html)
- Notes and research: [docs/notes.md](docs/notes.md)
- Cloud setup: [docs/gcp-setup.md](docs/gcp-setup.md)
- Rules for coding agents: [AGENTS.md](AGENTS.md)

Status: design stage. No code yet.
