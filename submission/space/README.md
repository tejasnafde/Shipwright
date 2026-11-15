---
title: Shipwright
emoji: ⚓
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: A 3B Tiny Aya support agent that heals itself, unattended
tags:
- agents
- self-healing
- tau2-bench
- tiny-aya
- lexsi-hackathon
models:
- mitushigoyal5/shipwright-lora
- CohereLabs/tiny-aya-global
datasets:
- mitushigoyal5/shipwright-sft
- mitushigoyal5/shipwright-runs
---

Shipwright: a 3.35B Tiny Aya support agent on tau2-bench retail that finds, sorts and fixes its own failures, and redeploys only when an 8-check AgentTune + AuditKIT gate approves. Lexsi Labs AI Safety & Alignment Hackathon, Track 3.
