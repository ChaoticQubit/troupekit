---
title: Roadmap
description: What each troupekit release adds, from 0.1 to the 1.0.0 on-device core and beyond.
---

# Roadmap

Troupekit aims at one Rust core that runs agents, durable workflows, RAG, memory, evals and proofs on the device: phones, desktops, servers and dev boards, switching between local and cloud models by policy.

## Releases

| Release | Theme | A user can |
|---|---|---|
| 0.1 | Survives a restart | Kill the app mid-run and resume from SQLite. Shipped. |
| 0.2 | Knows your docs | Search local docs by keyword, vector or hybrid, and remember facts. Shipped. |
| 0.3 | Thinks offline | Run on a local GGUF model, or switch to cloud by policy, and trace everything with one flag. This release. |
| 0.4 | In your app | Coming in 0.4: call troupekit from Swift, Kotlin and Python, and use Apple and Android on-device models. |
| 0.5 | Works in teams | Coming in 0.5: team, crew and handoff patterns in TOML, parallel branches with declared reducers, fan-out over lists, and forking a run from any checkpoint. |
| 0.6 | Plugs in anywhere | Coming in 0.6: use MCP tools, and expose troupekit as an MCP server. |
| 0.7 | Proven | Coming in 0.7: prove workflow properties with Bend, replay runs deterministically, run evals. |
| 1.0.0 | On-device core | The V1 acceptance criteria met on all six tier-1 targets. |

## After 1.0

- 2.0.0, Breadth: any major provider, server vector store or language, plus A2A and planners.
- 3.0.0, Edge and embedded: microcontrollers, browsers and NPUs, voice and vision on the device, device and server sync.
- 4.0.0, Platform: a hosted agent server, a visual studio, a registry, distributed graphs and on-device LoRA.

## Versions

Major versions are only the 1.0.0 to 4.0.0 milestones. A minor release adds a feature; a patch release fixes. Each release is a vertical slice and ships with an example you can run.
