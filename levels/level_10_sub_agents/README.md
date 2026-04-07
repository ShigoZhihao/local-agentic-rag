# Level 10 — Sub-Agents

**Core loop:** `user input → Orchestrator → [parallel subgraph spawning] → merge results → output`

Scale the agent system horizontally. An orchestrator fans out work to
parallel sub-agents, each running as an independent LangGraph subgraph.

**Framework:** LangGraph (subgraph composition)

## What you learn here

- LangGraph subgraph composition and nesting
- Parallel execution of sub-agents (fan-out / fan-in)
- Inter-agent communication and result merging
- Resource management across concurrent subgraphs

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| Composable skills | Level 11 |
| Autonomous scheduling | Level 12 |

## Status

> **Placeholder** — implementation coming soon.
