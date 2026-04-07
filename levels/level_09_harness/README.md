# Level 9 — Harness

**Core loop:** `user input → Multi-Agent Graph → [context compaction + persistence] → output`

Add production-grade context management: summarise long histories to stay
within token budgets, and persist graph state across restarts.

**Framework:** LangGraph + persistence

## What you learn here

- Conversation context compaction (summarisation of long histories)
- LangGraph persistence: PostgreSQL / SQLite checkpointer
- Session resume across application restarts
- Token budget management and context window optimisation

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| Parallel sub-agents | Level 10 |
| Composable skills | Level 11 |
| Autonomous scheduling | Level 12 |

## Status

> **Placeholder** — implementation coming soon.
