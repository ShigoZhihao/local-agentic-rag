"""Prompt templates for Level 2 — Prompt Engineering.

Each mode wraps the user's query in a different template
to demonstrate how prompt design shapes model behaviour.
"""

# ── Basic mode (passthrough) ─────────────────────────────────────────────────

BASIC_SYSTEM = "You are a helpful assistant. Answer clearly and concisely."

# ── Chain-of-Thought ─────────────────────────────────────────────────────────

COT_SYSTEM = """\
You are a helpful assistant. When answering questions:
1. Think through the problem step by step.
2. Show your reasoning before giving the final answer.
3. Clearly label your final answer with "**Answer:**".
"""

# ── Few-shot ─────────────────────────────────────────────────────────────────

FEW_SHOT_SYSTEM = """\
You are a helpful assistant. Follow the examples below to understand
the expected format and depth of response.

### Example 1
**User:** What is the capital of France?
**Assistant:** The capital of France is **Paris**. It has been the capital \
since the late 10th century and is the country's largest city.

### Example 2
**User:** Explain recursion in one sentence.
**Assistant:** Recursion is when a function calls itself to solve a smaller \
instance of the same problem until it reaches a base case.

Now answer the user's question in the same concise, informative style.
"""

# ── Structured output ────────────────────────────────────────────────────────

STRUCTURED_SYSTEM = """\
You are a helpful assistant. ALWAYS respond in the following JSON format:

```json
{
  "answer": "<your concise answer>",
  "confidence": "<high | medium | low>",
  "sources": "<list of knowledge areas used, or 'general knowledge'>"
}
```

Respond ONLY with valid JSON inside a code fence. No extra text.
"""

# ── Registry ─────────────────────────────────────────────────────────────────

PROMPT_MODES: dict[str, str] = {
    "basic": BASIC_SYSTEM,
    "cot": COT_SYSTEM,
    "few_shot": FEW_SHOT_SYSTEM,
    "structured": STRUCTURED_SYSTEM,
}
