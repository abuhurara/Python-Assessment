"""
Agent 1 — Query Discovery Agent.

Responsibility: Given a business profile, generate 10–20 realistic, commercially
relevant questions that users ask AI assistants when searching for products or
services in this competitive space.

Prompt engineering decisions:
- System prompt establishes a strict "SEO research specialist" persona so the
  model focuses on search intent rather than generic Q&A.
- Output schema is fully specified in the prompt: array of objects with
  query_text and commercial_intent (0.0-1.0). This means we never have to
  guess the structure.
- Explicit instruction: "Return ONLY a valid JSON array. No markdown, no prose."
  This eliminates the most common failure mode (code fences wrapping the JSON).
- Temperature is left at default (1.0) because we *want* diversity in the
  discovered queries. Determinism would produce less varied results.
"""
import logging
from typing import Any

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert SEO research specialist and AI search analyst.
Your role is to identify the exact questions that potential customers type into AI
assistants (ChatGPT, Claude, Perplexity, Google AI Overviews) when researching
products in a specific competitive space.

Focus on:
- Commercial / transactional queries (comparisons, "best X", pricing, alternatives)
- Evaluation queries (reviews, pros/cons, use-cases)
- Problem-solution queries ("how do I X with Y tool")
- Competitor comparison queries

CRITICAL: Return ONLY a valid JSON array. No markdown fences, no prose, no explanations.
Every element must have exactly the fields shown in the schema below.

Output schema:
[
  {
    "query_text": "string — the exact natural-language question",
    "commercial_intent": 0.85
  }
]

commercial_intent is a float 0.0–1.0:
  1.0 = strong purchase/evaluation intent (e.g. "best X tool", "X vs Y pricing")
  0.5 = moderate intent (how-to guides with clear tool focus)
  0.0 = purely informational with no commercial signal
"""

USER_PROMPT_TEMPLATE = """Generate 10–20 realistic AI search queries for the following business profile.

Business: {name}
Domain: {domain}
Industry: {industry}
Description: {description}
Main competitors: {competitors}

Rules:
1. Queries must be natural language questions a real user would ask an AI assistant.
2. Include a mix of: comparison queries, "best X" queries, problem-solution queries,
   feature-specific queries, and pricing/value queries.
3. At least 30% of queries should directly mention the business name or domain.
4. At least 30% should mention competitor names.
5. All queries must be commercially relevant — no purely academic questions.
6. Return exactly the JSON array. No extra text before or after.
"""


class QueryDiscoveryAgent(BaseAgent):
    """
    Agent 1: Discovers high-value queries in a business's competitive space.
    """

    def run(self, profile: dict) -> tuple[list[dict], int]:
        """
        Discover queries for the given profile.

        Args:
            profile: Dict with keys name, domain, industry, description, competitors.

        Returns:
            (list of query dicts, tokens_used)
        """
        competitors_str = ", ".join(profile.get("competitors", []))
        user_prompt = USER_PROMPT_TEMPLATE.format(
            name=profile["name"],
            domain=profile["domain"],
            industry=profile["industry"],
            description=profile.get("description", ""),
            competitors=competitors_str,
        )

        logger.info("Agent 1: Discovering queries for domain=%s", profile["domain"])
        raw, tokens = self._call_llm(SYSTEM_PROMPT, user_prompt)

        queries = self._safe_parse_json(raw, fallback=[], context="QueryDiscoveryAgent")

        # Validate and normalise each item
        validated: list[dict[str, Any]] = []
        for item in queries:
            if not isinstance(item, dict):
                continue
            query_text = item.get("query_text", "").strip()
            if not query_text:
                continue
            commercial_intent = float(item.get("commercial_intent", 0.5))
            commercial_intent = max(0.0, min(1.0, commercial_intent))
            validated.append({"query_text": query_text, "commercial_intent": commercial_intent})

        logger.info("Agent 1: Discovered %d valid queries (tokens=%d)", len(validated), tokens)
        return validated, tokens
