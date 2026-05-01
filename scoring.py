"""
Agent 2 — Visibility Scoring Agent.

Responsibility: For each discovered query, determine:
  1. Whether the target domain appears in a typical AI-generated answer.
  2. Estimated search volume (supplemented with real DataForSEO data when available).
  3. Competitive difficulty (0–100).

Real data integration:
  DataForSEO's Keywords Data API is called first for each query. If credentials
  are not set or the call fails, the agent falls back to LLM-estimated values.
  This is intentional: the system degrades gracefully rather than crashing.

Prompt engineering decisions:
- System prompt defines the agent as an "AI SERP analyst" — this improves the
  model's ability to reason about what AI assistants typically surface.
- We explicitly tell the model it is simulating an AI answer analysis, not a
  traditional SERP check. This prevents confabulation about Google rankings.
- The schema embeds a `reasoning` field so the model must justify its visibility
  call — this improves accuracy (chain-of-thought in the output) and lets us
  audit unusual decisions.
- Per-query batching: each query is scored independently. This costs more tokens
  than batching but isolates failures — a bad response for query N won't corrupt
  query N+1.
"""
import logging
import os
import base64

import requests

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI search analyst who specialises in understanding
how AI assistants (ChatGPT, Claude, Perplexity, Google AI Overviews) generate answers
to commercial and research queries.

Your task is to simulate an AI visibility analysis: given a search query and a target
domain, assess whether that domain would typically appear in an AI-generated answer for
that query. AI answers typically surface:
  - Authoritative review sites / comparison hubs
  - Official brand pages for directly mentioned products
  - High-ranking content that is comprehensive and well-cited

CRITICAL: Return ONLY a valid JSON object matching the schema below. No markdown, no prose.

Output schema:
{
  "query_text": "the original query",
  "domain_visible": true or false,
  "visibility_position": 1-5 or null (null if domain_visible is false),
  "estimated_search_volume": integer (monthly searches, your best estimate if no real data provided),
  "competitive_difficulty": integer 0-100 (how hard it is to appear in AI answers for this query),
  "reasoning": "1-2 sentence explanation of the visibility decision"
}

competitive_difficulty guidance:
  80-100: Dominated by major brands / established comparison sites — very hard to break in
  50-79:  Competitive but achievable with quality content
  20-49:  Moderate competition, good content can appear
  0-19:   Low competition, niche or long-tail query
"""

USER_PROMPT_TEMPLATE = """Analyse AI visibility for the following query.

Target domain: {domain}
Industry: {industry}
Query: "{query_text}"
{volume_hint}

Simulate what a typical AI assistant answer would look like for this query, then
determine whether {domain} would appear in that answer.

Return the JSON object. No extra text.
"""


# ---------------------------------------------------------------------------
# DataForSEO helper
# ---------------------------------------------------------------------------

def _fetch_dataforseo_volume(query: str) -> int | None:
    """
    Fetch real search volume from DataForSEO Keywords Data API.
    Returns None if credentials not set or request fails.
    """
    login = os.getenv("DATAFORSEO_LOGIN")
    password = os.getenv("DATAFORSEO_PASSWORD")
    if not login or not password:
        return None

    credentials = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
    }
    payload = [{"keywords": [query], "location_code": 2840, "language_code": "en"}]  # US

    try:
        resp = requests.post(
            "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live",
            headers=headers,
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # Navigate the DataForSEO response structure
        results = data.get("tasks", [{}])[0].get("result", [])
        if results:
            return results[0].get("search_volume")
    except Exception as exc:
        logger.warning("DataForSEO volume fetch failed for '%s': %s", query, exc)

    return None


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class VisibilityScoringAgent(BaseAgent):
    """
    Agent 2: Scores each discovered query for visibility and opportunity.
    """

    def run(self, query_text: str, profile: dict) -> tuple[dict, int]:
        """
        Score a single query for the given profile's domain.

        Args:
            query_text: The natural-language question to score.
            profile: Dict with domain, industry keys.

        Returns:
            (scoring_result dict, tokens_used)
        """
        domain = profile["domain"]
        industry = profile["industry"]

        # Attempt to get real search volume
        real_volume = _fetch_dataforseo_volume(query_text)
        volume_hint = (
            f"Real search volume data: {real_volume:,} monthly searches (from DataForSEO)."
            if real_volume is not None
            else "No real volume data available — estimate based on industry knowledge."
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            domain=domain,
            industry=industry,
            query_text=query_text,
            volume_hint=volume_hint,
        )

        logger.debug("Agent 2: Scoring query='%s' for domain=%s", query_text[:60], domain)
        raw, tokens = self._call_llm(SYSTEM_PROMPT, user_prompt)

        result = self._safe_parse_json(raw, fallback={}, context=f"VisibilityScoringAgent:{query_text[:40]}")

        if not result:
            logger.warning("Agent 2: Empty result for query '%s'. Using defaults.", query_text)
            result = {}

        # Normalise / enforce types
        domain_visible = bool(result.get("domain_visible", False))
        visibility_position = result.get("visibility_position")
        if domain_visible and visibility_position is None:
            visibility_position = 3  # default mid-rank if model omitted it

        estimated_volume = real_volume or int(result.get("estimated_search_volume") or 500)
        competitive_difficulty = int(result.get("competitive_difficulty") or 50)
        competitive_difficulty = max(0, min(100, competitive_difficulty))

        return {
            "query_text": query_text,
            "domain_visible": domain_visible,
            "visibility_position": visibility_position if domain_visible else None,
            "estimated_search_volume": estimated_volume,
            "competitive_difficulty": competitive_difficulty,
            "reasoning": result.get("reasoning", ""),
            "volume_source": "dataforseo" if real_volume is not None else "llm_estimate",
        }, tokens
