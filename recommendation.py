"""
Agent 3 — Content Recommendation Agent.

Responsibility: Given the top-scoring queries where the target domain is NOT
appearing, generate 3–5 specific, actionable content recommendations per query.

Prompt engineering decisions:
- System prompt establishes an "AI content strategist" persona who specialises
  in appearing in AI-generated answers — not just traditional SEO. This matters
  because AI visibility strategy differs from blue-link SEO (structured data,
  Q&A format, authoritative citations are more important than backlinks).
- Each recommendation must include: content_type, title, rationale, keywords,
  priority. The schema is fully defined in the prompt — no ambiguity.
- We pass the top 5 queries in a single LLM call to allow the model to avoid
  redundant recommendations across queries. This is the one place where batching
  is beneficial: the model can spot that two queries need similar content.
- priority assignment instructions are explicit in the prompt to prevent
  everything being "high".
"""
import logging
from typing import Any

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior AI content strategist specialising in making
business websites appear in AI-generated answers (ChatGPT, Claude, Perplexity,
Google AI Overviews).

Your recommendations must be specific and actionable — not generic advice like
"write better content". Each recommendation must:
  - Identify the EXACT content format needed (blog_post, landing_page, comparison_page,
    faq, case_study, tool_review, glossary_page, etc.)
  - Give a specific, compelling title a writer could use immediately
  - Explain precisely WHY this content would help the domain appear in AI answers
    for the target query (not just "it will rank better")
  - List 4–8 specific keywords / topics to cover, prioritising terms that appear
    directly in the queries

Priority assignment:
  high:   Query volume > 1000 AND domain completely absent from AI answers
  medium: Query volume 200–1000 OR domain partially visible
  low:    Niche query OR informational intent only

CRITICAL: Return ONLY a valid JSON object matching the schema below. No markdown, no prose.

Output schema:
{
  "recommendations": [
    {
      "query_uuid": "from the input",
      "content_type": "blog_post",
      "title": "Specific, actionable title",
      "rationale": "2-3 sentences on why this content will improve AI visibility for this query",
      "target_keywords": ["keyword1", "keyword2", "keyword3"],
      "priority": "high"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """Generate 3–5 content recommendations to help {domain} appear
in AI-generated answers for the following high-opportunity queries where it is currently
NOT visible.

Domain: {domain}
Industry: {industry}

Queries (sorted by opportunity score, highest first):
{queries_block}

For each query, produce 1–2 recommendations (3–5 total across all queries).
Avoid redundant recommendations — if two queries need similar content, combine them.
Do NOT recommend content for queries where the domain is already visible.
Return the JSON object. No extra text.
"""


class ContentRecommendationAgent(BaseAgent):
    """
    Agent 3: Generates content recommendations for queries where the domain is absent.
    """

    def run(self, queries: list[dict], profile: dict) -> tuple[list[dict], int]:
        """
        Generate content recommendations for the given queries.

        Args:
            queries: List of scored query dicts (from Agent 2 + opportunity score).
                     Each must have: query_uuid, query_text, opportunity_score,
                     estimated_search_volume, domain_visible.
            profile: Dict with domain, industry.

        Returns:
            (list of recommendation dicts, tokens_used)
        """
        # Only recommend for queries where domain is not visible
        target_queries = [q for q in queries if not q.get("domain_visible", True)]

        if not target_queries:
            logger.info("Agent 3: No non-visible queries — no recommendations to generate.")
            return [], 0

        # Take top 5 by opportunity score to keep the prompt focused
        top_queries = sorted(target_queries, key=lambda q: q.get("opportunity_score", 0), reverse=True)[:5]

        # Build the queries block for the prompt
        lines = []
        for q in top_queries:
            lines.append(
                f'- query_uuid: {q["query_uuid"]}\n'
                f'  query: "{q["query_text"]}"\n'
                f'  opportunity_score: {q.get("opportunity_score", 0):.2f}\n'
                f'  search_volume: {q.get("estimated_search_volume", 0):,}'
            )
        queries_block = "\n".join(lines)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            domain=profile["domain"],
            industry=profile["industry"],
            queries_block=queries_block,
        )

        logger.info("Agent 3: Generating recommendations for %d queries (domain=%s)", len(top_queries), profile["domain"])
        raw, tokens = self._call_llm(SYSTEM_PROMPT, user_prompt)

        parsed = self._safe_parse_json(raw, fallback={"recommendations": []}, context="ContentRecommendationAgent")

        raw_recs: list[dict[str, Any]] = parsed.get("recommendations", []) if isinstance(parsed, dict) else []

        # Validate and normalise
        valid_recs: list[dict[str, Any]] = []
        valid_uuids = {q["query_uuid"] for q in top_queries}
        valid_priorities = {"high", "medium", "low"}
        valid_content_types = {
            "blog_post", "landing_page", "comparison_page", "faq", "case_study",
            "tool_review", "glossary_page", "guide", "tutorial", "whitepaper"
        }

        for rec in raw_recs:
            if not isinstance(rec, dict):
                continue
            query_uuid = rec.get("query_uuid", "")
            if query_uuid not in valid_uuids:
                # If model hallucinated a UUID, skip
                logger.debug("Agent 3: Skipping rec with unknown query_uuid: %s", query_uuid)
                continue

            content_type = rec.get("content_type", "blog_post")
            if content_type not in valid_content_types:
                content_type = "blog_post"

            priority = rec.get("priority", "medium")
            if priority not in valid_priorities:
                priority = "medium"

            keywords = rec.get("target_keywords", [])
            if not isinstance(keywords, list):
                keywords = []

            valid_recs.append({
                "query_uuid": query_uuid,
                "content_type": content_type,
                "title": rec.get("title", "Untitled Content Piece"),
                "rationale": rec.get("rationale", ""),
                "target_keywords": [str(k) for k in keywords],
                "priority": priority,
            })

        logger.info("Agent 3: Generated %d recommendations (tokens=%d)", len(valid_recs), tokens)
        return valid_recs, tokens
