"""
Unit tests for all three agents using mocked LLM responses.

Strategy:
- Mock `BaseAgent._call_llm` to return canned JSON strings.
- This lets us test prompt parsing, validation, and fallback logic
  without making real API calls (fast, free, deterministic).
- We also test the opportunity score formula independently.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.agents.discovery import QueryDiscoveryAgent
from app.agents.scoring import VisibilityScoringAgent
from app.agents.recommendation import ContentRecommendationAgent
from app.utils.scoring import compute_opportunity_score


# ---------------------------------------------------------------------------
# Agent 1 — QueryDiscoveryAgent
# ---------------------------------------------------------------------------

class TestQueryDiscoveryAgent:

    def _make_agent(self):
        # Patch the Anthropic client so no real API call is made
        with patch("app.agents.base.get_client", return_value=MagicMock()):
            return QueryDiscoveryAgent()

    def test_happy_path_returns_validated_queries(self):
        agent = self._make_agent()
        mock_response = json.dumps([
            {"query_text": "Best SEO content tool 2024", "commercial_intent": 0.9},
            {"query_text": "Frase vs Surfer SEO", "commercial_intent": 0.85},
            {"query_text": "How to write content briefs with AI", "commercial_intent": 0.6},
        ])

        with patch.object(agent, "_call_llm", return_value=(mock_response, 500)):
            profile = {
                "name": "Frase", "domain": "frase.io", "industry": "SEO Tools",
                "description": "AI content briefs", "competitors": ["surferseo.com"]
            }
            queries, tokens = agent.run(profile)

        assert len(queries) == 3
        assert queries[0]["query_text"] == "Best SEO content tool 2024"
        assert queries[0]["commercial_intent"] == 0.9
        assert tokens == 500

    def test_commercial_intent_clamped_to_0_1(self):
        agent = self._make_agent()
        mock_response = json.dumps([
            {"query_text": "Some query", "commercial_intent": 1.5},
            {"query_text": "Another query", "commercial_intent": -0.2},
        ])

        with patch.object(agent, "_call_llm", return_value=(mock_response, 100)):
            queries, _ = agent.run({"name": "X", "domain": "x.io", "industry": "Y", "description": "", "competitors": []})

        assert queries[0]["commercial_intent"] == 1.0
        assert queries[1]["commercial_intent"] == 0.0

    def test_empty_query_text_filtered_out(self):
        agent = self._make_agent()
        mock_response = json.dumps([
            {"query_text": "", "commercial_intent": 0.5},
            {"query_text": "   ", "commercial_intent": 0.5},
            {"query_text": "Valid query", "commercial_intent": 0.7},
        ])

        with patch.object(agent, "_call_llm", return_value=(mock_response, 100)):
            queries, _ = agent.run({"name": "X", "domain": "x.io", "industry": "Y", "description": "", "competitors": []})

        assert len(queries) == 1
        assert queries[0]["query_text"] == "Valid query"

    def test_malformed_json_returns_empty_list(self):
        agent = self._make_agent()

        with patch.object(agent, "_call_llm", return_value=("this is not json at all!!", 50)):
            queries, _ = agent.run({"name": "X", "domain": "x.io", "industry": "Y", "description": "", "competitors": []})

        assert queries == []

    def test_markdown_fenced_json_is_parsed(self):
        agent = self._make_agent()
        mock_response = '```json\n[{"query_text": "Best X tool", "commercial_intent": 0.8}]\n```'

        with patch.object(agent, "_call_llm", return_value=(mock_response, 100)):
            queries, _ = agent.run({"name": "X", "domain": "x.io", "industry": "Y", "description": "", "competitors": []})

        assert len(queries) == 1
        assert queries[0]["query_text"] == "Best X tool"


# ---------------------------------------------------------------------------
# Agent 2 — VisibilityScoringAgent
# ---------------------------------------------------------------------------

class TestVisibilityScoringAgent:

    def _make_agent(self):
        with patch("app.agents.base.get_client", return_value=MagicMock()):
            return VisibilityScoringAgent()

    def test_happy_path_visible_domain(self):
        agent = self._make_agent()
        mock_response = json.dumps({
            "query_text": "Best SEO tool",
            "domain_visible": True,
            "visibility_position": 2,
            "estimated_search_volume": 3400,
            "competitive_difficulty": 72,
            "reasoning": "The domain is a major player in this space.",
        })

        with patch("app.agents.scoring._fetch_dataforseo_volume", return_value=None):
            with patch.object(agent, "_call_llm", return_value=(mock_response, 300)):
                result, tokens = agent.run("Best SEO tool", {"domain": "frase.io", "industry": "SEO"})

        assert result["domain_visible"] is True
        assert result["visibility_position"] == 2
        assert result["estimated_search_volume"] == 3400
        assert result["competitive_difficulty"] == 72
        assert tokens == 300

    def test_not_visible_domain_has_null_position(self):
        agent = self._make_agent()
        mock_response = json.dumps({
            "query_text": "Best SEO tool",
            "domain_visible": False,
            "visibility_position": None,
            "estimated_search_volume": 1200,
            "competitive_difficulty": 55,
            "reasoning": "Domain is not prominent here.",
        })

        with patch("app.agents.scoring._fetch_dataforseo_volume", return_value=None):
            with patch.object(agent, "_call_llm", return_value=(mock_response, 200)):
                result, _ = agent.run("Best SEO tool", {"domain": "frase.io", "industry": "SEO"})

        assert result["domain_visible"] is False
        assert result["visibility_position"] is None

    def test_dataforseo_volume_overrides_llm_estimate(self):
        agent = self._make_agent()
        mock_response = json.dumps({
            "query_text": "Best SEO tool",
            "domain_visible": False,
            "visibility_position": None,
            "estimated_search_volume": 500,  # LLM estimate
            "competitive_difficulty": 60,
            "reasoning": "Nope.",
        })

        with patch("app.agents.scoring._fetch_dataforseo_volume", return_value=8900):  # Real data
            with patch.object(agent, "_call_llm", return_value=(mock_response, 200)):
                result, _ = agent.run("Best SEO tool", {"domain": "frase.io", "industry": "SEO"})

        assert result["estimated_search_volume"] == 8900
        assert result["volume_source"] == "dataforseo"

    def test_malformed_json_returns_safe_defaults(self):
        agent = self._make_agent()

        with patch("app.agents.scoring._fetch_dataforseo_volume", return_value=None):
            with patch.object(agent, "_call_llm", return_value=("BROKEN OUTPUT", 100)):
                result, _ = agent.run("Some query", {"domain": "x.io", "industry": "Y"})

        # Should not raise — should return safe defaults
        assert result["domain_visible"] is False
        assert result["competitive_difficulty"] == 50

    def test_difficulty_clamped_to_0_100(self):
        agent = self._make_agent()
        mock_response = json.dumps({
            "query_text": "q",
            "domain_visible": False,
            "visibility_position": None,
            "estimated_search_volume": 100,
            "competitive_difficulty": 150,  # Out of range
            "reasoning": "test",
        })

        with patch("app.agents.scoring._fetch_dataforseo_volume", return_value=None):
            with patch.object(agent, "_call_llm", return_value=(mock_response, 100)):
                result, _ = agent.run("q", {"domain": "x.io", "industry": "Y"})

        assert result["competitive_difficulty"] == 100


# ---------------------------------------------------------------------------
# Agent 3 — ContentRecommendationAgent
# ---------------------------------------------------------------------------

class TestContentRecommendationAgent:

    def _make_agent(self):
        with patch("app.agents.base.get_client", return_value=MagicMock()):
            return ContentRecommendationAgent()

    SAMPLE_QUERIES = [
        {
            "query_uuid": "uuid-1",
            "query_text": "Best AI content brief tool",
            "opportunity_score": 0.81,
            "estimated_search_volume": 1200,
            "domain_visible": False,
        },
        {
            "query_uuid": "uuid-2",
            "query_text": "Frase vs Surfer SEO",
            "opportunity_score": 0.75,
            "estimated_search_volume": 900,
            "domain_visible": False,
        },
    ]

    def test_happy_path_generates_recommendations(self):
        agent = self._make_agent()
        mock_response = json.dumps({
            "recommendations": [
                {
                    "query_uuid": "uuid-1",
                    "content_type": "comparison_page",
                    "title": "Best AI Content Brief Tools in 2024: A Complete Comparison",
                    "rationale": "A comparison page directly addresses the query intent.",
                    "target_keywords": ["ai content brief", "content brief tool", "frase review"],
                    "priority": "high",
                },
                {
                    "query_uuid": "uuid-2",
                    "content_type": "blog_post",
                    "title": "Frase vs Surfer SEO: Which Is Better for Content Teams?",
                    "rationale": "Head-to-head comparison captures both brand queries.",
                    "target_keywords": ["frase vs surfer seo", "seo content tools"],
                    "priority": "high",
                },
            ]
        })

        with patch.object(agent, "_call_llm", return_value=(mock_response, 600)):
            recs, tokens = agent.run(self.SAMPLE_QUERIES, {"domain": "frase.io", "industry": "SEO"})

        assert len(recs) == 2
        assert recs[0]["content_type"] == "comparison_page"
        assert recs[0]["priority"] == "high"
        assert tokens == 600

    def test_unknown_query_uuid_filtered_out(self):
        agent = self._make_agent()
        mock_response = json.dumps({
            "recommendations": [
                {
                    "query_uuid": "hallucinated-uuid-99",
                    "content_type": "blog_post",
                    "title": "Something",
                    "rationale": "...",
                    "target_keywords": [],
                    "priority": "low",
                },
            ]
        })

        with patch.object(agent, "_call_llm", return_value=(mock_response, 200)):
            recs, _ = agent.run(self.SAMPLE_QUERIES, {"domain": "frase.io", "industry": "SEO"})

        # Hallucinated UUID should be filtered out
        assert recs == []

    def test_invalid_priority_normalised_to_medium(self):
        agent = self._make_agent()
        mock_response = json.dumps({
            "recommendations": [
                {
                    "query_uuid": "uuid-1",
                    "content_type": "blog_post",
                    "title": "Test",
                    "rationale": "...",
                    "target_keywords": [],
                    "priority": "CRITICAL",  # Invalid
                },
            ]
        })

        with patch.object(agent, "_call_llm", return_value=(mock_response, 150)):
            recs, _ = agent.run(self.SAMPLE_QUERIES, {"domain": "frase.io", "industry": "SEO"})

        assert recs[0]["priority"] == "medium"

    def test_all_visible_queries_returns_empty(self):
        agent = self._make_agent()
        visible_queries = [
            {**q, "domain_visible": True} for q in self.SAMPLE_QUERIES
        ]

        # Should not even call the LLM
        with patch.object(agent, "_call_llm") as mock_llm:
            recs, tokens = agent.run(visible_queries, {"domain": "frase.io", "industry": "SEO"})

        assert recs == []
        assert tokens == 0
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Opportunity Score Formula
# ---------------------------------------------------------------------------

class TestOpportunityScore:

    def test_high_volume_not_visible_high_intent_scores_near_top(self):
        score = compute_opportunity_score(
            estimated_search_volume=50_000,
            competitive_difficulty=30,
            domain_visible=False,
            commercial_intent=1.0,
        )
        assert score > 0.7, f"Expected > 0.7, got {score}"

    def test_zero_volume_caps_at_low_score(self):
        score = compute_opportunity_score(
            estimated_search_volume=0,
            competitive_difficulty=50,
            domain_visible=False,
            commercial_intent=0.5,
        )
        # volume_score=0, gap=0.30, intent=0.10, ease=0.075 → 0.475
        assert score < 0.55

    def test_visible_domain_reduces_score(self):
        score_invisible = compute_opportunity_score(1000, 50, False, 0.5)
        score_visible = compute_opportunity_score(1000, 50, True, 0.5)
        assert score_invisible > score_visible

    def test_score_always_in_0_1_range(self):
        cases = [
            (0, 0, True, 0.0),
            (100_000, 0, False, 1.0),
            (100_000, 100, False, 1.0),
            (500, 50, False, 0.5),
        ]
        for vol, diff, vis, intent in cases:
            score = compute_opportunity_score(vol, diff, vis, intent)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score} for ({vol}, {diff}, {vis}, {intent})"

    def test_lower_difficulty_yields_higher_score(self):
        score_easy = compute_opportunity_score(1000, 10, False, 0.5)
        score_hard = compute_opportunity_score(1000, 90, False, 0.5)
        assert score_easy > score_hard
