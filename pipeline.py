"""
Pipeline Orchestrator Service.

Coordinates the three agents in sequence:
  Agent 1 (QueryDiscoveryAgent) → Agent 2 (VisibilityScoringAgent) → Agent 3 (ContentRecommendationAgent)

Failure isolation:
  - If Agent 2 fails for a single query, the error is logged and processing continues
    for the remaining queries. The query is saved with visibility_status="unknown".
  - If Agent 1 fails completely, the run is marked "failed" immediately.
  - If Agent 3 fails, the run is marked "completed" with a warning — recommendations
    are optional from a pipeline perspective.

Correlation IDs:
  Each pipeline run gets a UUID (PipelineRun.uuid) that appears in all log messages
  for that run. This makes tracing easy in production logs.
"""
import logging
from datetime import datetime, timezone

from app import db
from app.agents.discovery import QueryDiscoveryAgent
from app.agents.scoring import VisibilityScoringAgent
from app.agents.recommendation import ContentRecommendationAgent
from app.models.profile import BusinessProfile
from app.models.pipeline_run import PipelineRun
from app.models.query import DiscoveredQuery
from app.models.recommendation import ContentRecommendation
from app.utils.scoring import compute_opportunity_score

logger = logging.getLogger(__name__)


def run_pipeline(profile: BusinessProfile) -> PipelineRun:
    """
    Execute the full 3-agent pipeline for a profile.

    Creates a PipelineRun record, runs all agents, persists results, and returns
    the completed (or failed) PipelineRun.
    """
    # Create run record
    pipeline_run = PipelineRun(profile_uuid=profile.uuid, status="running")
    db.session.add(pipeline_run)
    db.session.commit()

    run_id = pipeline_run.uuid
    logger.info("[run=%s] Pipeline started for profile=%s (%s)", run_id, profile.uuid, profile.domain)

    total_tokens = 0
    profile_dict = {
        "name": profile.name,
        "domain": profile.domain,
        "industry": profile.industry,
        "description": profile.description,
        "competitors": profile.competitors or [],
    }

    # ------------------------------------------------------------------
    # Agent 1: Query Discovery
    # ------------------------------------------------------------------
    try:
        agent1 = QueryDiscoveryAgent()
        raw_queries, tokens1 = agent1.run(profile_dict)
        total_tokens += tokens1
    except Exception as exc:
        logger.exception("[run=%s] Agent 1 failed: %s", run_id, exc)
        pipeline_run.status = "failed"
        pipeline_run.error_message = f"Agent 1 (discovery) failed: {exc}"
        pipeline_run.completed_at = datetime.now(timezone.utc)
        pipeline_run.tokens_used = total_tokens
        db.session.commit()
        return pipeline_run

    logger.info("[run=%s] Agent 1 complete: %d queries discovered", run_id, len(raw_queries))
    pipeline_run.queries_discovered = len(raw_queries)
    db.session.commit()

    # Persist initial query records (no scoring yet)
    query_records: list[DiscoveredQuery] = []
    for q in raw_queries:
        record = DiscoveredQuery(
            profile_uuid=profile.uuid,
            run_uuid=run_id,
            query_text=q["query_text"],
            commercial_intent_score=q.get("commercial_intent", 0.5),
            visibility_status="unknown",
        )
        db.session.add(record)
        query_records.append(record)
    db.session.commit()

    # ------------------------------------------------------------------
    # Agent 2: Visibility Scoring (per-query, failure-isolated)
    # ------------------------------------------------------------------
    agent2 = VisibilityScoringAgent()
    queries_scored = 0
    scored_query_dicts: list[dict] = []

    for record in query_records:
        try:
            scoring, tokens2 = agent2.run(record.query_text, profile_dict)
            total_tokens += tokens2

            domain_visible = scoring["domain_visible"]
            visibility_status = "visible" if domain_visible else "not_visible"

            opportunity = compute_opportunity_score(
                estimated_search_volume=scoring["estimated_search_volume"],
                competitive_difficulty=scoring["competitive_difficulty"],
                domain_visible=domain_visible,
                commercial_intent=record.commercial_intent_score or 0.5,
            )

            record.estimated_search_volume = scoring["estimated_search_volume"]
            record.competitive_difficulty = scoring["competitive_difficulty"]
            record.opportunity_score = opportunity
            record.domain_visible = domain_visible
            record.visibility_position = scoring.get("visibility_position")
            record.visibility_status = visibility_status
            record.last_checked_at = datetime.now(timezone.utc)

            queries_scored += 1
            scored_query_dicts.append({
                "query_uuid": record.uuid,
                "query_text": record.query_text,
                "opportunity_score": opportunity,
                "estimated_search_volume": scoring["estimated_search_volume"],
                "domain_visible": domain_visible,
            })

            logger.debug(
                "[run=%s] Scored '%s': visible=%s, score=%.3f",
                run_id, record.query_text[:50], domain_visible, opportunity,
            )

        except Exception as exc:
            logger.warning("[run=%s] Agent 2 failed for query '%s': %s", run_id, record.query_text[:50], exc)
            record.visibility_status = "unknown"
            # Continue processing remaining queries

        db.session.commit()

    pipeline_run.queries_scored = queries_scored
    logger.info("[run=%s] Agent 2 complete: %d/%d queries scored", run_id, queries_scored, len(query_records))

    # ------------------------------------------------------------------
    # Agent 3: Content Recommendations
    # ------------------------------------------------------------------
    recommendation_records: list[ContentRecommendation] = []
    try:
        agent3 = ContentRecommendationAgent()
        recs, tokens3 = agent3.run(scored_query_dicts, profile_dict)
        total_tokens += tokens3

        for rec in recs:
            r = ContentRecommendation(
                profile_uuid=profile.uuid,
                query_uuid=rec["query_uuid"],
                content_type=rec["content_type"],
                title=rec["title"],
                rationale=rec["rationale"],
                target_keywords=rec["target_keywords"],
                priority=rec["priority"],
            )
            db.session.add(r)
            recommendation_records.append(r)
        db.session.commit()

        logger.info("[run=%s] Agent 3 complete: %d recommendations generated", run_id, len(recommendation_records))

    except Exception as exc:
        logger.warning("[run=%s] Agent 3 failed (non-fatal): %s", run_id, exc)

    # ------------------------------------------------------------------
    # Finalise run
    # ------------------------------------------------------------------
    pipeline_run.status = "completed"
    pipeline_run.tokens_used = total_tokens
    pipeline_run.completed_at = datetime.now(timezone.utc)
    db.session.commit()

    logger.info(
        "[run=%s] Pipeline completed. tokens=%d, queries=%d, recs=%d",
        run_id, total_tokens, queries_scored, len(recommendation_records),
    )
    return pipeline_run


def build_run_response(run: PipelineRun, profile: BusinessProfile) -> dict:
    """
    Build the full response dict for the POST /run endpoint.
    Includes top 3 queries and all recommendations.
    """
    # Top 3 queries by opportunity score
    top_queries = (
        DiscoveredQuery.query
        .filter_by(run_uuid=run.uuid)
        .order_by(DiscoveredQuery.opportunity_score.desc())
        .limit(3)
        .all()
    )

    # All recommendations from this run
    recommendations = (
        ContentRecommendation.query
        .filter(
            ContentRecommendation.query_uuid.in_(
                db.session.query(DiscoveredQuery.uuid).filter_by(run_uuid=run.uuid)
            )
        )
        .all()
    )

    return {
        "run_uuid": run.uuid,
        "profile_uuid": run.profile_uuid,
        "status": run.status,
        "queries_discovered": run.queries_discovered,
        "queries_scored": run.queries_scored,
        "tokens_used": run.tokens_used,
        "error_message": run.error_message,
        "started_at": run.started_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "top_opportunity_queries": [q.to_dict() for q in top_queries],
        "content_recommendations": [r.to_dict() for r in recommendations],
    }
