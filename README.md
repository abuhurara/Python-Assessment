# AI Visibility Intelligence API

A RESTful Flask API that discovers high-value queries in a business's competitive
AI-search space, scores them for opportunity, and generates actionable content
recommendations using a three-agent LLM pipeline.

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
git clone <repo-url>
cd ai_visibility_api

# Copy and fill in your keys
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required) and optionally DATAFORSEO_LOGIN/PASSWORD

docker-compose up --build
```

The API is now available at `http://localhost:5000`.

### Option B — Local Python

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY

# Initialise the database
flask db upgrade        # runs existing migrations
# OR on first run:
# flask db init && flask db migrate -m "initial schema" && flask db upgrade

python run.py
```

### Run Tests

```bash
ANTHROPIC_API_KEY=test DATABASE_URL=sqlite:///test.db pytest -v
```

All 19 tests pass with zero real API calls (LLM responses are mocked).

---

## API Reference

### Base URL: `http://localhost:5000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/profiles` | Register a new business profile |
| GET | `/profiles/{uuid}` | Get profile + summary stats |
| POST | `/profiles/{uuid}/run` | Trigger the full 3-agent pipeline |
| GET | `/profiles/{uuid}/queries` | List discovered queries (paginated, filterable) |
| GET | `/profiles/{uuid}/recommendations` | List content recommendations |
| POST | `/queries/{uuid}/recheck` | Re-run Agent 2 on a single query |

#### Example: Create a Profile

```bash
curl -X POST http://localhost:5000/api/v1/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Frase",
    "domain": "frase.io",
    "industry": "SEO Content Tools",
    "description": "AI-powered content briefs and SEO research",
    "competitors": ["surferseo.com", "marketmuse.com", "clearscope.io"]
  }'
```

#### Example: Run the Pipeline

```bash
curl -X POST http://localhost:5000/api/v1/profiles/{profile_uuid}/run
```

Pipeline runs synchronously (10–30 seconds). Returns discovered queries, scores,
and content recommendations in a single response.

#### Example: Filter Queries

```bash
# High-opportunity queries only
GET /api/v1/profiles/{uuid}/queries?min_score=0.7

# Queries where domain is not appearing
GET /api/v1/profiles/{uuid}/queries?status=not_visible

# Paginated
GET /api/v1/profiles/{uuid}/queries?page=2&per_page=10
```

### Error Format

All errors use a consistent envelope:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Profile 'abc-123' not found."
  }
}
```

---

## Architecture

```
app/
├── __init__.py          ← create_app() factory, extension setup
├── api/
│   ├── profiles.py      ← Blueprint: profiles + pipeline + queries + recommendations
│   └── queries.py       ← Blueprint: recheck endpoint
├── agents/
│   ├── base.py          ← Anthropic client, JSON parsing, fallback handling
│   ├── discovery.py     ← Agent 1: QueryDiscoveryAgent
│   ├── scoring.py       ← Agent 2: VisibilityScoringAgent + DataForSEO integration
│   └── recommendation.py← Agent 3: ContentRecommendationAgent
├── models/
│   ├── profile.py       ← BusinessProfile
│   ├── pipeline_run.py  ← PipelineRun
│   ├── query.py         ← DiscoveredQuery
│   └── recommendation.py← ContentRecommendation
├── services/
│   └── pipeline.py      ← Orchestrator: coordinates all three agents
└── utils/
    ├── scoring.py        ← Opportunity score formula
    └── errors.py         ← Consistent error responses
```

---

## Agent Design

### Model Selection

All three agents use **Claude Sonnet 4** (`claude-sonnet-4-20250514`).

**Rationale:** This assessment requires structured JSON output from a synchronous
pipeline. Sonnet 4 offers the best balance of:
- Speed (faster than Opus, important for a 10–30s pipeline)
- Cost (lower token cost — relevant at scale)
- JSON reliability (comparable to Opus for structured output tasks)
- Instruction following (accurate schema adherence)

Opus would be justified for genuinely complex reasoning tasks (e.g. synthesising
multi-source research reports). For JSON generation with a well-defined schema,
Sonnet 4 is the right choice.

### Agent 1 — QueryDiscoveryAgent

**Prompt strategy:** A "SEO research specialist" persona focuses the model on
search intent. The full output schema is specified in the system prompt so the
model never has to guess the format. Temperature is left at default (1.0) to
maximise diversity in the discovered queries — determinism would be counterproductive here.

**Failure handling:** Empty `query_text` filtered; `commercial_intent` clamped to [0, 1].
If JSON is completely malformed, returns empty list (pipeline continues, run gets 0 queries).

### Agent 2 — VisibilityScoringAgent

**Prompt strategy:** An "AI SERP analyst" persona improves the model's reasoning
about what AI assistants surface (vs. traditional blue-link rankings). The
`reasoning` field in the schema forces chain-of-thought — the model must justify its
visibility decision, which improves accuracy and auditability.

**Real data integration:** DataForSEO Keywords Data API is called first. If credentials
are not configured or the call fails, the system degrades gracefully to LLM estimates.
`volume_source` in the response indicates which was used.

**Per-query isolation:** Each query is scored in a separate LLM call. More tokens than
batching, but a malformed response for query N won't corrupt query N+1.

### Agent 3 — ContentRecommendationAgent

**Prompt strategy:** An "AI content strategist" persona emphasises AI-visibility tactics
(structured data, Q&A format, authoritative citations) over generic SEO advice.
The prompt is explicit about priority assignment rules to prevent everything being "high".

**Batching:** Top 5 non-visible queries are sent in a single call, allowing the model
to avoid redundant recommendations across similar queries. This is the one agent where
batching is intentional.

**Validation:** Recommendations with hallucinated query UUIDs are filtered out. Invalid
`content_type` and `priority` values are normalised to safe defaults.

---

## Opportunity Score Formula

```
opportunity_score = (
    volume_score  × 0.35
  + gap_score     × 0.30
  + intent_score  × 0.20
  + ease_score    × 0.15
)
```

| Component | Weight | Formula |
|-----------|--------|---------|
| `volume_score` | 35% | `log10(1 + volume) / log10(1 + 100_000)` — log scale, capped at 1.0 |
| `gap_score` | 30% | `1.0` if not visible, `0.2` if already visible |
| `intent_score` | 20% | Commercial intent from Agent 1 (0.0–1.0) |
| `ease_score` | 15% | `1 - (competitive_difficulty / 100)` |

**Reasoning:**

- **Volume (35%)** is the largest factor: traffic potential is the foundation of opportunity.
  Log scale prevents 100k-search queries from completely drowning out 10k ones — both are
  high-value.

- **Gap (30%)** is second: a query where you're absent is more actionable than one where
  you already appear. If you're visible, the score is not zero (there's still room to improve
  position), but it's significantly discounted.

- **Intent (20%):** A high-volume informational query ("what is SEO") may generate traffic
  but won't convert. Commercial intent shapes actual business value.

- **Ease (15%):** Difficulty matters, but a hard, high-volume query where you're absent is
  still more valuable than an easy, low-volume one.

---

## Database Schema

Four models with UUID primary keys throughout (no sequential ID leakage in URLs).

**BusinessProfile** → **PipelineRun** (one-to-many)  
**BusinessProfile** → **DiscoveredQuery** (one-to-many)  
**DiscoveredQuery** → **ContentRecommendation** (one-to-many)  
**PipelineRun** → **DiscoveredQuery** (one-to-many, via `run_uuid` FK — enables per-run query tracing)

`competitors` is stored as a JSON array (not a join table) because it's a simple list of
strings that's always read and written together — a separate table would add complexity
with no benefit.

`target_keywords` on ContentRecommendation is similarly a JSON array: keywords are always
retrieved with the recommendation, never queried independently.

`commercial_intent_score` is stored on DiscoveredQuery (from Agent 1) so the opportunity
score can be recomputed after a recheck without re-calling Agent 1.

---

## Tradeoffs

**Synchronous pipeline:** The assessment spec says synchronous is fine. In production
this would be a background task (Celery + Redis) with a polling endpoint. The pipeline
takes 15–40 seconds depending on query count and DataForSEO calls.

**SQLite default:** Works out of the box. Set `DATABASE_URL=postgresql://...` for
production. All queries use SQLAlchemy ORM so the switch is transparent.

**Per-query scoring (Agent 2):** Trades token efficiency for failure isolation. With
15 queries and DataForSEO calls, Agent 2 makes 15 LLM calls and up to 15 API calls.
Batching 3–5 queries per call would reduce this 3–5×, at the cost of harder failure
isolation and response parsing.

**No auth layer:** Out of scope per the assessment. In production: API key auth with
per-key rate limiting on the pipeline trigger endpoint.

---

## AI Tools Used

- **Claude (claude.ai):** Used for architectural review and README drafting. All code
  was written and reviewed manually — Claude was used as a sounding board, not a
  code generator. Prompt engineering and agent design decisions are my own.

---

## Environment Variables

See `.env.example` for the full list. Required:

- `ANTHROPIC_API_KEY` — Anthropic API key
- `DATABASE_URL` — SQLAlchemy connection string (default: `sqlite:///dev.db`)
- `SECRET_KEY` — Flask secret key

Optional:
- `DATAFORSEO_LOGIN` / `DATAFORSEO_PASSWORD` — Enables real search volume data
- `OPENAI_API_KEY` — Not used; included in `.env.example` for reference
