"""
Reasoning Generator — Produces specific, data-grounded reasoning for each ranked candidate.

The JD explicitly says:
- "Plain-language reasoning that demonstrates you actually understood the candidate's profile
   will rank highly here."
- "Don't try to be impressive; try to be specific and honest."

Penalized:
- Empty reasoning
- All-identical strings
- Templated reasoning that just inserts candidate name
- Mentions skills not in the candidate's profile (hallucination)
- Reasoning that contradicts the rank

Our approach: Build reasoning from the actual data in the candidate profile,
referencing real titles, skills, experience, and signals.
"""

from src.jd_parser import (
    MUST_HAVE_SKILL_CLUSTERS,
    NICE_TO_HAVE_SKILL_CLUSTERS,
    HIGHLY_RELEVANT_TITLES,
    CONSULTING_COMPANIES,
)


def generate_reasoning(candidate, dimension_scores, composite_score, rank):
    """
    Generate a specific, honest 1-2 sentence reasoning for a candidate.

    Args:
        candidate: full candidate dict
        dimension_scores: dict of dimension -> score
        composite_score: float, final composite score
        rank: int, the candidate's rank (1-100)

    Returns:
        str: 1-2 sentence reasoning
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    parts = []

    # ── Part 1: Title and experience summary ──
    title = profile.get("current_title", "Professional")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)

    title_str = f"{title}"
    if company:
        title_str += f" at {company}"

    parts.append(f"{title_str} with {yoe:.1f} yrs experience")

    # ── Part 2: Key strength (what makes them fit) ──
    strengths = []

    # Check must-have skills matched
    all_text = _build_search_text(candidate)
    must_have_names = []
    for cluster_name, cluster in MUST_HAVE_SKILL_CLUSTERS.items():
        for kw in cluster["keywords"]:
            if kw.lower() in all_text:
                must_have_names.append(cluster["description"].split("experience with ")[-1].split(" (")[0])
                break

    if must_have_names:
        if len(must_have_names) >= 3:
            strengths.append(f"covers {len(must_have_names)}/4 must-have areas ({', '.join(must_have_names[:2])})")
        elif len(must_have_names) >= 1:
            strengths.append(f"has {', '.join(must_have_names[:2])}")

    # Check for production ML experience from career descriptions
    has_production_ml = False
    for job in career:
        desc = (job.get("description", "") or "").lower()
        if any(w in desc for w in ["production", "deployed", "shipped", "real users", "scale"]):
            if any(w in desc for w in ["ml", "model", "ranking", "embedding", "recommendation",
                                       "machine learning", "ai", "search"]):
                has_production_ml = True
                break

    if has_production_ml:
        strengths.append("production ML deployment experience")

    # Check career history for relevant roles
    relevant_roles = []
    for job in career:
        job_title = (job.get("title", "") or "").lower()
        for t in HIGHLY_RELEVANT_TITLES:
            if t in job_title:
                relevant_roles.append(job.get("title", ""))
                break

    if relevant_roles and not any("title" in s.lower() for s in strengths):
        if len(relevant_roles) > 1:
            strengths.append(f"held {len(relevant_roles)} relevant AI/ML roles")
        else:
            strengths.append(f"prior role as {relevant_roles[0]}")

    # ── Part 3: Behavioral/availability signal ──
    behavioral_notes = []

    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate >= 0.7:
        behavioral_notes.append(f"strong {response_rate:.0%} recruiter response rate")
    elif response_rate >= 0.4:
        behavioral_notes.append(f"{response_rate:.0%} response rate")
    elif response_rate < 0.15:
        behavioral_notes.append(f"low {response_rate:.0%} response rate")

    if signals.get("open_to_work_flag", False):
        behavioral_notes.append("open to work")

    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 30:
        behavioral_notes.append(f"{notice_days}d notice")

    github = signals.get("github_activity_score", -1)
    if github >= 50:
        behavioral_notes.append(f"active GitHub (score {github:.0f})")

    # ── Part 4: Any notable weakness ──
    weaknesses = []

    # Consulting-only career
    all_consulting = True
    for job in career:
        company_name = (job.get("company", "") or "").lower()
        if not any(c in company_name for c in CONSULTING_COMPANIES):
            all_consulting = False
            break
    if all_consulting and len(career) > 1:
        weaknesses.append("consulting-only background")

    # Location mismatch
    country = (profile.get("country", "") or "").lower()
    if "india" not in country and not signals.get("willing_to_relocate", False):
        weaknesses.append(f"based in {profile.get('country', 'abroad')}")

    # ── Assemble reasoning ──
    reasoning = parts[0]

    if strengths:
        reasoning += "; " + "; ".join(strengths[:2])

    if behavioral_notes:
        reasoning += "; " + "; ".join(behavioral_notes[:2])

    if weaknesses and rank > 30:
        reasoning += "; note: " + ", ".join(weaknesses[:1])

    # Truncate to reasonable length
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning


def _build_search_text(candidate):
    """Build lowercase searchable text from all candidate fields."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append((profile.get("headline", "") or "").lower())
    parts.append((profile.get("summary", "") or "").lower())
    for job in candidate.get("career_history", []):
        parts.append((job.get("title", "") or "").lower())
        parts.append((job.get("description", "") or "").lower())
    for s in candidate.get("skills", []):
        parts.append((s.get("name", "") or "").lower())
    return " ".join(parts)
