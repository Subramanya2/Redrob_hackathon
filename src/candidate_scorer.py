"""
Candidate Scorer — Multi-dimensional scoring engine.

Scores each candidate across 7 dimensions:
1. Career Relevance (0.30) — title, industry, product vs consulting, progression
2. Skill Match (0.25) — must-have/nice-to-have overlap, proficiency, assessments
3. Semantic Profile Match (0.15) — TF-IDF + BM25 text similarity
4. Experience Fit (0.10) — years in band, AI/ML depth
5. Behavioral Signals (0.10) — activity, response rate, interview completion
6. Availability & Logistics (0.05) — location, notice period, salary, work mode
7. Credibility Signals (0.05) — GitHub, profile completeness, verifications
"""

import re
from datetime import datetime, date

from src.jd_parser import (
    MUST_HAVE_SKILL_CLUSTERS,
    NICE_TO_HAVE_SKILL_CLUSTERS,
    AI_ML_CORE_SKILLS,
    HIGHLY_RELEVANT_TITLES,
    MODERATELY_RELEVANT_TITLES,
    IRRELEVANT_TITLES,
    HIGHLY_RELEVANT_INDUSTRIES,
    CONSULTING_COMPANIES,
    EXPERIENCE_RANGE,
    PREFERRED_LOCATIONS,
    PREFERRED_COUNTRIES,
)


# ─────────────────────────────────────────────────────────────
# Dimension 1: Career Relevance (weight: 0.30)
# ─────────────────────────────────────────────────────────────
def score_career_relevance(candidate):
    """
    Analyze career trajectory for genuine AI/ML engineering fit.

    Key signals:
    - Current/recent title relevance
    - Career history in product companies vs consulting
    - Industry alignment
    - Job hopping pattern (anti-signal per JD)
    - Career descriptions mentioning production ML work
    """
    score = 0.0
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # ── Current title relevance ──
    current_title = (profile.get("current_title", "") or "").lower()

    title_score = 0.0
    for title in HIGHLY_RELEVANT_TITLES:
        if title in current_title:
            title_score = 1.0
            break
    if title_score == 0:
        for title in MODERATELY_RELEVANT_TITLES:
            if title in current_title:
                title_score = 0.4
                break
    if title_score == 0:
        for title in IRRELEVANT_TITLES:
            if title in current_title:
                title_score = 0.0
                break
        if title_score == 0:
            title_score = 0.2  # unknown title, give small benefit of doubt

    score += title_score * 0.25

    # ── Career history title relevance (check ALL roles, not just current) ──
    career_title_scores = []
    ai_role_months = 0
    for job in career:
        job_title = (job.get("title", "") or "").lower()
        job_score = 0.0
        for title in HIGHLY_RELEVANT_TITLES:
            if title in job_title:
                job_score = 1.0
                break
        if job_score == 0:
            for title in MODERATELY_RELEVANT_TITLES:
                if title in job_title:
                    job_score = 0.4
                    break

        career_title_scores.append(job_score)
        if job_score >= 0.4:
            ai_role_months += job.get("duration_months", 0)

    if career_title_scores:
        avg_career_title = sum(career_title_scores) / len(career_title_scores)
        max_career_title = max(career_title_scores)
        # Weighted: best role matters more than average
        career_title_combined = 0.6 * max_career_title + 0.4 * avg_career_title
        score += career_title_combined * 0.20

    # ── AI/ML role depth (months in relevant roles) ──
    # JD wants 4-5 years in applied ML/AI roles
    ai_years = ai_role_months / 12.0
    if ai_years >= 4:
        score += 0.15
    elif ai_years >= 2:
        score += 0.10
    elif ai_years >= 1:
        score += 0.05

    # ── Product company vs consulting ──
    # JD explicitly says consulting-only career is a disqualifier
    current_company = (profile.get("current_company", "") or "").lower()
    current_industry = (profile.get("current_industry", "") or "").lower()

    has_product_experience = False
    all_consulting = True

    for job in career:
        company = (job.get("company", "") or "").lower()
        industry = (job.get("industry", "") or "").lower()

        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        is_consulting = is_consulting or "consulting" in industry.lower()
        is_consulting = is_consulting or industry in ["it services", "outsourcing"]

        if not is_consulting:
            all_consulting = False
            # Check if it's a product company
            if any(ind in industry for ind in HIGHLY_RELEVANT_INDUSTRIES):
                has_product_experience = True

    if all_consulting and len(career) > 1:
        score -= 0.10  # JD explicitly penalizes consulting-only careers
    elif has_product_experience:
        score += 0.10

    # ── Career descriptions — production ML signals ──
    production_signals = [
        "production", "deployed", "shipped", "launched", "scale",
        "real users", "a/b test", "monitoring", "pipeline",
        "latency", "throughput", "sla", "uptime", "incident",
        "on-call", "model serving", "inference", "endpoint",
    ]
    ml_production_count = 0
    for job in career:
        desc = (job.get("description", "") or "").lower()
        ml_words = sum(1 for s in production_signals if s in desc)
        if ml_words >= 2:
            ml_production_count += 1

    if ml_production_count >= 2:
        score += 0.10
    elif ml_production_count >= 1:
        score += 0.05

    # ── Job hopping penalty ──
    # JD: "Title-chasers... switching every 1.5 years, we're not a fit"
    if len(career) >= 3:
        durations = [j.get("duration_months", 0) for j in career]
        avg_tenure = sum(durations) / len(durations) if durations else 0
        if avg_tenure < 15:  # < 1.25 years average
            score -= 0.05
        elif avg_tenure < 18:  # < 1.5 years average
            score -= 0.02

    # ── Industry relevance ──
    for job in career:
        industry = (job.get("industry", "") or "").lower()
        if any(ind in industry for ind in HIGHLY_RELEVANT_INDUSTRIES):
            score += 0.05
            break

    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────
# Dimension 2: Skill Match (weight: 0.25)
# ─────────────────────────────────────────────────────────────
def score_skill_match(candidate):
    """
    Match candidate skills against JD requirements.

    Goes beyond keyword matching:
    - Checks skill proficiency levels
    - Validates skill duration (0-month experts are suspicious)
    - Weights must-have vs nice-to-have
    - Uses skill assessment scores from Redrob platform
    """
    score = 0.0
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})

    # Build searchable text from skills and career
    skill_text = " ".join([
        (s.get("name", "") or "").lower() for s in skills
    ])
    career_text = " ".join([
        ((j.get("description", "") or "") + " " + (j.get("title", "") or "")).lower()
        for j in candidate.get("career_history", [])
    ])
    summary_text = (candidate.get("profile", {}).get("summary", "") or "").lower()
    all_text = f"{skill_text} {career_text} {summary_text}"

    # ── Must-have skills ──
    must_have_matched = 0
    must_have_total = len(MUST_HAVE_SKILL_CLUSTERS)
    must_have_weighted = 0.0

    for cluster_name, cluster in MUST_HAVE_SKILL_CLUSTERS.items():
        matched = False
        match_quality = 0.0

        for keyword in cluster["keywords"]:
            kw_lower = keyword.lower()
            if kw_lower in all_text:
                matched = True
                # Check if it's in formal skills list with proficiency
                for s in skills:
                    if kw_lower in (s.get("name", "") or "").lower():
                        prof = s.get("proficiency", "beginner")
                        dur = s.get("duration_months", 0)
                        prof_score = {
                            "expert": 1.0, "advanced": 0.8,
                            "intermediate": 0.5, "beginner": 0.2
                        }.get(prof, 0.3)
                        # Penalize zero-duration skills
                        dur_factor = min(1.0, dur / 24) if dur > 0 else 0.3
                        match_quality = max(match_quality, prof_score * dur_factor)

                        # Check assessment score
                        if s["name"] in assessments:
                            assess_score = assessments[s["name"]] / 100.0
                            match_quality = max(match_quality, assess_score)
                        break

                if match_quality == 0:
                    # Found in text but not in skills list — still counts, less weight
                    match_quality = 0.4
                break

        if matched:
            must_have_matched += 1
            must_have_weighted += match_quality * cluster["weight"]

    # Normalize must-have score
    max_must_have = sum(c["weight"] for c in MUST_HAVE_SKILL_CLUSTERS.values())
    if max_must_have > 0:
        must_have_score = must_have_weighted / max_must_have
    else:
        must_have_score = 0.0

    score += must_have_score * 0.55

    # ── Nice-to-have skills ──
    nice_to_have_weighted = 0.0
    for cluster_name, cluster in NICE_TO_HAVE_SKILL_CLUSTERS.items():
        for keyword in cluster["keywords"]:
            if keyword.lower() in all_text:
                nice_to_have_weighted += cluster["weight"]
                break

    max_nice = sum(c["weight"] for c in NICE_TO_HAVE_SKILL_CLUSTERS.values())
    if max_nice > 0:
        nice_score = nice_to_have_weighted / max_nice
    else:
        nice_score = 0.0

    score += nice_score * 0.25

    # ── General AI/ML skill coverage ──
    ai_keywords = AI_ML_CORE_SKILLS["keywords"]
    ai_matches = sum(1 for kw in ai_keywords if kw.lower() in all_text)
    ai_coverage = min(1.0, ai_matches / 8)  # 8+ matches = full score
    score += ai_coverage * 0.20

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────
# Dimension 3: Experience Fit (weight: 0.10)
# ─────────────────────────────────────────────────────────────
def score_experience_fit(candidate):
    """Score how well experience level matches the 5-9 year range."""
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)

    ideal_min = EXPERIENCE_RANGE["ideal_min"]
    ideal_max = EXPERIENCE_RANGE["ideal_max"]
    accept_min = EXPERIENCE_RANGE["acceptable_min"]
    accept_max = EXPERIENCE_RANGE["acceptable_max"]

    if ideal_min <= yoe <= ideal_max:
        # Sweet spot
        score = 1.0
    elif accept_min <= yoe < ideal_min:
        # Below ideal but acceptable
        score = 0.5 + 0.5 * (yoe - accept_min) / (ideal_min - accept_min)
    elif ideal_max < yoe <= accept_max:
        # Above ideal but acceptable
        score = 0.5 + 0.5 * (accept_max - yoe) / (accept_max - ideal_max)
    elif yoe < accept_min:
        score = max(0.1, yoe / accept_min * 0.5)
    else:
        score = max(0.1, 0.5 * (20 - yoe) / (20 - accept_max)) if yoe < 20 else 0.1

    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────
# Dimension 4: Behavioral Signals (weight: 0.10)
# ─────────────────────────────────────────────────────────────
def score_behavioral_signals(candidate):
    """
    Score platform engagement and availability signals.

    These are multipliers — a great candidate who's inactive
    is less hireable than a good candidate who's responsive.
    """
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # ── Recency: last active date ──
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_ago = (date(2026, 6, 15) - last_dt).days
            if days_ago <= 7:
                score += 0.20
            elif days_ago <= 30:
                score += 0.15
            elif days_ago <= 90:
                score += 0.10
            elif days_ago <= 180:
                score += 0.05
            # > 180 days: no bonus
        except (ValueError, TypeError):
            pass

    # ── Open to work ──
    if signals.get("open_to_work_flag", False):
        score += 0.15

    # ── Recruiter response rate ──
    response_rate = signals.get("recruiter_response_rate", 0)
    score += response_rate * 0.20

    # ── Response time (lower is better) ──
    avg_response_hours = signals.get("avg_response_time_hours", 999)
    if avg_response_hours <= 4:
        score += 0.10
    elif avg_response_hours <= 24:
        score += 0.07
    elif avg_response_hours <= 72:
        score += 0.03

    # ── Interview completion rate ──
    interview_rate = signals.get("interview_completion_rate", 0)
    score += interview_rate * 0.15

    # ── Offer acceptance rate ──
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate >= 0:
        score += offer_rate * 0.10

    # ── Saved by recruiters (social proof) ──
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 10:
        score += 0.05
    elif saved >= 5:
        score += 0.03

    # ── Applications submitted (shows active intent) ──
    apps = signals.get("applications_submitted_30d", 0)
    if 1 <= apps <= 10:
        score += 0.05  # Active but not desperate
    elif apps > 10:
        score += 0.02  # Might be spraying

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────
# Dimension 5: Availability & Logistics (weight: 0.05)
# ─────────────────────────────────────────────────────────────
def score_availability(candidate):
    """Score location, notice period, salary, and work mode fit."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # ── Location ──
    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()

    if any(c in country for c in PREFERRED_COUNTRIES):
        score += 0.15
        if any(loc in location for loc in PREFERRED_LOCATIONS):
            score += 0.15
    elif signals.get("willing_to_relocate", False):
        score += 0.10

    # ── Notice period ──
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 30:
        score += 0.25
    elif notice_days <= 60:
        score += 0.15
    elif notice_days <= 90:
        score += 0.05

    # ── Work mode ──
    work_mode = signals.get("preferred_work_mode", "")
    if work_mode in ("hybrid", "flexible"):
        score += 0.15
    elif work_mode == "onsite":
        score += 0.10
    elif work_mode == "remote":
        score += 0.05

    # ── Willingness to relocate ──
    if signals.get("willing_to_relocate", False):
        score += 0.10

    # ── Salary range reasonableness (for senior AI engineer in India) ──
    salary = signals.get("expected_salary_range_inr_lpa", {})
    min_salary = salary.get("min", 0)
    max_salary = salary.get("max", 0)
    # Reasonable range for this role: 25-60 LPA
    if 15 <= min_salary <= 60 and max_salary <= 80:
        score += 0.10
    elif min_salary > 80:
        score += 0.02  # Possibly out of budget

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────
# Dimension 6: Credibility Signals (weight: 0.05)
# ─────────────────────────────────────────────────────────────
def score_credibility(candidate):
    """Score GitHub activity, profile completeness, education, verifications."""
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])
    score = 0.0

    # ── GitHub activity ──
    github_score = signals.get("github_activity_score", -1)
    if github_score >= 50:
        score += 0.25
    elif github_score >= 20:
        score += 0.15
    elif github_score >= 1:
        score += 0.05
    # -1 means no GitHub linked — neutral, not penalized

    # ── Profile completeness ──
    completeness = signals.get("profile_completeness_score", 0)
    score += (completeness / 100) * 0.20

    # ── Verifications ──
    if signals.get("verified_email", False):
        score += 0.10
    if signals.get("verified_phone", False):
        score += 0.05
    if signals.get("linkedin_connected", False):
        score += 0.10

    # ── Education tier ──
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = (edu.get("field_of_study", "") or "").lower()

        # Relevant field of study
        relevant_fields = [
            "computer science", "cs", "information technology", "it",
            "data science", "artificial intelligence", "machine learning",
            "mathematics", "statistics", "electrical engineering",
            "electronics", "ece",
        ]
        field_relevant = any(f in field for f in relevant_fields)

        if tier == "tier_1":
            score += 0.20
        elif tier == "tier_2":
            score += 0.10
        elif tier == "tier_3" and field_relevant:
            score += 0.05

        if field_relevant:
            score += 0.05

        break  # Only consider first/primary education

    # ── Connection count (networking signal) ──
    connections = signals.get("connection_count", 0)
    if connections >= 500:
        score += 0.05
    elif connections >= 200:
        score += 0.03

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────
# Composite scorer
# ─────────────────────────────────────────────────────────────
DIMENSION_WEIGHTS = {
    "career_relevance": 0.30,
    "skill_match": 0.25,
    "semantic_match": 0.15,  # Filled externally from TextScorer
    "experience_fit": 0.10,
    "behavioral_signals": 0.10,
    "availability": 0.05,
    "credibility": 0.05,
}


def compute_structured_scores(candidate):
    """
    Compute all non-text-based dimension scores for a candidate.

    Returns dict of dimension -> score (0 to 1).
    The 'semantic_match' dimension is computed externally by TextScorer
    and merged later.
    """
    return {
        "career_relevance": score_career_relevance(candidate),
        "skill_match": score_skill_match(candidate),
        "experience_fit": score_experience_fit(candidate),
        "behavioral_signals": score_behavioral_signals(candidate),
        "availability": score_availability(candidate),
        "credibility": score_credibility(candidate),
    }


def compute_composite_score(dimension_scores):
    """
    Compute weighted composite score from dimension scores.

    Args:
        dimension_scores: dict with keys matching DIMENSION_WEIGHTS

    Returns:
        float in [0, 1]
    """
    total = 0.0
    for dim, weight in DIMENSION_WEIGHTS.items():
        total += dimension_scores.get(dim, 0.0) * weight
    return total
