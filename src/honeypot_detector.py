"""
Honeypot Detector — Identifies candidates with subtly impossible profiles.

The dataset contains ~80 honeypot candidates designed to trap keyword-matching systems.
Indicators include:
- Impossibly many expert skills with 0 duration
- Experience years at companies that couldn't exist that long
- Career history contradictions (overlapping impossible dates)
- Skill assessment scores wildly mismatched with proficiency claims
- Title-description semantic mismatches

Submissions with >10% honeypots in top 100 are disqualified.
"""

from datetime import datetime, date


def detect_honeypot(candidate):
    """
    Returns a honeypot confidence score between 0.0 (clean) and 1.0 (certain honeypot).

    Multiple weak signals are aggregated — a single flag isn't enough,
    but several flags together strongly indicate a honeypot.
    """
    flags = []

    # ── Flag 1: Too many expert skills with zero or very low duration ──
    skills = candidate.get("skills", [])
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    zero_duration_experts = [
        s for s in expert_skills
        if s.get("duration_months", 0) == 0
    ]
    if len(zero_duration_experts) >= 3:
        flags.append(("zero_duration_experts", 0.4))
    elif len(zero_duration_experts) >= 2:
        flags.append(("zero_duration_experts", 0.2))

    # Expert in 10+ skills is suspicious
    if len(expert_skills) >= 10:
        flags.append(("too_many_expert_skills", 0.3))
    elif len(expert_skills) >= 8:
        flags.append(("too_many_expert_skills", 0.15))

    # ── Flag 2: Skills with impossibly high endorsements for low duration ──
    for s in skills:
        dur = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)
        if dur == 0 and endorsements > 20:
            flags.append(("impossible_endorsements", 0.2))
            break

    # ── Flag 3: Career history date impossibilities ──
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    total_exp = profile.get("years_of_experience", 0)

    # Check for impossibly long tenure relative to total experience
    for job in career:
        dur_months = job.get("duration_months", 0)
        if dur_months > (total_exp + 2) * 12:  # tenure > total experience + buffer
            flags.append(("impossible_tenure", 0.35))
            break

    # Check for career history total exceeding stated experience by too much
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    stated_months = total_exp * 12
    if stated_months > 0 and total_career_months > stated_months * 2.5:
        flags.append(("career_exceeds_experience", 0.25))

    # ── Flag 4: Title-description mismatch ──
    # E.g., current_title is "Marketing Manager" but description talks about ML systems
    for job in career:
        title = (job.get("title", "") or "").lower()
        desc = (job.get("description", "") or "").lower()

        # Non-tech title with tech description
        non_tech_titles = [
            "marketing", "sales", "accountant", "hr manager",
            "content writer", "graphic designer", "civil engineer",
            "mechanical engineer", "operations manager",
        ]
        tech_desc_signals = [
            "machine learning", "deep learning", "neural network",
            "pytorch", "tensorflow", "deployed model", "embedding",
            "ranking system", "ml pipeline",
        ]

        is_non_tech_title = any(nt in title for nt in non_tech_titles)
        tech_desc_count = sum(1 for ts in tech_desc_signals if ts in desc)

        if is_non_tech_title and tech_desc_count >= 3:
            flags.append(("title_desc_mismatch", 0.15))
            break

    # ── Flag 5: Skill assessment scores contradict proficiency ──
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    for skill in skills:
        skill_name = skill.get("name", "")
        proficiency = skill.get("proficiency", "")
        if skill_name in assessments:
            score = assessments[skill_name]
            # Expert proficiency but very low assessment score
            if proficiency == "expert" and score < 20:
                flags.append(("assessment_proficiency_mismatch", 0.2))
                break
            # Beginner proficiency but very high assessment score (less suspicious but notable)

    # ── Flag 6: Profile completeness contradictions ──
    completeness = signals.get("profile_completeness_score", 50)
    # Very high completeness but missing many fields
    education = candidate.get("education", [])
    certs = candidate.get("certifications", [])
    if completeness > 95 and len(education) == 0 and len(skills) < 3 and len(certs) == 0:
        flags.append(("completeness_contradiction", 0.15))

    # ── Flag 7: Impossibly high activity with old last_active_date ──
    last_active = signals.get("last_active_date", "")
    saved_30d = signals.get("saved_by_recruiters_30d", 0)
    search_30d = signals.get("search_appearance_30d", 0)
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_inactive = (date(2026, 6, 15) - last_dt).days  # approximate "now"
            if days_inactive > 180 and (saved_30d > 20 or search_30d > 200):
                flags.append(("activity_date_mismatch", 0.2))
        except (ValueError, TypeError):
            pass

    # ── Flag 8: Experience at a company for longer than company could exist ──
    # This is hard to check without a company database, but we can check for
    # very long tenures at small/unknown companies
    for job in career:
        dur_months = job.get("duration_months", 0)
        company_size = job.get("company_size", "")
        if company_size in ("1-10", "11-50") and dur_months > 120:
            flags.append(("long_tenure_tiny_company", 0.15))
            break

    # ── Aggregate flags ──
    if not flags:
        return 0.0

    # Combine scores with diminishing returns
    sorted_scores = sorted([score for _, score in flags], reverse=True)
    combined = sorted_scores[0]
    for i, score in enumerate(sorted_scores[1:], 1):
        combined += score * (0.7 ** i)  # diminishing weight for additional flags

    return min(combined, 1.0)


def is_likely_honeypot(candidate, threshold=0.5):
    """Quick boolean check for honeypot status."""
    return detect_honeypot(candidate) >= threshold
