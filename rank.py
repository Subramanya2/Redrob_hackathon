"""
rank.py — Main entry point for the Intelligent Candidate Discovery & Ranking System.

Usage:
    py rank.py --candidates ./candidates.jsonl --out ./submission.csv

Architecture:
    1. Load 100K candidates from JSONL
    2. Fast pre-filter: eliminate obviously irrelevant candidates (→ ~15-20K)
    3. Compute structured scores (career, skills, experience, behavioral, logistics)
    4. Compute semantic scores (TF-IDF + BM25 on text)
    5. Detect and penalize honeypots
    6. Merge scores, rank, select top 100
    7. Generate reasoning for top 100
    8. Output CSV

Constraints: 5 min, 16 GB RAM, CPU only, no network calls.
"""

import argparse
import json
import csv
import sys
import time
import os
import io

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

from src.jd_parser import (
    HIGHLY_RELEVANT_TITLES,
    MODERATELY_RELEVANT_TITLES,
    IRRELEVANT_TITLES,
    AI_ML_CORE_SKILLS,
    MUST_HAVE_SKILL_CLUSTERS,
    NICE_TO_HAVE_SKILL_CLUSTERS,
)
from src.candidate_scorer import (
    compute_structured_scores,
    compute_composite_score,
    DIMENSION_WEIGHTS,
)
from src.text_scorer import TextScorer, build_candidate_text
from src.honeypot_detector import detect_honeypot
from src.reasoning_generator import generate_reasoning


def load_candidates(filepath):
    """Load candidates from JSONL file."""
    candidates = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def fast_prefilter(candidates):
    """
    Quick pre-filter to reduce 100K to a manageable pool (~15-20K).

    This is NOT about ranking quality — it's about computational feasibility.
    We eliminate candidates who are clearly not fits (accountants, nurses, etc.)
    while keeping any candidate who might plausibly be relevant.

    The filter is deliberately generous — we'd rather score 20K candidates
    than miss a hidden gem.
    """
    # Build a set of all AI/ML related keywords for fast matching
    ai_keywords = set()
    for kw in AI_ML_CORE_SKILLS["keywords"]:
        ai_keywords.add(kw.lower())
    for cluster in MUST_HAVE_SKILL_CLUSTERS.values():
        for kw in cluster["keywords"]:
            ai_keywords.add(kw.lower())
    for cluster in NICE_TO_HAVE_SKILL_CLUSTERS.values():
        for kw in cluster["keywords"]:
            ai_keywords.add(kw.lower())

    # Note: we don't include generic tech keywords ("software", "data", "engineer")
    # in the pre-filter to keep it focused on AI/ML candidates

    relevant_title_words = set()
    for t in HIGHLY_RELEVANT_TITLES + MODERATELY_RELEVANT_TITLES:
        relevant_title_words.update(t.lower().split())

    passed = []
    for cand in candidates:
        profile = cand.get("profile", {})
        current_title = (profile.get("current_title", "") or "").lower()

        # ── Pass 1: Title-based quick pass ──
        # If current title contains any relevant words, keep
        title_relevant = any(w in current_title for w in relevant_title_words)
        if title_relevant:
            passed.append(cand)
            continue

        # ── Pass 2: Check if title is clearly irrelevant ──
        clearly_irrelevant = False
        for irr_title in IRRELEVANT_TITLES:
            if irr_title in current_title:
                clearly_irrelevant = True
                break

        # Even if title is irrelevant, check skills and text for AI/ML signals
        # (The JD warns about plain-language Tier 5 candidates)
        if clearly_irrelevant:
            # Check skills for AI/ML keywords
            skill_names = " ".join(
                (s.get("name", "") or "").lower()
                for s in cand.get("skills", [])
            )
            career_text = " ".join(
                ((j.get("title", "") or "") + " " + (j.get("description", "") or "")).lower()
                for j in cand.get("career_history", [])
            )
            summary = (profile.get("summary", "") or "").lower()
            combined = f"{skill_names} {career_text} {summary}"

            # Count AI/ML keyword matches in their full profile
            ai_matches = sum(1 for kw in ai_keywords if kw in combined)

            if ai_matches >= 3:
                # Has enough AI/ML signal despite irrelevant title — keep for scoring
                passed.append(cand)
                continue

            # Also check career history titles
            for job in cand.get("career_history", []):
                job_title = (job.get("title", "") or "").lower()
                if any(w in job_title for w in relevant_title_words):
                    passed.append(cand)
                    break
            continue

        # ── Pass 3: Unknown title — check for tech/AI signals ──
        headline = (profile.get("headline", "") or "").lower()
        summary = (profile.get("summary", "") or "").lower()
        combined_quick = f"{current_title} {headline} {summary}"

        # Require AI-specific keywords, not just general tech keywords
        ai_hits = sum(1 for kw in ai_keywords if kw in combined_quick)
        if ai_hits >= 2:
            passed.append(cand)
            continue

        # Check skills for AI-specific keywords only (not general tech)
        skill_names = " ".join(
            (s.get("name", "") or "").lower()
            for s in cand.get("skills", [])
        )
        skill_hits = sum(1 for kw in ai_keywords if kw in skill_names)
        if skill_hits >= 2:
            passed.append(cand)

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Candidate Discovery & Ranking System"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to candidates.jsonl file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission.csv",
        help="Output CSV file path (default: submission.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to output (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    args = parser.parse_args()

    start_time = time.time()

    # ── Step 1: Load candidates ──
    print(f"[1/7] Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"       Loaded {len(candidates)} candidates in {time.time() - start_time:.1f}s")

    # ── Step 2: Pre-filter ──
    step_time = time.time()
    print(f"[2/7] Pre-filtering candidates...")
    filtered = fast_prefilter(candidates)
    print(f"       {len(candidates)} -> {len(filtered)} candidates in {time.time() - step_time:.1f}s")

    # ── Step 3: Compute structured scores ──
    step_time = time.time()
    print(f"[3/7] Computing structured scores for {len(filtered)} candidates...")
    structured_scores = []
    for i, cand in enumerate(filtered):
        scores = compute_structured_scores(cand)
        structured_scores.append(scores)
        if args.verbose and (i + 1) % 5000 == 0:
            print(f"       Scored {i + 1}/{len(filtered)}...")
    print(f"       Done in {time.time() - step_time:.1f}s")

    # ── Step 4: Compute semantic text scores ──
    step_time = time.time()
    print(f"[4/7] Computing semantic text scores...")
    text_scorer = TextScorer()
    candidate_texts = [build_candidate_text(cand) for cand in filtered]
    semantic_scores = text_scorer.fit_and_score(candidate_texts)
    print(f"       Done in {time.time() - step_time:.1f}s")

    # ── Step 5: Detect honeypots ──
    step_time = time.time()
    print(f"[5/7] Detecting honeypots...")
    honeypot_scores = []
    honeypot_count = 0
    for cand in filtered:
        hp_score = detect_honeypot(cand)
        honeypot_scores.append(hp_score)
        if hp_score >= 0.5:
            honeypot_count += 1
    print(f"       Found {honeypot_count} likely honeypots in {time.time() - step_time:.1f}s")

    # ── Step 6: Compute composite scores and rank ──
    step_time = time.time()
    print(f"[6/7] Computing composite scores and ranking...")
    results = []
    for i, cand in enumerate(filtered):
        dim_scores = structured_scores[i].copy()
        dim_scores["semantic_match"] = float(semantic_scores[i])

        composite = compute_composite_score(dim_scores)

        # Apply honeypot penalty
        hp = honeypot_scores[i]
        if hp >= 0.5:
            composite *= (1.0 - hp * 0.9)  # Heavy penalty for likely honeypots
        elif hp >= 0.3:
            composite *= (1.0 - hp * 0.5)  # Moderate penalty for suspicious profiles

        results.append({
            "candidate": cand,
            "dimension_scores": dim_scores,
            "composite_score": composite,
            "honeypot_score": hp,
        })

    # Sort by composite score descending
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # Take top N
    top_n = results[:args.top_n]
    print(f"       Done in {time.time() - step_time:.1f}s")

    # ── Step 7: Generate reasoning and write CSV ──
    step_time = time.time()
    print(f"[7/7] Generating reasoning and writing {args.out}...")

    with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        # Ensure all scores are unique by adding tiny rank-based tiebreaker
        # (spec penalizes identical scores)
        seen_scores = set()
        for rank_idx, result in enumerate(top_n):
            rank = rank_idx + 1
            cand = result["candidate"]
            score_raw = result["composite_score"]
            score_4dp = round(score_raw, 4)
            # Ensure uniqueness at 4 decimal places
            while score_4dp in seen_scores:
                score_raw -= 0.00005
                score_4dp = round(score_raw, 4)
            seen_scores.add(score_4dp)
            reasoning = generate_reasoning(
                cand,
                result["dimension_scores"],
                result["composite_score"],
                rank,
            )
            writer.writerow([
                cand["candidate_id"],
                rank,
                f"{score_4dp:.4f}",
                reasoning,
            ])

    elapsed = time.time() - start_time
    print(f"\n[OK] Done! Wrote top {args.top_n} candidates to {args.out}")
    print(f"  Total time: {elapsed:.1f}s")

    # Print summary of top 10
    print(f"\n{'='*80}")
    print(f"  Top 10 Candidates")
    print(f"{'='*80}")
    for i, result in enumerate(top_n[:10]):
        cand = result["candidate"]
        profile = cand.get("profile", {})
        print(
            f"  {i+1:2d}. {cand['candidate_id']} | "
            f"{profile.get('current_title', 'N/A'):30s} | "
            f"{result['composite_score']:.4f} | "
            f"HP:{result['honeypot_score']:.2f}"
        )

    # Print timing breakdown
    print(f"\n  Timing: {elapsed:.1f}s total (limit: 300s)")
    if elapsed > 300:
        print("  [!!] WARNING: Exceeded 5-minute time limit!")
    else:
        print(f"  [OK] Within time limit ({elapsed/300*100:.0f}% used)")


if __name__ == "__main__":
    main()
