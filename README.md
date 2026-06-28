# Intelligent Candidate Discovery & Ranking System

**Redrob Hackathon — India Runs Data & AI Challenge**

An AI-powered candidate ranking system that goes beyond keyword matching to understand who genuinely fits a role. Built for the Senior AI Engineer JD, this system analyzes career trajectories, validates skill depth, weighs behavioral signals, and detects honeypot profiles — producing a ranked shortlist that a recruiter can trust.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run ranking (produces submission.csv)
python rank.py --candidates ./data/India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

**Runtime**: ~90-120 seconds on a standard 16GB CPU machine.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  100,000 Candidates (JSONL)               │
└──────────────────────┬───────────────────────────────────┘
                       │
                 ┌─────▼─────┐
                 │ Pre-Filter │  Eliminate clearly irrelevant
                 │ (Title +   │  candidates while keeping
                 │  Keywords) │  any plausible fit
                 └─────┬─────┘
                       │ ~15-20K candidates
          ┌────────────┼────────────────┐
          │            │                │
    ┌─────▼─────┐ ┌───▼────┐  ┌───────▼────────┐
    │ Structured │ │Semantic│  │   Honeypot     │
    │  Scoring   │ │ Text   │  │   Detection    │
    │ (6 dims)   │ │Matching│  │ (8 flag types) │
    └─────┬─────┘ └───┬────┘  └───────┬────────┘
          │            │                │
          └────────────┼────────────────┘
                       │
              ┌────────▼────────┐
              │ Composite Score │  Weighted merge +
              │   + Honeypot    │  honeypot penalty
              │    Penalty      │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Top 100 + CSV  │  Ranked output with
              │  with Reasoning │  data-driven reasoning
              └─────────────────┘
```

## Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| **Career Relevance** | 30% | Title match, product vs consulting, AI/ML role depth, job stability |
| **Skill Match** | 25% | Must-have/nice-to-have overlap with proficiency & duration validation |
| **Semantic Match** | 15% | TF-IDF + BM25 text similarity of profile to JD |
| **Experience Fit** | 10% | Years of experience within the 5-9 year sweet spot |
| **Behavioral Signals** | 10% | Platform activity, response rate, interview completion |
| **Availability** | 5% | Location, notice period, work mode, salary range |
| **Credibility** | 5% | GitHub activity, education tier, profile completeness |

## Key Design Decisions

### Why Not Use Embedding Models?
The 5-minute CPU constraint makes running sentence-transformer models over 15K+ candidates impractical. TF-IDF + BM25 achieves strong semantic matching at ~100x the speed on CPU, with no model download required.

### Why Career History > Keywords?
The JD explicitly warns: *"A candidate who has all AI keywords as skills but whose title is 'Marketing Manager' is not a fit."* We weight career trajectory (30%) highest — analyzing actual roles, industries, company types, and role descriptions.

### Honeypot Detection
~80 candidates have subtly impossible profiles (expert in 10+ skills with 0 duration, impossible company tenures). Our 8-flag detector catches these before they pollute rankings. Submissions with >10% honeypot rate are disqualified.

### Behavioral Signals as Multipliers
Following the signals doc guidance, behavioral data modifies skill-match scores rather than replacing them. A perfect-on-paper candidate inactive for 6 months gets down-weighted; an active, responsive candidate gets boosted.

## Project Structure

```
├── rank.py                      # Main entry point
├── src/
│   ├── jd_parser.py             # JD requirements → structured signals
│   ├── candidate_scorer.py      # 6-dimension structured scoring
│   ├── text_scorer.py           # TF-IDF + BM25 semantic matching
│   ├── honeypot_detector.py     # Impossible profile detection
│   └── reasoning_generator.py   # Per-candidate reasoning generation
├── requirements.txt             # Dependencies
├── submission.csv               # Output ranking (top 100)
└── README.md                    # This file
```

## Dependencies

- `scikit-learn` — TF-IDF vectorization and cosine similarity
- `numpy` — Numerical operations
- `pandas` — Data manipulation
- `rank-bm25` — BM25 scoring

All CPU-friendly, no GPU required, no large model downloads.

## Reproduction

```bash
# Single command to reproduce submission
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# With verbose output
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --verbose
```

## Author

**Subramanya Nayak**

Built for the Redrob Intelligent Candidate Discovery & Ranking Challenge.
