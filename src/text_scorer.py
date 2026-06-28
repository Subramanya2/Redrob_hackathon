"""
Text Scorer — Lightweight semantic matching using TF-IDF and BM25.

Scores candidate text (summary + career descriptions) against JD requirements
without needing GPU or large model downloads.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from src.jd_parser import JD_TEXT, JD_KEYWORDS_FOR_BM25


def _tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-+#.]', ' ', text)
    tokens = text.split()
    # Remove very short tokens except known abbreviations
    tokens = [t for t in tokens if len(t) > 1 or t in {'r', 'c', 'ai'}]
    return tokens


def build_candidate_text(candidate):
    """
    Combine all relevant text fields from a candidate into a single document.

    This creates the "text representation" of the candidate that we match
    against the JD. We include:
    - Profile headline and summary
    - All career history descriptions and titles
    - Skill names
    - Certification names
    """
    parts = []

    profile = candidate.get("profile", {})
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])
    if profile.get("current_title"):
        parts.append(profile["current_title"])

    # Career history — titles and descriptions are gold
    for job in candidate.get("career_history", []):
        if job.get("title"):
            parts.append(job["title"])
        if job.get("description"):
            parts.append(job["description"])
        if job.get("industry"):
            parts.append(job["industry"])

    # Skills as text
    skill_names = [s["name"] for s in candidate.get("skills", []) if s.get("name")]
    if skill_names:
        parts.append(" ".join(skill_names))

    # Certifications
    for cert in candidate.get("certifications", []):
        if cert.get("name"):
            parts.append(cert["name"])

    return " ".join(parts)


class TextScorer:
    """
    Hybrid TF-IDF + BM25 text scorer.

    Usage:
        scorer = TextScorer()
        scorer.fit(candidate_texts)  # list of candidate text documents
        scores = scorer.score_all()  # returns array of scores [0, 1]
    """

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            stop_words='english',
        )
        self.jd_text = JD_TEXT
        self.jd_keywords = JD_KEYWORDS_FOR_BM25
        self._tfidf_scores = None
        self._bm25_scores = None

    def fit_and_score(self, candidate_texts):
        """
        Fit on candidate texts and compute scores in one pass.

        Args:
            candidate_texts: list of str, one per candidate

        Returns:
            np.array of combined scores in [0, 1], shape (n_candidates,)
        """
        n = len(candidate_texts)

        # ── TF-IDF scoring ──
        # Add JD as the first document, then compute cosine similarity
        all_docs = [self.jd_text] + candidate_texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_docs)

        # Cosine similarity between JD (index 0) and each candidate
        jd_vector = tfidf_matrix[0:1]
        candidate_vectors = tfidf_matrix[1:]
        tfidf_similarities = cosine_similarity(jd_vector, candidate_vectors).flatten()

        # Normalize to [0, 1]
        if tfidf_similarities.max() > 0:
            tfidf_scores = tfidf_similarities / tfidf_similarities.max()
        else:
            tfidf_scores = np.zeros(n)

        # ── BM25 scoring ──
        # Tokenize all candidate texts for BM25
        tokenized_candidates = [_tokenize(text) for text in candidate_texts]
        bm25 = BM25Okapi(tokenized_candidates)

        # Score each candidate against the JD keywords
        jd_query = _tokenize(self.jd_text)
        bm25_raw_scores = bm25.get_scores(jd_query)

        # Normalize to [0, 1]
        if bm25_raw_scores.max() > 0:
            bm25_scores = bm25_raw_scores / bm25_raw_scores.max()
        else:
            bm25_scores = np.zeros(n)

        # ── Combine: 60% TF-IDF + 40% BM25 ──
        combined = 0.6 * tfidf_scores + 0.4 * bm25_scores

        self._tfidf_scores = tfidf_scores
        self._bm25_scores = bm25_scores

        return combined
