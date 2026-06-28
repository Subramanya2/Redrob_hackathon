"""
JD Parser — Extracts structured requirements from the Senior AI Engineer job description.

This module encodes the job description into structured signals that the scorer can use.
Rather than doing NLP on the JD text at runtime, we pre-encode the requirements since
the JD is fixed for this challenge.
"""

# ─────────────────────────────────────────────────────────────
# Must-have skills (JD: "Things you absolutely need")
# ─────────────────────────────────────────────────────────────
MUST_HAVE_SKILL_CLUSTERS = {
    "embeddings_retrieval": {
        "keywords": [
            "embeddings", "sentence-transformers", "sentence transformers",
            "openai embeddings", "bge", "e5", "embedding", "dense retrieval",
            "semantic search", "vector search", "embedding drift",
            "retrieval", "information retrieval", "IR",
        ],
        "weight": 1.0,
        "description": "Production experience with embeddings-based retrieval systems",
    },
    "vector_databases": {
        "keywords": [
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss", "vector database", "vector db",
            "hybrid search", "vector store", "ann", "approximate nearest",
            "hnsw", "ivf", "chromadb", "chroma",
        ],
        "weight": 1.0,
        "description": "Production experience with vector databases or hybrid search",
    },
    "python_strong": {
        "keywords": [
            "python", "pytest", "fastapi", "flask", "django",
            "asyncio", "pydantic", "poetry", "pip",
        ],
        "weight": 0.8,
        "description": "Strong Python engineering skills",
    },
    "ranking_evaluation": {
        "keywords": [
            "ndcg", "mrr", "map", "precision", "recall",
            "ranking", "evaluation", "a/b test", "ab test",
            "offline evaluation", "online evaluation",
            "learning to rank", "ltr", "reranking", "re-ranking",
            "ranking system", "search quality", "relevance",
        ],
        "weight": 1.0,
        "description": "Evaluation frameworks for ranking systems",
    },
}

# ─────────────────────────────────────────────────────────────
# Nice-to-have skills (JD: "Things we'd like you to have")
# ─────────────────────────────────────────────────────────────
NICE_TO_HAVE_SKILL_CLUSTERS = {
    "llm_finetuning": {
        "keywords": [
            "lora", "qlora", "peft", "fine-tuning", "fine tuning",
            "finetuning", "instruction tuning", "rlhf", "dpo",
            "sft", "supervised fine-tuning", "adapter",
        ],
        "weight": 0.5,
        "description": "LLM fine-tuning experience",
    },
    "learning_to_rank": {
        "keywords": [
            "xgboost", "lightgbm", "catboost", "gradient boosting",
            "learning to rank", "lambdamart", "lambdarank",
            "neural ranking", "cross-encoder",
        ],
        "weight": 0.4,
        "description": "Learning-to-rank models",
    },
    "hrtech_marketplace": {
        "keywords": [
            "hr tech", "hrtech", "recruiting", "recruitment",
            "talent", "hiring", "marketplace", "matching platform",
            "candidate matching", "job matching", "ats",
            "applicant tracking",
        ],
        "weight": 0.3,
        "description": "Prior exposure to HR-tech or marketplace products",
    },
    "distributed_systems": {
        "keywords": [
            "distributed systems", "kubernetes", "k8s", "docker",
            "microservices", "kafka", "rabbitmq", "celery",
            "inference optimization", "model serving", "triton",
            "onnx", "tensorrt", "scaling", "load balancing",
        ],
        "weight": 0.3,
        "description": "Distributed systems or inference optimization",
    },
    "open_source": {
        "keywords": [
            "open source", "open-source", "github", "contributor",
            "maintainer", "pull request", "oss",
        ],
        "weight": 0.2,
        "description": "Open-source contributions in AI/ML",
    },
}

# ─────────────────────────────────────────────────────────────
# Core AI/ML domain skills (broader matching for general fit)
# ─────────────────────────────────────────────────────────────
AI_ML_CORE_SKILLS = {
    "keywords": [
        # ML fundamentals
        "machine learning", "deep learning", "neural network",
        "nlp", "natural language processing", "text classification",
        "named entity recognition", "ner", "sentiment analysis",
        "transformer", "transformers", "bert", "gpt", "llm",
        "large language model", "attention mechanism",
        # ML frameworks
        "pytorch", "tensorflow", "keras", "huggingface",
        "hugging face", "scikit-learn", "sklearn", "jax",
        # MLOps
        "mlops", "ml pipeline", "model deployment", "mlflow",
        "weights and biases", "wandb", "experiment tracking",
        "model monitoring", "feature store",
        # Data engineering (relevant per JD)
        "data pipeline", "etl", "airflow", "spark", "pyspark",
        "data engineering", "data warehouse",
        # Search & recommendations
        "recommendation system", "recommender", "search engine",
        "search ranking", "bm25", "tf-idf", "lucene",
        "solr", "elastic", "rag", "retrieval augmented",
        # Specific AI/ML techniques
        "computer vision", "cv", "image classification",
        "object detection", "segmentation",
        "reinforcement learning", "rl",
        "generative ai", "diffusion", "gan",
    ],
}

# ─────────────────────────────────────────────────────────────
# Relevant title patterns (what the JD is actually looking for)
# ─────────────────────────────────────────────────────────────
HIGHLY_RELEVANT_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "senior machine learning engineer", "staff ml engineer",
    "applied ml engineer", "applied scientist",
    "data scientist", "senior data scientist",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation engineer", "retrieval engineer",
    "ml platform engineer", "mlops engineer",
    "research engineer",
]

MODERATELY_RELEVANT_TITLES = [
    "software engineer", "senior software engineer",
    "backend engineer", "senior backend engineer",
    "data engineer", "senior data engineer",
    "full stack engineer", "platform engineer",
    "technical lead", "tech lead",
    "engineering manager",
]

IRRELEVANT_TITLES = [
    "marketing manager", "hr manager", "human resources",
    "sales executive", "sales manager",
    "accountant", "accounting", "finance manager",
    "operations manager", "project manager",
    "content writer", "copywriter", "content strategist",
    "graphic designer", "ui designer", "ux designer",
    "customer support", "customer service",
    "business analyst", "product manager",
    "civil engineer", "mechanical engineer",
    "electrical engineer", "chemical engineer",
    "teacher", "professor", "lecturer",
    "lawyer", "legal", "advocate",
    "doctor", "nurse", "pharmacist",
]

# ─────────────────────────────────────────────────────────────
# Relevant industries
# ─────────────────────────────────────────────────────────────
HIGHLY_RELEVANT_INDUSTRIES = [
    "artificial intelligence", "machine learning", "ai",
    "technology", "software", "saas", "internet",
    "information technology", "it product",
    "cloud computing", "data analytics",
    "search", "e-commerce", "fintech",
]

CONSULTING_COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "l&t infotech", "lti", "ltimindtree",
    "hexaware", "persistent", "cyient", "zensar",
    "birlasoft", "niit technologies",
]

# ─────────────────────────────────────────────────────────────
# Experience requirements
# ─────────────────────────────────────────────────────────────
EXPERIENCE_RANGE = {
    "ideal_min": 5,
    "ideal_max": 9,
    "acceptable_min": 3,
    "acceptable_max": 14,
    "ai_ml_min_years": 2,  # Need at least ~2 years in AI/ML roles
}

# ─────────────────────────────────────────────────────────────
# Location preferences
# ─────────────────────────────────────────────────────────────
PREFERRED_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi",
    "new delhi", "gurgaon", "gurugram", "bangalore",
    "bengaluru", "chennai",
]

PREFERRED_COUNTRIES = ["india"]

# ─────────────────────────────────────────────────────────────
# JD text for semantic matching
# ─────────────────────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer, Founding Team at Redrob AI. Series A AI-native talent intelligence platform.
Own the intelligence layer: ranking, retrieval, and matching systems for recruiters and candidates.
Ship a v2 ranking system with embeddings, hybrid retrieval, LLM-based re-ranking.
Set up evaluation infrastructure: offline benchmarks, online A/B testing, recruiter-feedback loops.
Production experience with embeddings-based retrieval systems deployed to real users.
Production experience with vector databases or hybrid search infrastructure.
Strong Python, code quality matters.
Evaluation frameworks for ranking systems: NDCG, MRR, MAP, A/B test interpretation.
LLM fine-tuning, learning-to-rank models, HR-tech, distributed systems, open-source contributions.
Shipped end-to-end ranking, search, or recommendation system to real users at meaningful scale.
Strong opinions about retrieval hybrid vs dense, evaluation offline vs online, LLM integration.
Applied ML AI roles at product companies, not pure services.
Modern ML systems: embeddings, retrieval, ranking, LLMs, fine-tuning.
Scrappy product-engineering attitude, willing to ship fast.
Data pipelines, Spark, Airflow, feature engineering, ML infrastructure.
"""

JD_KEYWORDS_FOR_BM25 = [
    "ai", "ml", "machine learning", "deep learning",
    "embeddings", "retrieval", "ranking", "search",
    "vector database", "faiss", "pinecone", "weaviate",
    "python", "pytorch", "tensorflow",
    "nlp", "natural language processing", "transformer",
    "llm", "large language model", "fine-tuning",
    "recommendation", "evaluation", "ndcg", "mrr",
    "production", "deployed", "scale", "system",
    "data pipeline", "spark", "airflow",
    "product company", "startup",
]


def get_all_must_have_keywords():
    """Return flat set of all must-have skill keywords."""
    keywords = set()
    for cluster in MUST_HAVE_SKILL_CLUSTERS.values():
        keywords.update(kw.lower() for kw in cluster["keywords"])
    return keywords


def get_all_nice_to_have_keywords():
    """Return flat set of all nice-to-have skill keywords."""
    keywords = set()
    for cluster in NICE_TO_HAVE_SKILL_CLUSTERS.values():
        keywords.update(kw.lower() for kw in cluster["keywords"])
    return keywords


def get_all_ai_ml_keywords():
    """Return flat set of all AI/ML domain keywords."""
    return set(kw.lower() for kw in AI_ML_CORE_SKILLS["keywords"])
