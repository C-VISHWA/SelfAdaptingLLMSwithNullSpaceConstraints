"""Deterministic sampling of real SQuAD articles for continual self-edit sequences."""
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_articles(path: str, min_questions: int = 8) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    return [a for a in data if len(a.get("questions", [])) >= min_questions]


def sample_sequence(
    articles: List[Dict[str, Any]], k: int, questions_per_article: int, seed: int
) -> List[Dict[str, Any]]:
    if k > len(articles):
        raise ValueError(f"Cannot sample {k} articles from {len(articles)} candidates")
    if questions_per_article < 1:
        raise ValueError("questions_per_article must be positive")
    rng = random.Random(seed)
    chosen = rng.sample(articles, k)
    out = []
    for a in chosen:
        if len(a["questions"]) < questions_per_article:
            raise ValueError(f"Article {a.get('title', '<untitled>')!r} has too few questions")
        qs = rng.sample(a["questions"], questions_per_article)
        out.append({"title": a["title"], "context": a["context"], "questions": qs})
    return out
