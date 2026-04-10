from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from preprocess import preprocess_text


# Skill dictionary
SKILL_DICTIONARY: Dict[str, List[str]] = {
    "programming_languages": [
        "python", "java", "c++", "c", "javascript", "typescript", "r", "sql", "scala",
    ],
    "ml_and_ai": [
        "machine learning", "deep learning", "neural networks", "classification",
        "regression", "clustering", "nlp", "natural language processing",
        "computer vision", "time series",
    ],
    "ml_libraries": [
        "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
        "pandas", "numpy", "matplotlib", "seaborn",
    ],
    "data_engineering": [
        "sql", "etl", "data pipeline", "data warehouse", "spark", "hadoop", "airflow",
    ],
    "cloud_and_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git", "linux",
    ],
    "soft_skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "critical thinking", "presentation",
    ],
}


@dataclass
class SkillAnalysis:
    detected_skills: Set[str]
    required_skills: Set[str]
    missing_skills: Set[str]
    match_score: float


def _normalize_for_matching(text: str) -> str:
    return preprocess_text(text)


def _flatten_skill_dict() -> Set[str]:
    skills: Set[str] = set()
    for skill_list in SKILL_DICTIONARY.values():
        for skill in skill_list:
            skills.add(skill.lower())
    return skills


ALL_KNOWN_SKILLS: Set[str] = _flatten_skill_dict()


def extract_skills_from_text(text: str) -> Set[str]:
    """
    Keyword-based skill extraction with safer matching.
    """
    if not text:
        return set()

    normalized = _normalize_for_matching(text)
    tokens = set(normalized.split())

    detected: Set[str] = set()

    for skill in ALL_KNOWN_SKILLS:
        skill_lower = skill.lower()

        # ✅ Better matching:
        if " " in skill_lower:
            # multi-word skill
            if skill_lower in normalized:
                detected.add(skill_lower)
        else:
            # single-word skill → match only exact tokens
            if skill_lower in tokens:
                detected.add(skill_lower)

    return detected


def extract_required_skills_from_job_description(job_description: str) -> Set[str]:
    return extract_skills_from_text(job_description)


def analyze_skill_match(
    resume_text: str,
    job_description: str,
) -> SkillAnalysis:
    """
    Compute detected skills, required skills, gaps, and match percentage.
    """
    detected = extract_skills_from_text(resume_text)
    required = extract_required_skills_from_job_description(job_description)

    if not required:
        match_score = 100.0 if detected else 0.0
        missing: Set[str] = set()
    else:
        missing = required - detected
        covered = required & detected
        match_score = (len(covered) / len(required)) * 100.0

    return SkillAnalysis(
        detected_skills=detected,
        required_skills=required,
        missing_skills=missing,
        match_score=round(match_score, 2),
    )


def skill_sets_to_strings(skills: Set[str]) -> List[str]:
    return sorted(skills)