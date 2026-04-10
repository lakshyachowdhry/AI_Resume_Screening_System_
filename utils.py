from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


# Root directory
ROOT_DIR = Path(__file__).resolve().parent

# Model directory
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Output / logs directory
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "logs.txt"


def setup_logging() -> None:
    """
    Configure logging (safe for repeated calls).
    """
    if logging.getLogger().handlers:
        return  # جلوگیری duplicate handlers

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def log_event(event: str, extras: Optional[Dict[str, Any]] = None) -> None:
    """
    Append structured logs to file.
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "extras": extras or {},
    }

    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass  # fail silently (important for Streamlit)


def safe_load_joblib(path: Path) -> Any:
    """
    Load joblib file safely.
    """
    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


@dataclass
class ScoreBreakdown:
    similarity_score: float  # 0–100
    skills_score: float  # 0–100
    model_probability: Optional[float]  # 0–1
    final_score: float  # 0–100


def compute_final_score(
    similarity_score: float,
    skills_score: float,
    model_probability: Optional[float] = None,
    w_similarity: float = 0.4,
    w_skills: float = 0.4,
    w_model: float = 0.2,
) -> ScoreBreakdown:
    """
    Combine similarity, skills, and model probability into final score.
    """
    sim_pct = max(0.0, min(1.0, similarity_score))
    skills_pct = max(0.0, min(100.0, skills_score)) / 100.0

    if model_probability is None:
        # redistribute weights
        w_similarity += w_model / 2
        w_skills += w_model / 2
        w_model = 0.0
        model_probability = 0.0

    total = w_similarity + w_skills + w_model
    if total == 0:
        total = 1.0

    final = (
        (w_similarity / total) * sim_pct
        + (w_skills / total) * skills_pct
        + (w_model / total) * float(model_probability)
    )

    return ScoreBreakdown(
        similarity_score=round(sim_pct * 100, 2),
        skills_score=round(skills_pct * 100, 2),
        model_probability=round(float(model_probability), 4)
        if model_probability is not None
        else None,
        final_score=round(final * 100, 2),
    )


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to CSV (for Streamlit download).
    """
    try:
        return df.to_csv(index=False).encode("utf-8")
    except Exception:
        return b""