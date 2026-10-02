"""NLP inference engine with persistent model caching and real model execution."""

import logging
import threading
from typing import Any, Dict, List, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)


class NLPEngine:
    """Thread-safe NLP inference engine that loads models once per worker process."""

    _instance: Optional["NLPEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self._spacy_nlp = None
        self._sentiment_pipeline = None
        self._models_loaded = False
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "NLPEngine":
        """Return singleton instance of NLPEngine."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_models(self) -> None:
        """Load both spaCy NER model and HuggingFace sentiment analysis pipeline once into memory."""
        with self._init_lock:
            if self._models_loaded:
                logger.debug("NLP models already loaded in process.")
                return

            logger.info("Loading NLP models into worker process memory...")

            # 1. Load spaCy model for Named Entity Recognition
            self._load_spacy_model()

            # 2. Load HuggingFace Transformers pipeline for Sentiment Analysis
            self._load_sentiment_model()

            self._models_loaded = True
            logger.info("All NLP models successfully loaded into worker memory.")

    def _load_spacy_model(self) -> None:
        """Load the spaCy language model for NER."""
        import spacy

        model_name = self.settings.SPACY_MODEL
        try:
            logger.info("Loading spaCy model: %s", model_name)
            self._spacy_nlp = spacy.load(model_name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found locally. Attempting automatic download...",
                model_name,
            )
            from spacy.cli import download

            download(model_name)
            self._spacy_nlp = spacy.load(model_name)
        logger.info("spaCy model '%s' loaded successfully.", model_name)

    def _load_sentiment_model(self) -> None:
        """Load HuggingFace Transformers sentiment analysis pipeline."""
        from transformers import pipeline

        model_name = self.settings.TRANSFORMERS_MODEL
        logger.info("Loading HuggingFace sentiment pipeline: %s", model_name)
        # Use CPU device (device=-1)
        self._sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1,
        )
        logger.info("HuggingFace sentiment model '%s' loaded successfully.", model_name)

    def _ensure_models_loaded(self) -> None:
        """Verify models are loaded; if not, trigger loading with lock."""
        if not self._models_loaded:
            self.load_models()

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis on input text using HuggingFace Transformers.

        Args:
            text: Input string to analyze.

        Returns:
            Dict containing sentiment 'label' ('POSITIVE'/'NEGATIVE') and 'score'.
        """
        self._ensure_models_loaded()
        if not self._sentiment_pipeline:
            raise RuntimeError("Sentiment analysis model is not initialized.")

        # Truncate text to avoid transformer sequence length exceedance (> 512 tokens)
        truncated_text = text[:1500]
        results = self._sentiment_pipeline(truncated_text)

        if not results:
            raise RuntimeError("Sentiment model returned empty prediction.")

        prediction = results[0]
        label = prediction.get("label", "UNKNOWN").upper()
        score = round(float(prediction.get("score", 0.0)), 4)

        logger.info("Sentiment inference result: label=%s, score=%.4f", label, score)
        return {
            "label": label,
            "score": score,
        }

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Perform Named Entity Recognition (NER) on input text using spaCy.

        Args:
            text: Input string to process.

        Returns:
            Dict containing list of extracted entities with text, label, start, and end offsets.
        """
        self._ensure_models_loaded()
        if not self._spacy_nlp:
            raise RuntimeError("spaCy NER model is not initialized.")

        doc = self._spacy_nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

        logger.info("NER inference completed: extracted %d entities", len(entities))
        return {
            "entities": entities,
        }


def get_nlp_engine() -> NLPEngine:
    """Convenience helper to access the global NLPEngine instance."""
    return NLPEngine.get_instance()
