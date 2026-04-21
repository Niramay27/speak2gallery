#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_processor.py
==================
Step 3 & 4 of the hybrid search pipeline:

  3. Normalize query
       - optional translation to English (Hindi / Bengali → English)
       - lowercasing
       - remove filler phrases ("show me", "give me", "find", etc.)

  4. Extract query components
       - objects  : dog, bench, paper
       - attributes : red, old, black
       - relations  : on, beside, wearing
       - quoted text : "hello world"

Usage (standalone):
    from query_processor import QueryProcessor
    qp = QueryProcessor(translate=True)
    result = qp.process("show me a red dog sitting on a bench")
    print(result)
"""

import re
import string
from dataclasses import dataclass, field
from typing import List, Optional

# ─── optional heavy deps (graceful fallback) ──────────────────────────────────
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# ─── filler phrases to strip from spoken queries ─────────────────────────────
FILLER_PHRASES = [
    # English
    r"\bshow me\b", r"\bgive me\b", r"\bfind me\b", r"\bfind\b",
    r"\bsearch for\b", r"\blook for\b", r"\bget me\b", r"\bi want to see\b",
    r"\bcan you show\b", r"\bplease show\b", r"\bdisplay\b", r"\bfetch\b",
    r"\bwhere is\b", r"\bwhere are\b",
    # Hindi romanised / transliterated common phrases
    r"\bdikha\b", r"\bdikhaao\b", r"\bkhojo\b", r"\blaao\b",
    # Bengali romanised
    r"\bdekhaao\b", r"\bkhunjo\b",
]
_FILLER_RE = re.compile("|".join(FILLER_PHRASES), re.IGNORECASE)

# ─── common English stop words that are not useful for tag matching ───────────
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "and", "or", "but", "if",
    "in", "at", "to", "for", "with", "about", "by", "from", "of", "that",
    "this", "these", "those", "it", "its", "my", "me", "i", "you", "he",
    "she", "we", "they", "some", "any", "which", "who", "what", "where",
    "when", "how", "there", "here",
}

# ─── simple POS-tag lists (fallback when spaCy unavailable) ──────────────────
# Top COCO object labels – used for keyword-based object detection fallback
COCO_OBJECTS = {
    "person", "people", "man", "woman", "boy", "girl", "child", "baby",
    "bicycle", "bike", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic", "fire", "hydrant", "stop", "sign",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "umbrella", "handbag", "bag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "ball", "kite",
    "baseball", "bat", "glove", "skateboard", "surfboard", "racket",
    "bottle", "wine", "glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza",
    "donut", "cake", "chair", "couch", "sofa", "plant", "bed", "table",
    "toilet", "tv", "television", "laptop", "mouse", "remote", "keyboard",
    "phone", "microwave", "oven", "toaster", "sink", "refrigerator", "fridge",
    "book", "clock", "vase", "scissors", "teddy", "bear", "toothbrush",
    "street", "road", "park", "beach", "ocean", "sea", "sky", "mountain",
    "tree", "field", "grass", "snow", "rain", "building", "house", "room",
    "kitchen", "bathroom", "bedroom", "door", "window", "wall", "floor",
    "surfboard", "helmet", "hat", "shirt", "jacket", "pants", "shoes",
}

# Simple colour / attribute words (fallback)
COMMON_ATTRIBUTES = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "grey", "gray", "brown", "golden", "silver", "dark", "light",
    "bright", "big", "large", "small", "little", "tall", "short", "long",
    "young", "old", "new", "old", "wooden", "metal", "plastic", "glass",
    "open", "closed", "empty", "full", "wet", "dry", "hot", "cold",
    "happy", "sad", "angry", "running", "sitting", "standing", "lying",
    "flying", "swimming", "eating", "drinking", "playing", "sleeping",
}

# Common spatial/other relations
COMMON_RELATIONS = {
    "on", "in", "at", "by", "near", "beside", "next", "behind", "front",
    "above", "below", "under", "over", "between", "around", "through",
    "wearing", "holding", "carrying", "riding", "with", "without",
    "inside", "outside", "against", "along", "across", "toward",
}


@dataclass
class QueryComponents:
    """Structured output of query processing."""
    raw_text: str = ""                      # original text (pre-normalization)
    normalized_text: str = ""               # cleaned English text
    detected_lang: str = "en"
    objects: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    quoted_text: List[str] = field(default_factory=list)
    all_keywords: List[str] = field(default_factory=list)  # union of above

    def __repr__(self):
        return (
            f"QueryComponents(\n"
            f"  raw='{self.raw_text}'\n"
            f"  normalized='{self.normalized_text}'\n"
            f"  lang='{self.detected_lang}'\n"
            f"  objects={self.objects}\n"
            f"  attributes={self.attributes}\n"
            f"  relations={self.relations}\n"
            f"  quoted={self.quoted_text}\n"
            f"  keywords={self.all_keywords}\n"
            f")"
        )


class QueryProcessor:
    """
    Processes a raw text query (already ASR-transcribed) into a structured
    QueryComponents object suitable for hybrid retrieval.

    Parameters
    ----------
    translate : bool
        If True and transformers is available, translate Hindi/Bengali to
        English before extraction.  Requires ``facebook/nllb-200-distilled-600M``
        to be locally available or downloadable.
    use_spacy : bool
        If True and spaCy is installed with an English model, use it for
        richer NER/POS extraction.  Falls back to keyword matching otherwise.
    spacy_model : str
        Name of the spaCy model to load (default: ``en_core_web_sm``).
    """

    def __init__(
        self,
        translate: bool = True,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        self.translate = translate
        self._translator = None
        self._nlp = None

        # ── optional spaCy ──
        if use_spacy and _SPACY_AVAILABLE:
            try:
                import spacy
                self._nlp = spacy.load(spacy_model)
                print(f"[QueryProcessor] spaCy model '{spacy_model}' loaded.")
            except OSError:
                print(
                    f"[QueryProcessor] spaCy model '{spacy_model}' not found. "
                    f"Run: python -m spacy download {spacy_model}\n"
                    f"Falling back to keyword extraction."
                )

        # ── optional translation pipeline ──
        if translate and _HF_AVAILABLE:
            self._load_translator()

    def _load_translator(self):
        """Lazy-load NLLB translation pipeline (same model used in training)."""
        try:
            from transformers import pipeline as hf_pipeline
            print("[QueryProcessor] Loading NLLB translator (this may take a moment)…")
            self._translator = hf_pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M",
                device=-1,   # CPU; set to 0 for GPU
            )
            print("[QueryProcessor] Translator ready.")
        except Exception as e:
            print(f"[QueryProcessor] Could not load translator: {e}")
            self._translator = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, text: str, src_lang: str = "auto") -> QueryComponents:
        """
        Full pipeline: detect language → translate → normalize → extract.

        Parameters
        ----------
        text : str
            Raw transcribed text (from ASR or typed input).
        src_lang : str
            Source language hint: 'en', 'hi', 'bn', or 'auto' for detection.

        Returns
        -------
        QueryComponents
        """
        qc = QueryComponents(raw_text=text)

        # 1. detect language
        qc.detected_lang = src_lang if src_lang != "auto" else self._detect_lang(text)

        # 2. translate to English if needed
        english_text = text
        if qc.detected_lang in ("hi", "bn") and self._translator is not None:
            english_text = self._translate_to_english(text, qc.detected_lang)
        elif qc.detected_lang in ("hi", "bn"):
            # Crude romanisation fallback: just keep as-is and hope for the best
            # (CLIP text encoder can handle some Hindi/Bengali in CLIP space
            #  because of the language-bridge training)
            english_text = text

        # 3. normalize
        normalized = self._normalize(english_text)
        qc.normalized_text = normalized

        # 4. extract components
        quoted = self._extract_quoted(normalized)
        qc.quoted_text = quoted

        # remove quotes from the text before further processing
        clean = re.sub(r'"[^"]*"', " ", normalized)

        if self._nlp is not None:
            self._extract_with_spacy(clean, qc)
        else:
            self._extract_with_keywords(clean, qc)

        # add quoted text entities as objects too
        qc.objects = list(dict.fromkeys(qc.objects + quoted))  # deduplicate

        # build all_keywords (unique, ordered)
        all_kw = []
        seen = set()
        for lst in (qc.objects, qc.attributes, qc.relations, qc.quoted_text):
            for w in lst:
                w_l = w.lower()
                if w_l not in seen:
                    seen.add(w_l)
                    all_kw.append(w_l)
        qc.all_keywords = all_kw

        return qc

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_lang(text: str) -> str:
        """
        Heuristic language detection based on Unicode character ranges.
        Returns 'hi', 'bn', or 'en'.
        """
        devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        bengali     = sum(1 for c in text if "\u0980" <= c <= "\u09FF")
        total = len(text.strip()) or 1
        if devanagari / total > 0.15:
            return "hi"
        if bengali / total > 0.15:
            return "bn"
        return "en"

    def _translate_to_english(self, text: str, src_lang: str) -> str:
        """Translate Hindi or Bengali text to English using NLLB."""
        lang_map = {"hi": "hin_Deva", "bn": "ben_Beng"}
        src_code = lang_map.get(src_lang, "hin_Deva")
        try:
            result = self._translator(
                text,
                src_lang=src_code,
                tgt_lang="eng_Latn",
                max_length=256,
            )
            translated = result[0]["translation_text"]
            print(f"[QueryProcessor] Translated ({src_lang}→en): '{text}' → '{translated}'")
            return translated
        except Exception as e:
            print(f"[QueryProcessor] Translation failed: {e}. Using raw text.")
            return text

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip fillers, remove punctuation (except quotes)."""
        text = text.lower()

        # extract and protect quoted substrings
        quotes = re.findall(r'"[^"]*"', text)
        placeholder_map = {}
        for i, q in enumerate(quotes):
            ph = f"__QUOTE{i}__"
            placeholder_map[ph] = q
            text = text.replace(q, ph, 1)

        # strip filler phrases
        text = _FILLER_RE.sub(" ", text)

        # remove punctuation except apostrophes and placeholders
        allowed = set(string.ascii_letters + string.digits + " _'")
        text = "".join(c if c in allowed else " " for c in text)

        # restore quoted substrings
        for ph, q in placeholder_map.items():
            text = text.replace(ph, q)

        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _extract_quoted(text: str) -> List[str]:
        """Return list of strings inside double quotes."""
        return re.findall(r'"([^"]+)"', text)

    def _extract_with_spacy(self, text: str, qc: QueryComponents):
        """
        Use spaCy POS/NER tagging for richer extraction.
        - NOUN / PROPN  → objects
        - ADJ           → attributes
        - Named Entities that are not proper nouns → objects
        """
        doc = self._nlp(text)

        objects, attributes, relations = [], [], []

        for token in doc:
            w = token.text.lower()
            if w in STOP_WORDS or len(w) < 2:
                continue
            if token.pos_ in ("NOUN", "PROPN"):
                objects.append(w)
            elif token.pos_ == "ADJ":
                attributes.append(w)
            elif token.pos_ == "ADP" and w in COMMON_RELATIONS:
                relations.append(w)
            elif token.pos_ == "VERB" and w in COMMON_RELATIONS:
                relations.append(w)

        # deduplicate preserving order
        qc.objects    = list(dict.fromkeys(objects))
        qc.attributes = list(dict.fromkeys(attributes))
        qc.relations  = list(dict.fromkeys(relations))

    @staticmethod
    def _extract_with_keywords(text: str, qc: QueryComponents):
        """
        Fallback: simple token-level keyword matching against known word sets.
        Works for COCO-domain queries without any ML dependency.
        """
        tokens = text.lower().split()
        objects, attributes, relations = [], [], []
        seen = set()

        for tok in tokens:
            clean_tok = tok.strip(string.punctuation)
            if clean_tok in seen or clean_tok in STOP_WORDS or len(clean_tok) < 2:
                continue
            seen.add(clean_tok)
            if clean_tok in COCO_OBJECTS:
                objects.append(clean_tok)
            elif clean_tok in COMMON_ATTRIBUTES:
                attributes.append(clean_tok)
            elif clean_tok in COMMON_RELATIONS:
                relations.append(clean_tok)

        qc.objects    = objects
        qc.attributes = attributes
        qc.relations  = relations


# ─── Convenience function ────────────────────────────────────────────────────

def process_query(text: str, src_lang: str = "auto", translate: bool = False) -> QueryComponents:
    """
    Module-level convenience wrapper. Creates a stateless QueryProcessor
    (no translation, no spaCy by default for speed) and processes the query.
    For repeated calls, instantiate QueryProcessor once and reuse it.
    """
    qp = QueryProcessor(translate=translate, use_spacy=True)
    return qp.process(text, src_lang=src_lang)


# ─── CLI demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    examples = [
        ("show me a red dog sitting on a bench", "en"),
        ("find me photos of a woman wearing a blue hat", "en"),
        ("kutta surfboard par", "hi"),          # "dog on surfboard" in Hindi
        ("একটি লাল গাড়ি রাস্তায়", "bn"),       # "a red car on the road" in Bengali
        ('find images with "hello world"', "en"),
        ("show me old wooden bench beside a tree", "en"),
    ]

    qp = QueryProcessor(translate=False, use_spacy=True)
    print("=" * 60)
    for text, lang in examples:
        result = qp.process(text, src_lang=lang)
        print(result)
        print("-" * 60)
