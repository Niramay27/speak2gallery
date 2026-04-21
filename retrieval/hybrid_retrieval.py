#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_retrieval.py
===================
Step 5 of the hybrid search pipeline: retrieve images using:
  a) Semantic similarity from the full normalised sentence (CLIP text embedding)
  b) Object/tag-based filtering using extracted keywords

The two scores are fused with a configurable alpha parameter:

    final_score = alpha * semantic_score + (1 - alpha) * tag_score

Architecture
------------
- Image index: pre-built .pt file with {image_paths, embeddings}
  (same format produced by build_image_index.py)
- CLIP text encoder: used to embed the normalised query string
- Tag index (optional): a JSON mapping image_path → [list of COCO tags]
  If not provided, tag filtering is skipped (pure semantic search).

Usage
-----
    from hybrid_retrieval import HybridRetriever
    from query_processor import QueryProcessor

    retriever = HybridRetriever(
        image_index_path="image_index.pt",
        clip_model_name="ViT-B-32",
    )
    qp = QueryProcessor(translate=False)
    qc = qp.process("red dog on surfboard")

    results = retriever.retrieve(qc, top_k=9)
    for rank, r in enumerate(results, 1):
        print(rank, r['score']:.4f, r['image_path'])
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
import open_clip

from query_processor import QueryComponents


class HybridRetriever:
    """
    Loads a pre-built CLIP image index and performs hybrid retrieval.

    Parameters
    ----------
    image_index_path : str
        Path to ``image_index.pt`` produced by ``build_image_index.py``.
    clip_model_name : str
        Open-CLIP model name (default: ``ViT-B-32``).
    clip_pretrained : str
        Open-CLIP pretrained weights key (default: ``openai``).
    tag_index_path : str or None
        Optional path to a JSON file mapping image paths to tag lists:
        ``{"path/to/img.jpg": ["dog", "beach", "person"], ...}``
        If None, tag filtering is disabled (pure semantic mode).
    alpha : float
        Weight for semantic score in fusion (0.0 = pure tag, 1.0 = pure semantic).
        Default: 0.7
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        image_index_path: str,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        tag_index_path: Optional[str] = None,
        alpha: float = 0.7,
        device: Optional[str] = None,
    ):
        self.alpha = alpha
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HybridRetriever] Using device: {self.device}")

        # ── load CLIP text encoder ──
        self._clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self._clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        self._clip_model = self._clip_model.to(self.device).eval()
        for p in self._clip_model.parameters():
            p.requires_grad = False
        print(f"[HybridRetriever] CLIP model '{clip_model_name}' loaded.")

        # ── load image index ──
        index_data = torch.load(image_index_path, map_location="cpu")
        self.image_paths: List[str] = index_data["image_paths"]
        self.image_embeddings: torch.Tensor = index_data["embeddings"]  # [N, D], normalised
        print(f"[HybridRetriever] Image index loaded: {len(self.image_paths)} images.")

        # ── optional tag index ──
        self.tag_index: Optional[Dict[str, List[str]]] = None
        if tag_index_path and Path(tag_index_path).exists():
            with open(tag_index_path, "r", encoding="utf-8") as f:
                self.tag_index = json.load(f)
            print(f"[HybridRetriever] Tag index loaded: {len(self.tag_index)} entries.")
        else:
            print("[HybridRetriever] No tag index provided — running in pure semantic mode.")

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode a text string to a normalised CLIP embedding [1, D]."""
        tokens = self._clip_tokenizer([text]).to(self.device)
        emb = self._clip_model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu()

    def _compute_semantic_scores(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between query embedding and all image embeddings.
        Returns [N] tensor in range [-1, 1].
        """
        # image_embeddings already normalised: sim = q · i^T
        scores = (self.image_embeddings @ query_emb.T).squeeze(-1)  # [N]
        return scores

    def _compute_tag_scores(self, keywords: List[str]) -> torch.Tensor:
        """
        Simple keyword-overlap score with per-image tag lists.
        score_i = |keywords ∩ tags_i| / max(|keywords|, 1)
        Returns [N] tensor in range [0, 1].
        """
        if self.tag_index is None or not keywords:
            return torch.zeros(len(self.image_paths))

        kw_set = set(k.lower() for k in keywords)
        scores = []
        for path in self.image_paths:
            tags = set(t.lower() for t in self.tag_index.get(path, []))
            overlap = len(kw_set & tags)
            scores.append(overlap / max(len(kw_set), 1))
        return torch.tensor(scores, dtype=torch.float32)

    def retrieve(
        self,
        query: "QueryComponents",
        top_k: int = 9,
        alpha: Optional[float] = None,
        min_tag_score: float = 0.0,
    ) -> List[Dict]:
        """
        Hybrid retrieval from a QueryComponents object.

        Parameters
        ----------
        query : QueryComponents
            Output of QueryProcessor.process().
        top_k : int
            Number of results to return.
        alpha : float or None
            Override instance-level alpha for this call.
        min_tag_score : float
            If >0 and tag_index is available, images with tag_score below this
            threshold are excluded before semantic re-ranking.

        Returns
        -------
        List of dicts:
            [{
                'rank': int,
                'image_path': str,
                'semantic_score': float,
                'tag_score': float,
                'final_score': float,
            }, ...]
        """
        alpha = alpha if alpha is not None else self.alpha

        # ── encode full normalised query for semantic search ──
        text_to_encode = query.normalized_text or query.raw_text
        query_emb = self._encode_text(text_to_encode)

        # ── semantic scores ──
        sem_scores = self._compute_semantic_scores(query_emb)        # [N]

        # ── tag scores ──
        tag_scores = self._compute_tag_scores(query.all_keywords)    # [N]

        # ── optional hard tag filtering ──
        if min_tag_score > 0.0 and self.tag_index is not None and query.all_keywords:
            mask = tag_scores >= min_tag_score       # [N] bool
            if mask.sum() >= top_k:
                # enough candidates: restrict to those
                sem_scores = torch.where(mask, sem_scores, torch.tensor(-1.0))
                tag_scores = torch.where(mask, tag_scores, torch.tensor(0.0))

        # ── min-max normalise tag_scores to [0, 1] (already in [0,1] but ensure) ──
        # Normalise semantic scores to [0, 1] for fair fusion
        sem_min, sem_max = sem_scores.min(), sem_scores.max()
        sem_range = (sem_max - sem_min).clamp(min=1e-8)
        sem_norm = (sem_scores - sem_min) / sem_range

        tag_min, tag_max = tag_scores.min(), tag_scores.max()
        tag_range = (tag_max - tag_min).clamp(min=1e-8)
        tag_norm = (tag_scores - tag_min) / tag_range if tag_max > 0 else tag_scores

        # ── fuse ──
        final_scores = alpha * sem_norm + (1.0 - alpha) * tag_norm  # [N]

        # ── top-k ──
        k = min(top_k, len(self.image_paths))
        top_indices = torch.topk(final_scores, k=k).indices.tolist()

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank":           rank,
                "image_path":     self.image_paths[idx],
                "semantic_score": float(sem_scores[idx]),
                "tag_score":      float(tag_scores[idx]),
                "final_score":    float(final_scores[idx]),
            })

        return results

    def retrieve_from_audio_embedding(
        self,
        audio_emb: torch.Tensor,
        query: "QueryComponents",
        top_k: int = 9,
        alpha: Optional[float] = None,
    ) -> List[Dict]:
        """
        Variant: use a pre-computed audio embedding for the semantic score
        instead of re-encoding the text.  Used in speak2gallery where the
        WhisperAudioEncoder output is already available.

        Parameters
        ----------
        audio_emb : torch.Tensor
            Normalised audio embedding [1, D] or [D].
        query : QueryComponents
            Used only for tag filtering.
        """
        alpha = alpha if alpha is not None else self.alpha

        if audio_emb.dim() == 1:
            audio_emb = audio_emb.unsqueeze(0)
        audio_emb = audio_emb.cpu()

        # semantic scores from audio embedding
        sem_scores = (self.image_embeddings @ audio_emb.T).squeeze(-1)  # [N]

        # tag scores
        tag_scores = self._compute_tag_scores(query.all_keywords)

        # normalise & fuse (same as retrieve())
        sem_min, sem_max = sem_scores.min(), sem_scores.max()
        sem_norm = (sem_scores - sem_min) / (sem_max - sem_min).clamp(min=1e-8)

        tag_max = tag_scores.max()
        tag_norm = (tag_scores / tag_max.clamp(min=1e-8)) if tag_max > 0 else tag_scores

        final_scores = alpha * sem_norm + (1.0 - alpha) * tag_norm

        k = min(top_k, len(self.image_paths))
        top_indices = torch.topk(final_scores, k=k).indices.tolist()

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank":           rank,
                "image_path":     self.image_paths[idx],
                "semantic_score": float(sem_scores[idx]),
                "tag_score":      float(tag_scores[idx]),
                "final_score":    float(final_scores[idx]),
            })

        return results


# ─── CLI smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from query_processor import QueryProcessor

    INDEX_PATH = sys.argv[1] if len(sys.argv) > 1 else "image_index.pt"
    QUERY_TEXT = sys.argv[2] if len(sys.argv) > 2 else "a dog on a surfboard at the beach"

    retriever = HybridRetriever(image_index_path=INDEX_PATH)
    qp        = QueryProcessor(translate=False)
    qc        = qp.process(QUERY_TEXT)

    print("\n[Query Components]")
    print(qc)

    print("\n[Top-9 Results]")
    for r in retriever.retrieve(qc, top_k=9):
        print(f"  #{r['rank']:2d}  sem={r['semantic_score']:.4f}  "
              f"tag={r['tag_score']:.4f}  final={r['final_score']:.4f}  "
              f"{r['image_path']}")
