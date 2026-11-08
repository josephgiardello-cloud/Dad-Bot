from __future__ import annotations

import logging
import math
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from threading import RLock
from time import perf_counter

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MARGIN = 0.15
CANDIDATE_LIMIT = 5
logger = logging.getLogger(__name__)
_MODEL_LOCK = RLock()
_MODELS: dict[str, LocalQwenReranker] = {}


class LocalQwenReranker:
    """Offline yes/no relevance scoring using the published Qwen protocol."""

    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Local reranker model directory not found: {model_path}")
        self.torch = torch
        self._lock = RLock()
        torch.set_num_threads(4)
        start = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            padding_side="left",
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            .to("cpu")
            .eval()
        )
        self.load_seconds = perf_counter() - start
        self.prefix = self.tokenizer.encode(
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False,
        )
        self.suffix = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.yes_id == self.no_id or self.tokenizer.unk_token_id in (self.yes_id, self.no_id):
            raise ValueError("Reranker tokenizer must contain distinct yes/no tokens.")

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not query.strip() or not documents or len(documents) > CANDIDATE_LIMIT:
            raise ValueError("Reranker requires a nonempty query and one to five documents.")
        prompts = [
            "<Instruct>: Given a web search query, retrieve relevant passages that answer the query"
            f"\n<Query>: {query}\n<Document>: {document}"
            for document in documents
        ]
        encoded = self.tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)
        tokens = [self.prefix + ids + self.suffix for ids in encoded["input_ids"]]
        if any(len(ids) > 2048 for ids in tokens):
            raise ValueError("Reranker input exceeds 2048 tokens; refusing silent truncation.")
        inputs = self.tokenizer.pad({"input_ids": tokens}, padding=True, return_tensors="pt")
        with self._lock, self.torch.inference_mode():
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            scores = logits[:, [self.no_id, self.yes_id]].softmax(dim=-1)[:, 1].tolist()
        if len(scores) != len(documents) or any(not math.isfinite(score) for score in scores):
            raise ValueError("Invalid reranker score output.")
        logger.info("Local Qwen reranker scored %d candidates.", len(documents))
        return scores


def get_local_reranker(model_path: str) -> LocalQwenReranker:
    if not model_path.strip():
        raise ValueError("Enabled reranker requires DADBOT_QWEN_RERANKER_PATH.")
    path = str(Path(model_path).resolve())
    with _MODEL_LOCK:
        if path not in _MODELS:
            _MODELS[path] = LocalQwenReranker(path)
        return _MODELS[path]


def select_candidates[T](
    ranked: Sequence[tuple[float, T]],
    key: Callable[[T], Hashable],
    *,
    top_score_threshold: float | None = None,
) -> tuple[bool, float, list[tuple[float, T]]]:
    if top_score_threshold is not None and (
        not math.isfinite(top_score_threshold) or not 0 < top_score_threshold <= 1.0
    ):
        raise ValueError("Reranker top-score threshold must be finite and in (0, 1].")
    distinct: list[tuple[float, T]] = []
    seen: set[Hashable] = set()
    for item in ranked:
        identity = key(item[1])
        if identity not in seen:
            distinct.append(item)
            seen.add(identity)
        if len(distinct) == CANDIDATE_LIMIT:
            break
    gap = distinct[0][0] - distinct[1][0] if len(distinct) > 1 else math.inf
    confident = bool(distinct and top_score_threshold is not None and distinct[0][0] >= top_score_threshold)
    return gap < MARGIN and not confident, gap, distinct


def reranked_candidates(candidates, scores):
    if len(candidates) != len(scores) or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Reranker must return one finite probability per candidate.")
    return [
        item
        for _, item in sorted(
            zip(scores, candidates, strict=True),
            key=lambda entry: (-entry[0], -entry[1][0], entry[1][1].get("summary", "")),
        )
    ]
