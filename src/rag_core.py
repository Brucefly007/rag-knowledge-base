"""教学用 RAG 核心：分块 / BM25 / 简易稠密 / RRF / 评测。

面试要点：生产环境把「简易 TF 稠密」换成 BGE-M3，「内存索引」换成 Milvus，
「关键词重叠精排」换成 bge-reranker；融合与评测逻辑不变。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Chunk:
    id: str
    text: str
    heading: str


def char_bigrams(text: str) -> list[str]:
    s = re.sub(r"\s+", "", text.lower())
    if len(s) < 2:
        return [s] if s else []
    return [s[i : i + 2] for i in range(len(s) - 1)]


def chunk_markdown(md: str, max_chars: int = 180, overlap: int = 40) -> list[Chunk]:
    """标题层级分块；小文档用较短 chunk，方便本地演示。简历写 512/64 是生产参数。"""
    chunks: list[Chunk] = []
    heading = "根"
    buf: list[str] = []

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        if not text:
            buf = []
            return
        # 滑窗切分
        start = 0
        part = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            piece = text[start:end].strip()
            if piece:
                cid = f"c{len(chunks):03d}"
                chunks.append(Chunk(id=cid, text=f"【{heading}】\n{piece}", heading=heading))
                part += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        buf = []

    for line in md.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip() or heading
            continue
        buf.append(line)
    flush()
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.df: Counter[str] = Counter()
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        for d in docs:
            for t in set(d):
                self.df[t] += 1

    def score(self, query: list[str], idx: int) -> float:
        dl = self.doc_len[idx]
        freqs = Counter(self.docs[idx])
        s = 0.0
        for t in query:
            if t not in freqs:
                continue
            n = self.df[t]
            idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            tf = freqs[t]
            s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def rank(self, query: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DenseTF:
    """极简字 bigram TF 向量，演示「稠密召回」位置。生产换 Embedding 模型。"""

    def __init__(self, docs: list[list[str]]):
        self.docs = docs
        vocab = sorted({t for d in docs for t in d})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.mat = [self._vec(d) for d in docs]

    def _vec(self, tokens: list[str]) -> list[float]:
        v = [0.0] * len(self.vocab)
        if not tokens:
            return v
        c = Counter(tokens)
        n = len(tokens)
        for t, f in c.items():
            if t in self.vocab:
                v[self.vocab[t]] = f / n
        return v

    @staticmethod
    def cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def rank(self, query: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        qv = self._vec(query)
        scores = [(i, self.cos(qv, self.mat[i])) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def rrf_fuse(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：只看排名。面试必讲。"""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def naive_rerank(query: str, chunks: list[Chunk], candidates: list[int], top_k: int = 5) -> list[tuple[int, float]]:
    """教学用精排：查询字 bigram 与 chunk 重叠率。生产换 cross-encoder。"""
    q = set(char_bigrams(query))
    scored = []
    for i in candidates:
        d = set(char_bigrams(chunks[i].text))
        overlap = len(q & d) / (len(q) + 1e-9)
        scored.append((i, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hit_at_k(chunk_text: str, keywords: Iterable[str]) -> bool:
    if not keywords:
        return False
    return any(kw.lower() in chunk_text.lower() for kw in keywords)


def mrr_score(ranked_texts: list[str], keywords: list[str]) -> float:
    for i, t in enumerate(ranked_texts, start=1):
        if hit_at_k(t, keywords):
            return 1.0 / i
    return 0.0
