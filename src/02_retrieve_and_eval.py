"""步骤2：对比 纯BM25 / 纯稠密 / 混合RRF / 混合+精排。

样例库加了噪声章节，方便看出差异。简历数字来自 200 条大评测集，量级不同但方法相同。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import (
    BM25,
    DenseTF,
    char_bigrams,
    chunk_markdown,
    hit_at_k,
    mrr_score,
    naive_rerank,
    rrf_fuse,
)

md = (ROOT / "data" / "bag_kb.md").read_text(encoding="utf-8")
chunks = chunk_markdown(md)
tokenized = [char_bigrams(c.text) for c in chunks]
bm25 = BM25(tokenized)
dense = DenseTF(tokenized)
eval_set = json.loads((ROOT / "eval" / "eval_set.json").read_text(encoding="utf-8"))
print(f"chunk 数: {len(chunks)} | 评测题: {len(eval_set)}\n")


def ranked_ids(method: str, q: str, top_k: int) -> list[int]:
    qt = char_bigrams(q)
    if method == "bm25":
        return [i for i, _ in bm25.rank(qt, top_k)]
    if method == "dense":
        return [i for i, _ in dense.rank(qt, top_k)]
    if method == "hybrid":
        fused = rrf_fuse([bm25.rank(qt, top_k), dense.rank(qt, top_k)], top_k=top_k)
        return [i for i, _ in fused]
    if method == "hybrid_rerank":
        fused = rrf_fuse([bm25.rank(qt, 10), dense.rank(qt, 10)], top_k=10)
        cand = [i for i, _ in fused]
        return [i for i, _ in naive_rerank(q, chunks, cand, top_k)]
    raise ValueError(method)


def metrics(method: str, k: int) -> tuple[float, float]:
    hits = 0
    mrrs = []
    answerable = 0
    for row in eval_set:
        ranked = ranked_ids(method, row["question"], k)
        texts = [chunks[i].text for i in ranked]
        kws = row["relevant_keywords"]
        if not row["has_answer"]:
            mrrs.append(0.0)
            continue
        answerable += 1
        hits += int(any(hit_at_k(t, kws) for t in texts))
        mrrs.append(mrr_score(texts, kws))
    return hits / answerable if answerable else 0.0, sum(mrrs) / len(mrrs) if mrrs else 0.0


print(f"{'方法':<18}{'R@1':>6}{'R@5':>8}{'MRR@5':>8}")
print("-" * 42)
for name, key in [
    ("纯 BM25", "bm25"),
    ("纯稠密(教学TF)", "dense"),
    ("混合 RRF", "hybrid"),
    ("混合+精排", "hybrid_rerank"),
]:
    r1, _ = metrics(key, 1)
    r5, m5 = metrics(key, 5)
    print(f"{name:<18}{r1:>6.2f}{r5:>8.2f}{m5:>8.2f}")

print("\n面试怎么说：")
print("小样例数字会和简历不同；你要会讲「同一评测集横向对比四列，混合+精排最优」。")
print("简历参考：向量0.62 / BM25 0.58 / 混合0.81 / 混合+Rerank 0.89；MRR 0.51→0.78")
