"""步骤3：简易问答 —— 检索 Top 段落后拼成「带引用」的回答（不调用大模型）。

有 API Key 后可把 assemble() 换成真正的 LLM 生成；引用格式保持不变。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import BM25, DenseTF, char_bigrams, chunk_markdown, naive_rerank, rrf_fuse

md = (ROOT / "data" / "bag_kb.md").read_text(encoding="utf-8")
chunks = chunk_markdown(md)
tokenized = [char_bigrams(c.text) for c in chunks]
bm25 = BM25(tokenized)
dense = DenseTF(tokenized)


def answer(question: str) -> None:
    qt = char_bigrams(question)
    fused = rrf_fuse([bm25.rank(qt, 5), dense.rank(qt, 5)], top_k=8)
    cand = [i for i, _ in fused]
    top = naive_rerank(question, chunks, cand, top_k=3)
    print(f"\n问：{question}")
    print("答（摘录拼接，生产由 LLM 基于下列引用生成）：")
    for rank, (i, score) in enumerate(top, 1):
        c = chunks[i]
        print(f"  [{rank}] ({c.id}, rerank={score:.2f}) {c.text.replace(chr(10), ' ')[:120]}")


if __name__ == "__main__":
    demos = ["双肩包能装14寸电脑吗？", "质量问题怎么换货？", "支持分期吗？"]
    for q in demos:
        answer(q)
    print("\n自己试：改 demos 列表后重新运行 python src/03_qa_demo.py，或下一步做成 input()。")
