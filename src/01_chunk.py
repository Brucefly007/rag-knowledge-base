"""步骤1：把知识库 Markdown 按标题分块。先跑这个看分块结果。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import chunk_markdown

md = (ROOT / "data" / "bag_kb.md").read_text(encoding="utf-8")
chunks = chunk_markdown(md)
print(f"共 {len(chunks)} 个 chunk（教学参数偏小；生产常用参数 chunk=512 overlap=64）\n")
for c in chunks:
    preview = c.text.replace("\n", " ")[:80]
    print(f"- {c.id} | {c.heading} | {preview}...")
