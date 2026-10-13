# 箱包知识库 RAG · 教学可跑版

面向简历项目「白沟箱包店铺知识库问答」的 **教学可跑版**：解析 → 分块 → 混合检索（BM25 + 稠密）→ RRF 融合 → 精排 → 带引用回答，本地零第三方依赖即可跑通。

> **重要说明（请先读）**  
> 这是「**教学可跑版**」，不是完整生产栈。  
> - **本仓库**：中文字 bigram + 简易 TF 稠密向量 + 内存索引 + 关键词重叠精排，方便面试讲解与本地改参。  
> - **生产环境**：BGE-M3 稠密 Embedding + BM25 稀疏 + **Milvus** 向量库 + **bge-reranker** 精排。  
> - **不变的部分**：分块思路、RRF 融合、评测对比方法、带引用回答与拒答闸门——**融合 / 评测逻辑与线上一致**。

## 面试官 / HR 可以怎么理解

| 简历说法 | 本仓库能演示什么 |
|---------|------------------|
| 全链路自己做过 | 三个脚本覆盖分块 → 四路召回对比 → 带引用问答 |
| BM25 + 向量 + RRF | `rag_core.rrf_fuse`：只看排名 `1/(k+rank)`，不调两路权重 |
| Rerank Top-K | 教学版用 bigram 重叠；生产换 cross-encoder |
| 评测数字 | 小样例数字会与简历 200 条集不同；讲清「同一评测集横向对比四列」即可 |

配套口述结构见 [`面试口述稿.md`](./面试口述稿.md)。

## 环境

- Python **3.10+**
- 默认 **零 pip 依赖**（见 `requirements.txt` 注释）
- 可选：自行接入 `sentence-transformers` / `pymilvus` 等真 Embedding 与向量库

## 快速跑通

```bash
git clone https://github.com/Brucefly007/rag-knowledge-base.git
cd rag-knowledge-base

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt   # 教学版可为空依赖

python src/01_chunk.py
python src/02_retrieve_and_eval.py
python src/03_qa_demo.py
```

### 脚本说明

1. **`01_chunk.py`** — 按 Markdown 标题分块（教学参数偏小；简历生产参数约 chunk=512、overlap=64）。
2. **`02_retrieve_and_eval.py`** — 对比纯 BM25 / 纯稠密 / 混合 RRF / 混合+精排（Recall@K、MRR）。
3. **`03_qa_demo.py`** — 检索 Top 段落后拼成带引用的回答（不调用大模型；有 API Key 时可替换生成部分）。

## 目录结构

```
rag-knowledge-base/
├── README.md
├── LICENSE
├── requirements.txt
├── 面试口述稿.md
├── data/bag_kb.md          # 样例知识库
├── eval/eval_set.json      # 小样例评测集
├── src/
│   ├── rag_core.py         # 分块 / BM25 / DenseTF / RRF / 精排 / 指标
│   ├── 01_chunk.py
│   ├── 02_retrieve_and_eval.py
│   └── 03_qa_demo.py
└── 产出/                   # 可选：自己跑出来的指标截图
```

## 简历参考指标（生产大评测集）

同一套方法在约 200 条人工评测（含约 20% 无答案）上的参考量级：

- Recall@5：纯向量 ≈ 0.62 / BM25 ≈ 0.58 / 混合 ≈ 0.81 / 混合+Rerank ≈ 0.89  
- MRR：约 0.51 → 0.78  

本仓库小样例数字会不同，请以「方法与对比列」为准。

## License

MIT © Brucefly007
