# 面试报告模板 — 面试官使用手册

> 本文件在 Phase 3 报告生成时读取。包含：①完整报告 Markdown 模板 ②14 道题预置参考答案 ③评分细则。

---

## 一、报告 Markdown 模板

生成规则：
- 表格"参考答案"列使用 `[→ 查看](#a-锚点关键词)` 锚链接，指向本文件第二节对应答案
- 只为**本次实际问到的题目**复制对应答案块，未问到的不放入报告
- 严格按评分细则打分，不得因情绪照顾调分

```markdown
# 模拟面试报告

**项目**：RAG Framework
**面试时间**：{datetime}
**评分**：{score}/10

---

## 一、面试记录

> ✅ 答对核心要点 | ⚠️ 方向正确但细节缺失 | ❌ 未能答出或方向错误

### 方向 1：项目综述

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 参考答案 |
|-----|------|-------------|------|---------|
| 1 | {问题原文} | {2-3 句摘要} | ✅/⚠️/❌ | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | [→ 查看](#a-{锚点}) |

### 方向 2：简历深挖

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 露馅 | 参考答案 |
|-----|------|-------------|------|-----|---------|
| 1 | {问题原文} | {摘要} | ✅/⚠️/❌ | 是/否 | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | ... | [→ 查看](#a-{锚点}) |

### 方向 3：技术深挖

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 参考答案 |
|-----|------|-------------|------|---------|
| 1 | {问题原文} | {摘要} | ✅/⚠️/❌ | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | [→ 查看](#a-{锚点}) |

---

## 二、参考答案

> 仅复制本次实际问到的题目对应答案块，保留 <a id> 锚点。

{从下方"预置参考答案库"按需复制}

---

## 三、简历包装点评

### 包装合理 ✅
- **"{简历描述}"**：{说明候选人能自圆其说之处，具体指出哪句回答支撑了该判断}

### 露馅点 ❌
- **"{简历描述}"** → {面试中的具体表现}。**严重性：高/中/低**（{说明原因}）

### 改进建议
- {针对每个露馅点的具体、可操作建议，如"建议记住 RRF 公式并搞清楚 k 值是否可配置"}

---

## 四、综合评价

**优势**：
- {具体到哪道题答得好、好在哪个关键点}

**薄弱点**：
- {具体技术点 + 答错/答浅的表现描述}

**面试官建议**：
{针对每个薄弱点的具体改进方向，避免笼统表述}

---

## 五、评分

| 维度 | 分数（满分 10）| 评分依据（必须说明具体扣分原因） |
|-----|--------------|--------------------------------|
| 项目架构掌握 | x | {哪些点答到了，哪些点缺失} |
| 简历真实性 | x | {几处包装合理，几处露馅，差距} |
| 算法理论深度 | x | {RRF/Cross-Encoder/评估指标等作答情况} |
| 实现细节掌握 | x | {chunk_id/可插拔注册机制/MCP Transport/跨存储删除等} |
| 表达清晰度 | x | {回答完整性、逻辑清晰度、因果说明} |
| **综合** | **x** | {加权说明} |
```

---

## 二、预置参考答案库

> 按需复制到报告"二、参考答案"节，保留 `<a id>` 锚点不变。

---

### <a id="a-项目架构"></a>Q: 介绍项目整体架构和你具体负责的部分

**参考答案**：
系统分三大层次：
1. **Ingestion 层**：Loader → Sniffer → Router → Parser → Quality Gate → Cleaner → Block Splitter → Chunk Packer →（可选）Embedder →（可选）BM25/FAISS 索引写入 → DocStore + TraceStore
2. **检索层**：Hybrid Search（BM25 + FAISS 并行召回）→ RRF 融合 → Cross-Encoder / Voyage 精排
3. **服务层**：基于 FastMCP 的 MCP Server，默认 stdio Transport，对外暴露 7 个工具，AI Client 可直接调用

核心亮点：全链路可插拔（字典注册表 + 惰性导入，`"module.path:ClassName"` 格式），配置文件驱动零代码切换后端。

---

### <a id="a-ingestion链路"></a>Q: Ingestion 链路有哪些阶段？

**参考答案**：
真实阶段（`rag/pipelines/ingest_pipeline.py`）：
```
Loader → Sniffer → Router → Parser → Quality Gate → Cleaner
  → Block Splitter → Chunk Packer
  → (可选) Embedder → (可选) BM25 + FAISS 索引写入
  → DocStore + TraceStore
```
切分是纯机械的：`ParagraphBlockSplitter`（段落切分）+ `AnchorAwareChunkPacker`（锚点感知打包，按 token 预算装箱），**没有 LLM 驱动的 chunk 转换/元数据生成步骤**（虽然代码里有个可选的 `MetadataEnricherLLMBatch` 类，但不在默认流水线里）。

---

### <a id="a-mcp协议"></a>Q: MCP 是什么规范？暴露了哪些 Tool？

**参考答案**：
MCP（Model Context Protocol）是 Anthropic 提出的开放协议，定义 AI Client 与外部工具/数据源之间的标准通信接口。项目用官方 `mcp.server.fastmcp.FastMCP` 封装。

代码里 `mcp.run()` **没有传 transport 参数**，因此走 FastMCP 默认的 **stdio Transport**。注意 `docker-compose.yml` 给这个服务映射了 `8000:8000` 端口，但 stdio 模式根本不监听端口，这个映射目前**不对应任何实际监听的服务**——如果要真正支持远程/HTTP 访问，需要显式改成 `mcp.run(transport="sse")` 之类的配置。

对外暴露 **7 个** Tool：`rag.ingest` / `rag.query` / `retrieve` / `retrieve_with_metadata` / `list_collections` / `rag.eval.run` / `rag.sync_source`。

---

### <a id="a-rrf公式"></a>Q: RRF 融合公式是什么？k 值含义？为什么不用线性加权？

**参考答案**：

$$Score_{RRF}(d) = \sum_{i} \frac{1}{k + rank_i(d)}$$

- **实现位置**：`rag/infra/indexes/rrf_fusion.py`，`_DEFAULT_K = 60`
- **可配置**：通过 `configs/settings.yaml` 的 `retrieval.rrf_k` 覆盖默认值，不需要改代码
- **k 的含义**：平滑因子，防止排名头部文档分数被过度高估。k=60 来自 Cormack et al. 2009 的经验推荐值
- **为什么不用线性加权**：BM25 分数无上界，FAISS 这边用的是负 L2 距离，两路量纲完全不同，线性加权必须先归一化且引入额外超参。RRF 只依赖排名（序数信息），天然无需归一化

---

### <a id="a-cross-encoder"></a>Q: Cross-Encoder 和 Bi-Encoder 的区别？为什么不能做粗排召回？

**参考答案**：

| | Bi-Encoder | Cross-Encoder |
|--|-----------|--------------|
| 编码方式 | Query 和 Document **分别**编码为向量，算相似度 | Query 和 Document **拼接**一起输入模型，联合建模 |
| Document 向量 | 可**离线预计算**，查询时 O(1) | 每对 (Query, Chunk) 必须**实时推理**，O(n) |
| 精度 | 较低（无交互） | 更高（充分建模交互特征） |
| 适合场景 | 粗排召回（大规模，项目里对应 FAISS） | 精排（小候选集，项目里对应 `crossencoder_reranker.py`） |

项目里精排用的是 `cross-encoder/ms-marco-MiniLM-L-6-v2`（sentence-transformers），另有 Voyage `rerank-2` 作为云端替代方案，两者都通过 reranker 注册表可插拔切换。

---

### <a id="a-无rerank降级"></a>Q: 重排失败时系统有没有自动降级到 RRF 排名？

**参考答案**：
**没有**。重排在 `query_pipeline.py`（约第 375 行）里是直接调用，没有 try-except 包裹，也没有超时机制。如果 Cross-Encoder 或 Voyage 调用失败/超时，异常会直接向上抛出，**不会自动回退到 RRF 融合排名**。`plugin_registry.py` 里的错误处理只覆盖构造期（比如配置了不存在的 reranker 名字），不覆盖运行时调用失败。这是当前实现的一个真实设计缺口，如果简历上写"精排失败自动降级"，需要重点验证候选人是否清楚这一点不存在。

---

### <a id="a-recall-at-k"></a>Q: Recall@K 是怎么计算的？

**参考答案**：

$$Recall@K = \frac{\text{Top-K 结果中命中相关文档的查询数}}{\text{总查询数}}$$

实现在 `rag/pipelines/scoring/metrics.py` 的 `recall_at_k()`。对 Resume Gold Eval 中每条 `(query, expected_chunks)`，取 Top-K 检索结果，命中至少一条则计 1，否则计 0。项目里这个术语用的是 **Recall@K**，而不是"Hit Rate@K"（两者计算方式一致，但项目代码和 README 里统一叫 Recall@K）。

---

### <a id="a-可插拔架构"></a>Q: 新增一个 Embedding Provider 需要改哪些文件？

**参考答案**：
只需改 **2-3 处**，已有代码零修改：

1. **新建**模块，实现类继承 `BaseEmbeddingProvider`
2. **修改** `rag/infra/embedding/embedding_factory.py`：在 `_PROVIDER_REGISTRY` 里调用 `register_provider("your_provider", "your.module.path:YourClass")`——注意注册的是**字符串路径**而不是类对象本身，真正用到时才惰性 `importlib.import_module()` 导入，避免没装的第三方依赖在启动时就报错
3. **修改** `configs/settings.yaml`：把 `embedding.provider` 改成 `"your_provider"`

Reranker 走的是同一套模式，在 `rag/core/registry/plugin_registry.py` 的 `_RERANKER_REGISTRY` 里注册。

---

### <a id="a-幂等性"></a>Q: chunk_id 是怎么生成的？为什么不用 UUID？

**参考答案**：
`chunk_id = chunk.chunk_signature`——SHA-256 哈希，作用于**该 chunk 所含全部文本块（block）哈希的有序拼接**（`rag/core/contracts/chunk.py` 第 19 行），不是简单对 chunk 全文内容做一次哈希。

**为什么不用 UUID**：UUID 随机，重复摄取同一文件会产生新 ID，导致索引里出现重复 Chunk。确定性哈希保证相同内容永远生成相同 ID，天然支持幂等 Upsert。

去重实际是两层机制：① 文件级——`fingerprint_bytes()` 生成文件字节指纹 + `make_doc_id()`，DocStore 里已存在同一 doc_id 直接跳过整份文档；② 块级——`diff_blocks()` 比较新旧 block 哈希，只有变化的块才重新走 chunk/embed。

---

### <a id="a-ocr"></a>Q: 项目怎么处理扫描版 PDF？和普通 PDF 是同一条解析路径吗？

**参考答案**：
不是同一条路径。默认的 `PdfPyMuPDFParser` 是结构化解析，读取 PDF 自带的文本层，遇到扫描版（没有文本层）解析效果会很差或为空。

针对扫描版单独有 OCR 通路：`rag/infra/ocr/renderer_pymupdf.py` 用 PyMuPDF 把每页渲染成图像，再用 `rag/infra/ocr/paddleocr_provider.py`（PaddleOCR）识别文字，产出带页码、Bounding Box、置信度的结构化 `IRBlock`。项目**没有**基于 Vision LLM 的图片语义描述/多模态检索能力，OCR 只是把图像里的文字变成可检索文本。

---

### <a id="a-测试体系"></a>Q: 测试分几层？单元测试怎么 mock LLM？

**参考答案**：
pytest 用显式 marker 区分：`@pytest.mark.unit`（48 个）/ `@pytest.mark.integration`（6 个）/ `@pytest.mark.e2e`（4 个），其余绝大多数测试未打标记；`pytest --collect-only` 实测总数约 1112 条。

Mock 策略用标准库 `unittest.mock`：例如 `tests/test_query_pipeline_generation.py` 直接 mock 出一个返回预设 `LLMResponse`（含假 token 用量）的 LLM 客户端，避免真实 API 调用和费用。

---

### <a id="a-评估体系"></a>Q: Recall@K 和 MRR 怎么计算？RAGAS Faithfulness 衡量什么？

**参考答案**：
- **Recall@K**：见 [→ 查看](#a-recall-at-k)
- **MRR**：核心是 `reciprocal_rank()`（`rag/pipelines/scoring/metrics.py`），每条查询里第一条命中结果排名的倒数，取所有查询平均。第一条命中得 1 分，第 2 位命中得 0.5 分，衡量**头部排序质量**。
- **RAGAS Faithfulness**：衡量 LLM 生成的回答是否**完全基于检索到的 Context**，防止幻觉，衡量的是"答案质量"而不是"检索质量"——这一点和 Recall@K/MRR 有本质区别，Recall/MRR 只看检索排名对不对，Faithfulness 看最终生成的答案有没有编造内容。

项目真实评估数据：Resume Gold Eval（30 条人工标注）从 BM25-only 的 Recall@10 0.923 提升到 Hybrid+Rerank 的 0.964；BEIR SciFact 公开基准 nDCG@10 达 0.703，超过已发表 ColBERT baseline（0.671）。

---

### <a id="a-跨存储删除"></a>Q: 删除一个文档，DocStore 和检索索引是怎么保持一致的？

**参考答案**：
分两层，**不是同一套机制**：
1. **DocStore 层**（`rag/infra/stores/docstore_sqlite.py` 第 229 行 `delete_document()`）：一条 SQL DELETE 删 documents 表，靠 `ON DELETE CASCADE` 外键约束**自动级联**删除关联的 text_blocks 和 chunks——这是数据库原生行为
2. **索引层**（`ingest_pipeline.py` 第 467-480 行）：BM25 和 FAISS 独立于 SQLite，不会跟着级联，需要单独循环遍历 `prev_chunk_ids`，逐个调用 `keyword_index.remove(cid)` 和 `vector_index.remove(cid)`，删完再显式调用 `.save()` 落盘

**风险点**：这个循环没有事务保护，如果 BM25 删成功、FAISS 删到一半抛异常，两个索引会出现不一致状态——这是当前实现的真实局限，不是设计上的"故意为之"。

---

### <a id="a-可观测性"></a>Q: Trace 是怎么实现的？Query 链路的阶段有哪些？

**参考答案**：
**存储**：SQLite（`rag/infra/stores/tracestore_sqlite.py`），不是 JSON Lines。两张表：`runs`（run_id/run_type/metadata_json/created_at）+ `trace_events`（event_id/run_id/event_type/data_json/created_at）。

**Query 链路阶段**：`query`（开始）→ `query_pre_rerank`（重排前，即 RRF 融合结果）→ `query_post_rerank`（重排后）→ `query_generation`（LLM 生成阶段）→ `query_complete`（结束）——这个设计能清楚对比"重排前后排名到底变了没有"。

**Ingest 链路阶段**：`ingest`（开始）→（可能触发 `resync_threshold_exceeded` 告警）→ `ingest_complete`（结束）。

---

## 三、评分细则

**分档标准（严格执行，不得调整）**：

| 分档 | 标准 |
|-----|------|
| 9-10 | 所有核心问题答出关键细节，无露馅，表达清晰且有深度延伸 |
| 7-8 | 大部分问题答出主干，偶有细节遗漏（1-2 处），无严重露馅 |
| 5-6 | 架构层面基本掌握，但算法/实现细节有 3 处以上明显缺失，或有 1 处严重露馅 |
| 3-4 | 仅能描述表面概念，追问即露馅，简历存在明显虚报 |
| 1-2 | 核心技术点均无法解释，简历与实际能力严重不符 |

**5 个评分维度**：

| 维度 | 重点考察内容 |
|-----|------------|
| 项目架构掌握 | 三层架构、模块分工、可插拔设计能否清楚表达 |
| 简历真实性 | 量化指标有无测量方法支撑，强动词能否说清决策过程 |
| 算法理论深度 | RRF 公式、Cross-Encoder 原理、Recall@K/MRR 计算 |
| 实现细节掌握 | chunk_id 生成、可插拔注册机制、MCP Transport、跨存储删除的两套机制 |
| 表达清晰度 | 回答完整性、逻辑链完整、能说清"为什么"而非只说"是什么" |
