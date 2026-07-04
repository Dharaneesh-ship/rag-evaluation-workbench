# 项目技术知识库 — 面试官参考手册

> 本文件供面试官（AI Agent）使用，包含本项目（RAG Framework）的关键实现细节、高频面试题及参考答案。
> 所有细节均已对照源码核实（文件路径见各节），面试过程中用于生成精准追问和评估候选人回答质量。

---

## 模块一：Hybrid Search 混合检索

### 核心实现
- **双路并行召回**：Dense（FAISS 向量检索）+ Sparse（BM25 关键词匹配）
- **融合算法**：RRF（Reciprocal Rank Fusion），实现于 `rag/infra/indexes/rrf_fusion.py`
  公式：`Score = Σ 1/(k + rank_i)`，rank 为 1-based 排名
  `_DEFAULT_K = 60`（第 10 行），**可通过 `configs/settings.yaml` 的 `retrieval.rrf_k` 覆盖**
  并列分数按 chunk_id 字典序打破平局
- **Dense Route（FAISS）**：`rag/infra/indexes/faiss_local.py`
  - 索引类型是 `IndexFlatL2`（暴力扫描，非 IVF/HNSW，不做近似检索）
  - 相似度用**负 L2 距离**表示"越大越相似"（代码里是 `float(-dist)`），不是 Cosine
  - 持久化为两个文件：`faiss.index`（FAISS 原生索引）+ `faiss_mapping.pkl`（id 顺序、chunk、向量的映射）
  - collection 过滤采用 **overscan 策略**：先取 `top_k * 5` 条候选再按 collection 过滤，避免小 collection 被淹没
- **Sparse Route（BM25）**：`rag/infra/indexes/bm25_local.py`
  - 基于 `rank_bm25.BM25Okapi`，**b/k1 参数未暴露，用库默认值**，没有调参入口
  - 中文分词用 **jieba**，jieba 未安装时降级为逐字符切分
  - 持久化为 `bm25.pkl`（pickle 序列化）

### 高频面试题

**Q: 为什么要做 Hybrid Search？BM25 和向量检索各有什么优劣？**
A: BM25（稀疏检索）擅长精确关键词匹配，对专有名词效果好；FAISS 稠密检索擅长语义理解，处理同义词、模糊表达时优势明显。两者互补，用 RRF 融合，平衡 Precision 和 Recall。

**Q: RRF 公式里 k=60 是怎么来的？能改吗？**
A: k=60 是代码里的 `_DEFAULT_K`（`rag/infra/indexes/rrf_fusion.py` 第 10 行），来自 Cormack et al. 2009 的经验推荐值。可以改——通过 `configs/settings.yaml` 的 `retrieval.rrf_k` 覆盖默认值，不需要改代码。

**Q: 你们的 FAISS 索引是什么类型？为什么不用 IVF 或 HNSW？**
A: 用的是 `IndexFlatL2`，即暴力扫描（brute-force），不做近似最近邻检索。相似度计算是负 L2 距离取反表示"越大越相似"，不是 Cosine Similarity。这个选择在语料规模不大时召回质量最高（精确检索，无近似误差），但规模上去后（百万级向量）会有明显的线性扫描延迟问题，届时需要换 IVF/HNSW 做近似检索换速度。

**Q: BM25 的 b 和 k1 参数你们调过吗？**
A: 没有——用的是 `rank_bm25.BM25Okapi` 库的默认值，代码里没有暴露这两个参数的配置入口。这是当前实现的一个已知局限，如果要调优需要直接改 `bm25_local.py` 里 `BM25Okapi()` 的实例化参数。

---

## 模块二：Reranker 精排

### 核心实现
- **支持两种真实后端**：
  1. `Cross-Encoder`：`rag/infra/rerank/crossencoder_reranker.py`，模型是 `cross-encoder/ms-marco-MiniLM-L-6-v2`（sentence-transformers）
  2. `Voyage Rerank`：`rag/infra/rerank/voyage_rerank.py`，调用 Voyage API，模型是 `rerank-2`
  3. `Noop`：`rag/infra/rerank/noop.py`，不重排，直接透传 RRF 融合结果
- **⚠️ 没有超时+自动回退机制**：重排在 `query_pipeline.py`（约第 375 行）里直接调用，**没有 try-catch 包裹、没有超时包装**。如果 Cross-Encoder 或 Voyage 调用失败/超时，异常直接向上抛出，**不会自动降级回 RRF 排名**。Registry 层（`plugin_registry.py` 第 74-84 行）只处理构造期的错误（比如配置了不存在的 reranker 名字），不处理运行时调用失败。这是当前实现的一个真实设计缺口，可作为"你们有没有做 Graceful Fallback"这类问题的深挖点。

### 高频面试题

**Q: Cross-Encoder 和 Bi-Encoder 的区别？为什么 Cross-Encoder 不能做粗排召回？**
A: Bi-Encoder（如 FAISS 用的 Embedding 模型）将 Query 和 Document 分别编码为向量，可离线预计算 Document 向量，查询时 O(1)。Cross-Encoder 将 Query 和 Document 拼接后一起输入模型，能捕捉交互特征，精度更高，但必须对每对 (Query, Chunk) 实时推理，只适合对小候选集精排。

**Q: 精排阶段用的什么模型？如果重排服务挂了会怎样？**
A: Cross-Encoder 用 `cross-encoder/ms-marco-MiniLM-L-6-v2`，Voyage 用 `rerank-2`。如果调用失败——**目前没有降级机制**，异常会直接抛出到调用方，不会自动回退到 RRF 融合排名。这是需要改进的地方：应该在 query_pipeline 里包一层 try-except，失败时 fallback 到 RRF 结果，保证系统可用性。

---

## 模块三：数据摄取流水线（Ingestion Pipeline）

### 真实阶段（`rag/pipelines/ingest_pipeline.py`）
```
Loader → Sniffer → Router → Parser → Quality Gate → Cleaner
  → Block Splitter → Chunk Packer
  → (可选) Embedder → (可选) BM25 + FAISS 索引写入
  → DocStore + TraceStore
```

**⚠️ 没有 LLM 驱动的 Chunk 转换步骤**：切分是纯机械的——`ParagraphBlockSplitter`（段落切分）+ `AnchorAwareChunkPacker`（锚点感知打包，按 token 预算装箱）。**没有** ChunkRefiner / MetadataEnricher / ImageCaptioner 这类 LLM 增强步骤。代码里存在一个可选的 `MetadataEnricherLLMBatch` 类，但**不在默认流水线里**，需要显式启用。

### 幂等性设计
- **chunk_id 生成**：`chunk_id = chunk.chunk_signature`——SHA-256 哈希，作用于**该 chunk 包含的所有文本块（block）哈希的有序拼接**，不是简单对 chunk 全文内容做哈希（`rag/core/contracts/chunk.py` 第 19 行）
- **文件级去重**：`fingerprint_bytes(artifact.raw_bytes)` 生成文件指纹 + `make_doc_id(canonical, content_hash)` 生成 doc_id，若 DocStore 里已存在同一 doc_id 则整个文档跳过（`ingest_pipeline.py` 第 338-350 行）
- **块级差量**：`diff_blocks()` 比较新旧文本块的哈希列表，只有变化的块会重新走 chunk/embed 流程（第 404-420 行），这一层和文件级去重是**两套独立机制**，分别解决"整份文件没变"和"文件变了但只改了一部分"两种场景

### 高频面试题

**Q: chunk_id 是怎么生成的？为什么不用 UUID？**
A: `chunk_id = chunk_signature`，是对该 chunk 所含全部 block 哈希做有序拼接后取 SHA-256。不用 UUID 是因为 UUID 随机，重复摄取同一文件会产生新 ID 导致索引里出现重复 chunk；确定性哈希保证相同内容永远得到相同 ID，天然支持幂等 Upsert。

**Q: 如果只改了文档里的一段内容，会重新处理整个文档吗？**
A: 不会。有两层去重/差量机制：① 文件级——整份文件的字节指纹如果没变（`fingerprint_bytes`），直接跳过；② 块级——`diff_blocks()` 比较新旧文本块哈希，只有真正变化的块会被重新切分、打包、embed，没变的块复用已有 chunk。

**Q: Ingestion 里有没有用 LLM 做 chunk 优化或元数据生成？**
A: 默认流水线里没有。切分是纯机械的段落切分 + 锚点感知打包，没有类似 ChunkRefiner 那种用 LLM 合并/去噪段落的步骤。代码里确实有一个 `MetadataEnricherLLMBatch` 类可以选用，但不在默认路径里，需要显式启用才会跑。

---

## 模块四：可插拔架构

### 核心设计
- **注册机制**：字典注册表 + 惰性导入，格式是 `"module.path:ClassName"` 字符串（不是 import 好的类对象），需要时才 `importlib.import_module()` + `getattr()`
- **Embedding**：`rag/infra/embedding/embedding_factory.py`，`_PROVIDER_REGISTRY` 字典，内置 `"openai"` / `"multilingual"`，`register_provider(name, class_path)` 用于注册新 provider
- **Reranker**：`rag/core/registry/plugin_registry.py`，`_RERANKER_REGISTRY` 字典，`register_reranker(name, cls)` 注册，`build_reranker(config)` 按配置构造实例，内置 provider 通过 `_ensure_builtins_registered()` 惰性注册

### 新增 Provider 流程（面试经典追问）
以新增 Embedding Provider 为例：
1. 新建模块实现类，继承 `BaseEmbeddingProvider`
2. 在 `embedding_factory.py` 的 `_PROVIDER_REGISTRY` 里加一行 `register_provider("your_provider", "your.module.path:YourClass")`
3. `configs/settings.yaml` 里把 `embedding.provider` 改成 `"your_provider"`
其他组件（Reranker 等）遵循同一套"注册表 + 惰性导入"模式。

### 高频面试题

**Q: 可插拔架构是怎么实现的？新增一个 Provider 要改哪些文件？**
A: 核心是字典注册表 + 惰性导入——`_PROVIDER_REGISTRY` 存的是 `"module.path:ClassName"` 字符串而不是导入好的类，真正用到时才 `importlib.import_module` 动态加载，避免不必要的依赖（比如没装 Voyage SDK 也不会在启动时报错）。新增 provider 只需：写新类继承 Base 接口 → 调 `register_provider()` 注册一行 → 改配置文件里的 provider 名字，不需要改调用方代码。

**Q: 为什么用字符串 "module.path:ClassName" 而不是直接注册类对象？**
A: 惰性导入——如果某个 provider 依赖的第三方库没装（比如没装 Voyage SDK），只要不实际使用这个 provider，注册表里放着一条不会被 import 的字符串就不会报错。如果直接 `register_provider("voyage", VoyageClass)`，意味着模块加载时就要 import VoyageClass 所在文件，没装依赖会直接启动失败。

---

## 模块五：MCP 协议集成

### 核心实现
- **框架**：`mcp.server.fastmcp.FastMCP`（官方 MCP Python SDK 的高层封装），实例化时 `name="rag-framework"`
- **暴露的 7 个工具**：`rag.ingest` / `rag.query` / `retrieve` / `retrieve_with_metadata` / `list_collections` / `rag.eval.run` / `rag.sync_source`
- **启动方式**：`if __name__ == "__main__": mcp.run()`，**没有传 transport 参数**，FastMCP 默认走 **stdio transport**
- **⚠️ docker-compose 里有个不一致点**：`docker-compose.yml` 给 mcp 服务映射了 `"8000:8000"` 端口，但代码里 `mcp.run()` 走的是 stdio（标准输入输出），根本不监听 8000 端口——这个端口映射目前是**摆设，不对应任何实际监听的 HTTP 服务**。如果要让 MCP Server 真正支持 HTTP/SSE 访问，需要改成 `mcp.run(transport="sse")` 或类似配置，而不是现在这样直接 stdio 运行。这是一个真实存在的、可以拿来考"你是否真的读过自己项目代码"的题。
- **Collection 隔离**：不是分开的索引，是**元数据过滤**——`RetrieveInput` schema 里有 `collection: Optional[str]`，FAISS/BM25 各自在候选集里按 `metadata["collection"]` 过滤；FAISS 用 overscan（先多取几倍候选再过滤）避免小 collection 被淹没，过滤后无匹配时会记录 warning 并回退成不过滤的结果

### 高频面试题

**Q: MCP Server 用的是什么 Transport？为什么 docker-compose 里还映射了端口？**
A: 代码里 `mcp.run()` 没传参数，FastMCP 默认用 stdio transport（标准输入输出通信）。但 `docker-compose.yml` 给这个服务映射了 8000 端口——这其实是不一致的，因为 stdio 模式根本不监听端口，这个端口映射目前不对应任何实际服务。如果想真正通过 HTTP 访问，需要显式配置 `mcp.run(transport="sse")` 之类的选项。

**Q: collection 参数是怎么隔离检索范围的？是每个 collection 单独建索引吗？**
A: 不是单独索引，是**元数据过滤**。FAISS 和 BM25 各自还是全量索引，检索时按 `metadata["collection"]` 字段过滤候选结果。FAISS 那边为了避免小 collection 被过滤掉太多有效候选，用了 overscan 策略——先取 `top_k * 5` 条再过滤。如果过滤后一条都没有，会记录 warning 并退回不带 collection 过滤的结果，保证至少有响应。

---

## 模块六：扫描文档 OCR 处理

### 核心实现
- 默认 PDF 解析器 `PdfPyMuPDFParser` 是**结构化解析**（读取 PDF 里已有的文本层），**不做 OCR**
- 针对扫描版 PDF（没有文本层，只有图像），单独有一条 OCR 通路：`rag/infra/ocr/renderer_pymupdf.py`（PyMuPDF 把页面渲染成图像）+ `rag/infra/ocr/paddleocr_provider.py`（PaddleOCR 识别文字）
- 输出为 `IRBlock`，包含页码、Bounding Box、单块置信度
- **没有** Vision LLM 图片 Caption 生成、没有"搜文出图"这类多模态检索能力

### 高频面试题

**Q: 项目支持扫描版 PDF 吗？和普通 PDF 解析走的是同一条路径吗？**
A: 不是同一条路径。默认的 `PdfPyMuPDFParser` 是结构化解析，读 PDF 自带的文本层，遇到扫描版（没有文本层）会解析出空内容或效果很差。针对这种情况有单独的 OCR 通路：先用 PyMuPDF 把每一页渲染成图片，再用 PaddleOCR 识别文字，输出带页码、坐标框、置信度的结构化文本块。

**Q: 项目支持图片语义检索吗（比如通过文字描述搜到图）？**
A: 不支持。项目里没有 Vision LLM 生成图片 Caption 的机制，OCR 只是把扫描版 PDF 里的文字提取出来变成可检索文本，本质还是文本检索，不是多模态图像检索。

---

## 模块七：可观测性与追踪系统

### Trace 体系
- **存储**：SQLite（`rag/infra/stores/tracestore_sqlite.py`），**不是 JSON Lines**。两张表：`runs`（run_id/run_type/metadata_json/created_at）+ `trace_events`（event_id/run_id/event_type/data_json/created_at）
- **Query 链路阶段**：`query`（开始）→ `query_pre_rerank`（重排前）→ `query_post_rerank`（重排后）→ `query_generation`（LLM 生成阶段）→ `query_complete`（结束）
- **Ingest 链路阶段**：`ingest`（开始）→（可能触发 `resync_threshold_exceeded` 告警/错误）→ `ingest_complete`（结束）
- **Dashboard**：Streamlit Studio 4 个页面（Ingest & Inspect / Query Traces / Evaluation Panel / Connector Sync）

### 高频面试题

**Q: Trace 数据存在哪？为什么不用 LangSmith/LangFuse 这类现成工具？**
A: 存在本地 SQLite 里，`runs` 表记录每次调用的元信息，`trace_events` 表记录该次调用各阶段的事件详情。用 SQLite 而不是外部服务，是为了保持 Local-First、零外部依赖，`pip install` 装完就能跑，不需要额外账号/网络依赖。

**Q: Query 链路的 Trace 阶段具体有哪些？**
A: `query`（请求进来）→ `query_pre_rerank`（重排前的候选，即 RRF 融合结果）→ `query_post_rerank`（重排后的结果，可以对比重排前后的排名变化）→ `query_generation`（LLM 生成答案阶段）→ `query_complete`（整个请求结束）。这个链路设计能让 Dashboard 清楚展示"重排到底起没起作用"。

---

## 模块八：评估体系

### 指标体系（`rag/pipelines/scoring/metrics.py`）
- `recall_at_k()`：Top-K 结果中命中相关文档的比例（注意：项目里用的术语是 **Recall@K**，不是 Hit Rate@K）
- `reciprocal_rank()`：即 MRR 的核心计算，第一条命中结果排名的倒数
- `dcg_at_k()` / `ideal_dcg_at_k()` / `ndcg_at_k()`：nDCG 相关计算链路
- 真实评估结果（README 第 10 节）：Resume Gold Eval 基准（30 条人工标注）BM25 only → Recall@10 0.923 / MRR 0.615 / nDCG@10 0.688；Hybrid + Rerank + collection scope → Recall@10 **0.964** / MRR **0.622** / nDCG@10 **0.708**
- **BEIR SciFact 公开基准**：nDCG@10 达到 **0.703**，超过已发表的 ColBERT baseline（0.671）
- **RAGAS**：Faithfulness / Answer Relevancy / Context Precision，用于评估答案质量而非检索质量

### 高频面试题

**Q: Recall@K 和 MRR 分别怎么算？**
A: `recall_at_k(retrieved, relevant, k)`——Top-K 检索结果里只要出现至少一条相关文档就算命中，命中查询数 / 总查询数。MRR 基于 `reciprocal_rank()`，是每条查询里第一条命中结果排名的倒数，取所有查询的平均值，衡量"正确答案排得够不够靠前"。

**Q: BEIR SciFact 这个基准是怎么接入的？和你们自己标的 Resume Gold Eval 有什么区别？**
A: BEIR SciFact 是学术界公开的信息检索基准数据集，用来验证系统在通用领域检索任务上的横向对比能力（能和 ColBERT 这类已发表方法比较）。Resume Gold Eval 是项目自己标注的 30 条查询集，更贴近实际业务场景但规模小、领域窄。两者一起用是为了既有"业务场景真实性"又有"学术基准可比性"。

---

## 模块九：文档生命周期管理

### 核心实现（`rag/infra/stores/docstore_sqlite.py` 第 229 行 `delete_document()`）
- **DocStore 层**：单条 SQL DELETE 语句删 documents 表，靠 `ON DELETE CASCADE` 外键约束自动级联删除关联的 text_blocks 和 chunks 表记录——**这一层是数据库原生级联，不是应用层手动多表删除**
- **索引层**（`ingest_pipeline.py` 第 467-480 行）：单独循环遍历 `prev_chunk_ids`，逐个调用 `keyword_index.remove(cid)`（BM25）和 `vector_index.remove(cid)`（FAISS），删完后调用 `.save()` 落盘
- **两层删除是分开触发的**：DocStore 的级联删除处理"关系型数据一致性"，索引层的手动循环处理"检索索引一致性"，两者不是同一个事务

### 高频面试题

**Q: 删除一个文档，DocStore 和检索索引是怎么保持一致的？**
A: 分两层。DocStore（SQLite）里靠外键的 `ON DELETE CASCADE` 约束，删 documents 表一条记录会自动级联删掉它名下的所有 text_blocks 和 chunks，这是数据库原生行为。但 BM25 和 FAISS 是独立于 SQLite 的检索索引，不会跟着级联，所以 ingest_pipeline 里有单独一段代码，拿到要删除的 chunk_id 列表后手动循环调用两个索引各自的 `.remove()` 方法，删完再显式 `.save()` 落盘。

**Q: 如果删除过程中索引层的 remove 失败了一半会怎样？**
A: 目前是同步循环调用，没有事务保护——如果 BM25 那部分删成功了、FAISS 删到一半抛异常，会导致两个索引状态不一致（BM25 里已经没有这些 chunk 了，FAISS 里还残留部分）。这是当前实现的真实局限，生产环境要做得更稳健的话需要引入重试或补偿机制。

---

## 模块十：工程化实践

### 测试体系
- pytest 显式标记：`@pytest.mark.unit`（48 个）/ `@pytest.mark.integration`（6 个）/ `@pytest.mark.e2e`（4 个），**其余绝大多数测试未显式打标记**
- 全部收集到的测试总数约 **1112 条**（`pytest --collect-only` 实测）
- **Mock 策略**：用 `unittest.mock.MagicMock` + `patch`，例如 `tests/test_query_pipeline_generation.py` 里 mock 出一个返回假 `LLMResponse`（含假 token 用量）的 LLM 客户端，避免真实 API 调用
- **配置加载不做缓存**：`rag/core/registry/config_loader.py` 的 `load_config()` 每次调用都会重新读盘（`yaml.safe_load()`），不是模块级单例缓存；如果配了 profile，会再读一份 profile 文件用 `_deep_merge()` 合并

### 高频面试题

**Q: 测试怎么 mock LLM 调用的？**
A: 用标准库 `unittest.mock`，比如 `test_query_pipeline_generation.py` 里直接 mock 一个 LLM 客户端对象，让它返回预先构造好的 `LLMResponse`（带假的 token 计数），这样测试只验证业务逻辑（Pipeline 怎么处理这个响应），不依赖真实网络调用，也不产生真实费用。

**Q: 配置文件是怎么加载的？会不会每次请求都重新读一次盘？**
A: 是的，`load_config()` 没有做缓存，每次调用都会重新 `yaml.safe_load()` 读 `configs/settings.yaml`。好处是改配置不用重启进程就能生效（下次调用自动读到新值），坏处是高频调用场景下有不必要的文件 IO 开销——这是一个可以讨论"要不要加缓存、加了缓存怎么处理热更新"的开放性问题。

---

## 常见"露馅"警示点

面试中如候选人无法解释以下细节，需在报告中标记：

| 简历描述 | 深挖问题 | 露馅信号 |
|---------|---------|---------|
| "混合检索 Recall@10 提升到 0.96+" | 是在什么基准上测的？多少条？ | 说不清 Resume Gold Eval（30 条人工标注）这个基准，或说成"随便测的" |
| "RRF 融合算法" | k 值是多少？能不能改？ | 说不出 k=60 默认值，或说成写死不能改（实际可通过配置覆盖） |
| "设计可插拔架构" | 新增 Provider 要改哪些文件？ | 说不清"注册表 + 惰性导入"机制，或以为是简单的 if-else 分支 |
| "幂等摄取" | chunk_id 怎么生成的？ | 说是简单对内容做 hash，说不出是对 block 哈希的有序拼接 |
| "实现 MCP 协议集成" | Transport 用的什么？为什么 docker-compose 映射了端口？ | 不知道默认是 stdio，说不出 docker-compose 端口映射其实没用 |
| "扫描 PDF OCR 支持" | 和普通 PDF 解析是同一条路径吗？ | 以为所有 PDF 都走 OCR，不知道结构化解析和 OCR 是两条独立路径 |
| "重排失败有降级机制" | 具体在哪实现的？ | **实际没有这个机制**——如果候选人坚持说有，是明显的露馅/夸大 |
| "跨存储一致性删除" | DocStore 和索引层各自怎么删的？ | 不知道 DocStore 是数据库级联删除、索引层是手动循环删除这两种不同机制 |
| "TDD 开发，1100+ 测试" | 单元测试怎么 mock LLM？ | 不知道用的是 `unittest.mock`，或说不出具体测试文件 |
