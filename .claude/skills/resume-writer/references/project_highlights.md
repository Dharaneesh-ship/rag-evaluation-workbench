# 项目技术亮点清单（RAG Framework）

> 从 `dev_spec_v8.md`、`README.md` 与源码提炼，供简历编写时按需选取。每个亮点附带"简历话术方向"和"可量化角度"。

---

## 亮点 1：多阶段混合检索架构（Hybrid Search + Rerank）

**技术要点**：
- 检索层并行执行 Dense Retrieval（FAISS 向量检索）+ Sparse Retrieval（BM25 关键词匹配）
- 通过 RRF（Reciprocal Rank Fusion，`rag/infra/indexes/rrf_fusion.py`）融合双路结果，平衡查准率与查全率
- 精排阶段支持 Cross-Encoder 本地重排 / Voyage Rerank / Noop（不重排）三种模式可插拔切换
- 重排失败或未启用时自动回退至融合排名（Graceful Fallback），保障系统可用性

**简历话术方向**：
- "设计并实现了 Hybrid Search 混合检索引擎，结合 BM25 稀疏检索与 FAISS 稠密向量检索，通过 RRF 融合算法实现查准率与查全率的平衡"
- "引入可插拔 Rerank 模块（Cross-Encoder / Voyage），在 Resume Gold Eval 基准上将 Recall@10 从 0.923 提升至 0.964"

**可量化角度**：Recall@10（0.923 → 0.964）、MRR（0.615 → 0.622）、nDCG@10（0.688 → 0.708）、端到端查询延迟

---

## 亮点 2：全链路可插拔架构（Interface + Factory + 配置驱动）

**技术要点**：
- 为 LLM / Embedding / VectorIndex / Reranker / Evaluator / OCR 等核心组件定义统一抽象接口（`rag/core/interfaces/`）
- 采用工厂注册模式（`rag/core/registry/`）+ YAML 配置驱动（`configs/settings.yaml`），实现"改配置不改代码"的组件切换
- Embedding 支持 OpenAI / 本地多语言模型两种后端
- 向量索引默认 FAISS，接口层预留扩展空间
- 重排后端支持 Voyage API 与本地 Cross-Encoder 模型切换

**简历话术方向**：
- "设计了全链路可插拔架构，基于抽象接口 + 工厂注册 + YAML 配置驱动，实现 Embedding/VectorIndex/Reranker 等核心组件的零代码热切换"
- "架构支持云端 API（OpenAI/Voyage）与本地模型无缝切换，兼顾效果与成本"

**可量化角度**：支持的 Embedding/Reranker 后端数、配置切换零代码修改

---

## 亮点 3：多格式智能数据摄取流水线（Ingestion Pipeline）

**技术要点**：
- 自研摄取流水线：文件类型嗅探（`rag/infra/sniffing/`，基于 magic number + MIME）→ 解析 → 切分 → 向量化 → 幂等 Upsert
- 支持 PDF、HTML、Markdown、Word、Excel 等多格式解析器（`rag/infra/parsing/`）
- 段落感知切分（Paragraph Splitter）+ Chunk Packer 控制 chunk 粒度
- Chunk ID 基于内容哈希生成，实现增量摄取与幂等 Upsert，避免重复处理
- 双路向量化：Dense（Embedding）+ Sparse（BM25）并行编码

**简历话术方向**：
- "设计并实现多格式智能数据摄取流水线，整合文件类型自动嗅探、多格式解析（PDF/Word/Excel/HTML/Markdown）、段落感知切分与双路向量化"
- "实现基于内容哈希的增量摄取机制，避免重复处理，保障 Upsert 幂等性"

**可量化角度**：支持的文档格式数、处理文档数、生成 Chunk 数、增量摄取跳过率

---

## 亮点 4：MCP 协议集成（Model Context Protocol，Claude 集成）

**技术要点**：
- 基于 MCP 标准实现知识检索 Server（`rag/app/mcp_server/`）
- 暴露 7 个工具：`rag.ingest` / `rag.query` / `retrieve` / `retrieve_with_metadata` / `list_collections` / `rag.eval.run` / `rag.sync_source`
- 支持 Claude Desktop、Claude Code 等 MCP Client 即插即用调用
- 支持 collection 参数进行多语料库检索范围隔离（如仅检索指定文档集合）

**简历话术方向**：
- "基于 MCP（Model Context Protocol）标准实现知识检索 Server，支持 Claude Desktop / Claude Code 等 AI 客户端直接调用私有知识库"
- "设计 collection 级别的检索范围隔离，支持同一系统内多语料库互不干扰的场景化检索"

**可量化角度**：暴露 MCP 工具数（7 个）、支持的 MCP Client 数、工具调用成功率

---

## 亮点 5：扫描文档 OCR 处理（Page Rendering + OCR）

**技术要点**：
- 针对扫描版 PDF，采用 PyMuPDF 渲染页面为图像（`rag/infra/ocr/renderer_pymupdf.py`）+ PaddleOCR 提取文本（`rag/infra/ocr/paddleocr_provider.py`）
- 输出结构化 `IRBlock`（含页码、Bounding Box、单块置信度）
- `ParseReport` 记录 `parser_used` 与聚合置信度统计，便于摄取质量追溯

**简历话术方向**：
- "设计扫描 PDF 的 OCR 摄取通路，结合页面渲染与 PaddleOCR 文本识别，输出带置信度与坐标的结构化文本块，扩展框架对非文本 PDF 的覆盖能力"

**可量化角度**：OCR 平均置信度、扫描文档处理占比、单页 OCR 耗时

---

## 亮点 6：全链路可观测性与可视化管理平台

**技术要点**：
- SQLite 持久化的 Trace Store，记录摄取与查询各阶段耗时、候选数量、分数分布
- 基于 Streamlit 构建 Studio，含 4 大页面：
  - Ingest & Inspect（文档上传、Chunk 浏览）
  - Query Traces（BM25/向量/Rerank 前后对比、延迟与 Token 用量）
  - Evaluation Panel（检索质量基准 + RAGAS 答案质量评估）
  - Connector Sync（Email/Slack/Notion/Google Docs 数据源同步）
- 零外部依赖（无需 LangSmith/LangFuse），SQLite Local-First

**简历话术方向**：
- "构建全链路白盒化追踪体系（摄取 + 查询），每次检索过程透明可回溯，支持精准定位坏 Case"
- "基于 Streamlit 实现四页面可视化管理平台，覆盖数据摄取、查询追踪、评估面板与外部数据源同步"

**可量化角度**：Dashboard 页面数（4 个）、追踪覆盖阶段数、问题定位效率提升

---

## 亮点 7：自动化评估体系

**技术要点**：
- 检索质量评估：Recall@K / MRR / nDCG，基于 30 条人工标注的 Resume Gold Eval 基准（非循环标注，非自证）
- 答案质量评估：集成 RAGAS（Faithfulness / Answer Relevancy / Context Precision）
- 对齐 BEIR SciFact 公开基准评测：nDCG@10 达到 0.703，超过已发表的 ColBERT baseline（0.671）
- 评估历史持久化，支持策略调整前后的量化对比

**简历话术方向**：
- "建立基于人工标注基准 + RAGAS + BEIR 公开数据集的三层自动化评估闭环，拒绝'凭感觉调优'"
- "在 BEIR SciFact 基准上跑出 nDCG@10 = 0.703，超过已发表的 ColBERT baseline（0.671）"

**可量化角度**：Recall@10 / MRR / nDCG 具体数值、BEIR 基准得分、评估集规模（30 条标注）

---

## 亮点 8：外部数据源连接器

**技术要点**：
- 支持 Email（IMAP）/ Slack / Notion / Google Docs 四种外部数据源自动同步（`rag/infra/connectors/`）
- 通过 Studio 的 Connector Sync 页面或 `rag.sync_source` MCP 工具触发同步
- 同步后的文档复用统一摄取流水线，纳入增量哈希去重机制

**简历话术方向**：
- "设计可扩展的外部数据源连接器体系，支持 Email/Slack/Notion/Google Docs 一键同步至知识库，降低企业知识沉淀成本"

**可量化角度**：支持的数据源类型数（4 种）、同步文档量、同步耗时

---

## 亮点 9：工程化实践

**技术要点**：
- 分层测试金字塔：Unit → Integration → E2E，共 1100+ 自动化测试用例全绿
- SQLite 轻量持久化（文档存储 + Trace 存储 + BM25 索引），零外部数据库依赖
- 配置驱动的零代码组件切换（YAML Profile 支持多环境覆盖）
- Docker + docker-compose 一键部署 Studio UI 与 MCP Server 双服务
- Prompt 模板外置于 `configs/prompts/`，支持独立迭代而不改代码

**简历话术方向**：
- "遵循分层测试金字塔，累计编写 1100+ 自动化测试用例，覆盖单元/集成/E2E 三层"
- "采用 SQLite Local-First 持久化方案，零外部数据库依赖，配合 Docker Compose 实现一键部署"

**可量化角度**：测试用例数（1100+）、开发阶段完成度、部署所需服务数（2 个容器）

---

## 亮点 10：Agent 扩展性（面向 Agent 方向的延伸叙事）

**技术要点**：
- MCP Server 天然支持 Agent 的 Tool Calling 范式，可被 Claude 等 Agent 直接调用
- collection 级别的检索范围隔离机制，可快速适配"多业务场景共用一套检索框架"的需求
- 可插拔架构使得替换数据源、调整检索/重排策略、定制 Prompt 均无需改动核心代码

**简历话术方向**（适用于偏 Agent 方向的岗位）：
- "基于 MCP 协议构建知识检索 Agent 能力，支持 Tool Calling 范式，可嵌入 Claude 等 Agent 生态作为知识中枢"
- "设计通用化知识检索框架，支持通过配置快速适配不同业务场景，无需修改核心代码"

**可量化角度**：支持的 MCP Client 数、业务场景/Collection 适配数
