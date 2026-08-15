# RAG-MCP-Server ↔ MAEDA 集成交接文档

**发出方**：rag-framework
**接收方**：MAEDA
**日期**：2026-08-12（2026-08-14 三方审核 + 外部评审后修订）
**状态**：设计已对齐。第 8 节原有四项挂起决定中，8.1（连通性）与 8.4（事件脚本归属）已定案；8.2、8.3 留到 A/B 评测后用数据裁决。另附两处顺带发现的既有缺陷（第 6 节，不属于本次集成范畴）和两项 rag-framework 自主决定（附二，仅供参考，无需回应）

> **权威文件**：执行顺序与分工以 `~/ECOSYSTEM_INTEGRATION_PLAN.md`（v3，现场为 v4）为唯一权威，本文档与之冲突处一律以它为准；本文档提供的是背景论证与 rag 侧细节。下文标注的 **R1–R4** 即该计划阶段 1 中 rag-framework 的四项改动编号。
>
> **[2026-08-15 更新] R1–R4 已完成**（此前 v4.4 进度速览记录的是"本轮完全未碰"）：
> - **R1**（4.3）：`embedding_provider`/`reranker_provider` 改为显式枚举
>   `server_default`/`none`/显式 provider，服务端默认经
>   `RAG_EMBEDDING_PROVIDER_DEFAULT`/`RAG_RERANKER_PROVIDER_DEFAULT` 配置；
>   provider/model 分离；旧客户端 `null` 归一化为 `server_default`。实现见
>   `rag/app/mcp_server/retrieval_config.py`。
> - **R2**（4.4，rag 侧最高优先级）：新增 `index_manifest.json`
>   （`schema_version`/`embedding_provider`/`embedding_model`/`vector_dimension`），
>   ingest/connector sync/retrieve/retrieve_with_metadata/`rag.cli.eval` 共用
>   同一套配置解析与索引兼容性校验（`rag/app/mcp_server/index_validation.py`）；
>   不兼容默认（`RAG_INDEX_COMPAT_MODE=strict`）明确失败，`degraded` 模式下降级
>   到 BM25 且必须携带 `degraded_reason`。
> - **R3**（4.5）：新增 `rag/core/errors.py` 稳定错误码
>   （`INDEX_NOT_FOUND`/`INDEX_CONFIG_MISMATCH`/`EMBEDDING_PROVIDER_UNAVAILABLE`/
>   `RERANKER_UNAVAILABLE`/`INVALID_COLLECTION`/`RETRIEVAL_INTERNAL_ERROR`）；
>   `retrieve`/`retrieve_with_metadata` 内部故障现在抛真正的 MCP tool error
>   （`isError=True`），合法零命中仍为成功空 `chunks`。**MAEDA 侧可以在验证后移除
>   对这两个工具 `error` 字段的迁移期检查**（`rag.query` 未改动，其 `error`
>   字段行为保持原样）。
> - **R4**（4.6）：新增 `tests/test_retrieval_config.py`、
>   `tests/test_index_validation.py`、`tests/test_index_manifest.py`、
>   `tests/test_retrieve_tool.py` 覆盖 provider×server-default×索引兼容性矩阵、
>   非法枚举值校验、两个 retrieve 工具共用解析路径、trace 记录实际
>   mode/provider/model/reranker/降级原因。全量测试 1341 通过。
>
> 顺带处理（不在 R1–R4 编号内，但同一批次完成）：容器端口从 8000 改为
> 8002（定案 #1，docker-compose.yml + README §7/§8）；`pyproject.toml` 的
> `mcp` 依赖锁定为 `>=1.0.0,<2.0.0`（附录 F.4/G2 记录的紧迫问题）。
>
> 未在本轮处理：4.1（脏语料替换）、4.2（检索层评测报告 recall@k 等，`rag.cli.eval`
> 已支持 provider 选择但尚无固定 dev/test split）、附二.1（词汇扩展）——这些
> 依赖阶段 2/3 的事件脚本（TB5）尚未冻结，按计划顺序本就排在 R1–R4 之后。

---

## 0. 这份文档解决什么问题

我们要回答一个具体问题：**在自动化数据分析流程里，检索来的背景知识到底有没有用？** 不是在一般意义上回答，而是在 MAEDA + rag-framework 这个组合上，用可测量的方式回答。

结论先行：**有用，但当前的接法基本发挥不出价值。** 下文第 2 节给出证据（其中连通性一项在三方审核后已从「未知问题」变为「已定位的配置问题」，见 2.1）。

本文档分三部分：rag-framework 侧要改的、MAEDA 侧要改的、需要 MAEDA 重新拍板的。第 1 节先给出双方共同遵守的分工原则，第 3 节给出当前真实的执行顺序（供 MAEDA 侧核对理解是否一致）。

---

## 1. 分工原则：两层结构

这是整份文档所有具体条目的依据，请先认同这一节再看后面。

企业级 NL-to-analysis 系统在行业里已经收敛出一个两层结构（Snowflake 的 Cortex Analyst + Cortex Search + Cortex Agents 是这个形状，Databricks 的 Genie + Vector Search + Agent 也是）。两层的划分依据是**信息错了会造成什么后果**：

| | **策展层（Curation Layer）** | **检索层（Retrieval Layer）** |
|---|---|---|
| 装什么 | 列名语义、指标口径、字段编码变更 | 业务事件、促销日历、政策变更、邮件与会议纪要里的长尾事实 |
| 形态 | 小、结构化、人工审核过 | 大、杂、多来源、部分过时 |
| 归属 | **MAEDA** | **rag-framework** |
| 注入方式 | 确定性注入 prompt，不检索 | 检索，带引用 |
| 为什么这样分 | 错了会让 planner **选错列、算错数**，必须 100% 精准，宁可人工维护 | 长尾且易变，永远不会有人来得及整理进策展层，检索是唯一途径 |

**推论一**：口径解释不该用 RAG 做。检索有召回不确定性，一次漏召直接导致计算错误。能塞进结构化配置的，就不要走检索。

**推论二**：RAG 的合法辖区是「永远来不及进策展层的那部分」。如果 rag-framework 的语料装的是公司介绍、行业综述这类泛泛背景，那它对分析师确实零价值——这个质疑是成立的，解法是换语料，不是换检索算法。

**推论三（切分线）**：介于两者之间的「数据注意事项」，按**错了会不会让计算出错**来切。

- 会让 planner 选错列 / 写错过滤条件的 → 策展层。例：「2024 年前 region 字段用的是旧编码」「amount 列单位是分不是元」「status='X' 表示已作废，需排除」
- 只影响解释、不影响计算的 → 检索层。例：「三月华东断货」「四月上过满减」「七月改了退货政策」

---

## 2. 当前集成的真实状态（重要，请先读这一节）

以下两条是我们在梳理过程中查证出来的事实，它们改变了整个计划的优先级。

### 2.1 连通性诊断已定：README 的「0% 从未可达」是过时表述，真实根因是三层配置问题

本节初稿曾以 MAEDA README 的说法（Data Cleaner 和 RAG-MCP-Server**从未可达**，MCP 路径实际执行率 **0%**）为前提，请 MAEDA 确认不可达的具体表现。三方审核后这个问题已有答案，此处改为记录已确认的诊断结论：

**「从未可达」不成立。** MAEDA 自己的 `docs/eval_report.md` 记录了 2026 年 7 月的多次真实端到端 MCP 联调——rag-framework 手动起在 **8002 端口**（streamable-http），全部 20 个 golden case 的 `mcp_call_log.mcp_modes` 均含 `'mcp'` 而非仅 `'fallback'`（`baseline_b_rag_online.json` 起的多个 baseline）。README 那句「0%」晚于这些记录却未更新，属过时表述，待 MAEDA 侧修正。

**协议层双方是兼容的。** rag-framework 的 HTTP 可达性已在 commit `b0cb4d1` 修复；MAEDA 侧早期裸 POST JSON-RPC、不做 `initialize` 握手的 bug 已在 commit `b5242ea` 修复（改用官方 `mcp` SDK 的 streamable-http client）。

**「默认配置下不可达」的真实根因是三层配置问题，不是协议问题：**

1. 两个子系统的 MCP server 默认走 **stdio** transport 而非 HTTP——不显式设环境变量（rag-framework 侧为 `MCP_TRANSPORT=streamable-http`）就没有 HTTP 监听；
2. HTTP 模式下的默认端口互相冲突，且与 MAEDA 配置假设的端口不符（rag-framework docker-compose 发布在 8000，MAEDA `settings.py` 默认读 8002）；
3. MAEDA 自己的 `docker-compose.yml` 引用了从未构建过的镜像（`maeda-ecosystem/rag-mcp-server:latest`）和不存在的目录（`../rag-mcp-server`，实际仓库名是 `rag-framework`）。

三方已就端口达成协定（见 8.1）。在配置修复落地之前，本文档其余条目仍以连通为前提——但这已经是一项可枚举的运维待办，不再是未知的阻塞风险。

### 2.2 当前的 MCP 检索路径是「纯 BM25，无向量，无重排」

这一条尤其需要 MAEDA 侧知晓，因为它直接解释了「检索回来的东西为什么可能没用」。

`RetrieveInput` schema（rag-framework `rag/app/mcp_server/schemas.py:329`）中：

```python
embedding_provider: Optional[str] = Field(default=None)
```

而 `_run_retrieval`（rag-framework `rag/app/mcp_server/server.py:399`）的行为是：

```python
embed_provider = None
vec_index = None
if input.embedding_provider == "openai":
    ...  # 只有显式传入 "openai" 才启用向量检索
pipeline = QueryPipeline(
    keyword_index=mgr.bm25,
    vector_index=vec_index,          # → None
    embedding_provider=embed_provider,  # → None
    answer_composer=None,
    reranker=None,                    # → 无重排
    top_k=input.top_k,
)
```

而 MAEDA 侧的调用（MAEDA `src/mcp_client/rag_server.py`）只传了三个字段：

```python
input_args = {"query": query, "top_k": top_k}
if collection:
    input_args["collection"] = collection
```

**没有传 `embedding_provider`。** 因此当前这条路径上：

- 向量检索**未启用** → RRF 只有一路排序表可融合，混合检索退化为纯词法检索
- 重排器**未启用**（`reranker=None` 是 `_run_retrieval` 里写死的）
- 也就是说，rag-framework 引以为核心的「BM25 + 向量 → RRF 融合 → 重排」这条完整管线，**在 MCP 路径上一条都没生效**

这正面回答了一个疑问：「基于 BM25 关键字提取的背景信息真的能帮 data analyst 吗？」——当前它确实**就是**纯 BM25，所以这个疑虑在事实层面成立。修复方式见 4.3（配置契约）、**4.4（索引兼容性，只改 4.3 不够）**和 5.4。

---

## 3. 当前实际执行顺序（供 MAEDA 侧核对）

以下取自 MAEDA `src/graph/builder.py:53` 起的图定义，不是 README 的描述。

```
用户提问
   │
   ▼
[1] parse_intent                    ← 会话层在这里
   │   读：user_query, conversation_history, schema_summary
   │   出：parsed_intent{query_type, target_metrics, dimensions}
   │   条件分支 → ask_clarification（最多 1 次）→ 回到 parse_intent
   ▼
[2] connect_and_profile_data
   │   连接数据源 → 抽 schema → 生成 schema_summary（自动、机械）
   │   委托 Data Cleaner MCP 做质量剖析
   │   有严重问题 → 清洗循环（最多 3 次）→ 重抽 schema
   ▼
[3] plan_analysis                   ← 口径词表应在这里注入（见 5.1）
   │   LLM 一次性产出带依赖关系的类型化步骤计划
   ▼
[4] execute_analysis
   │   非 LLM 执行器按拓扑序执行；失败步骤有 1 次 LLM 辅助修复
   ▼
[5] generate_viz
   ▼
[6] retrieve_domain_knowledge       ← ★ RAG 在这里介入 ★
   │   query = insight_agent.build_retrieval_query(state)
   │   → MCP call: retrieve_with_metadata({"input": {query, top_k=5, collection}})
   │   出：rag_context[], rag_sources[]
   ▼
[7] generate_insights
   │   分析结果 + rag_context → insights + 报告
   ▼
[8] run_guardrails                  条件分支 → 回到 execute_analysis（最多 2 次）
   ▼
[9] run_eval
   ▼
[10] persist_run → END
```

### 3.1 关键点：RAG 收到的不是用户原话

发给 rag-framework 的查询由 MAEDA `src/agents/insight_agent.py:116` 的 `build_retrieval_query` 合成，**是纯 Python 字符串拼接，没有 LLM 参与**：

```python
parts = []
if metrics:    parts.append(f"{query_type} analysis of {', '.join(metrics)}")
if dimensions: parts.append(f"by {', '.join(dimensions)}")
if summaries:  parts.append("Key findings: " + "; ".join(summaries))  # 取前 3 条
query = " — ".join(parts) if parts else state.get("user_query", "")
```

实际发出的查询形如：

```
trend analysis of revenue — by region — Key findings: Q3 East region revenue fell 30%; ...
```

**这个设计有一个真实优点，请保留**：检索查询是被**已经算出来的真实数字**塑造过的。它不是在问「关于收入有什么背景」，而是在问「华东三季度掉了 30%，语料里有什么能解释」。对「解释异常」这个用途，这比在 planner 阶段盲检索更精准，因为那时还不知道哪里异常。

### 3.2 一个已知漏洞

`insight_agent.py:144` 的兜底分支：

```python
query = " — ".join(parts) if parts else state.get("user_query", "")
```

当意图为空且所有分析步骤都失败时，**用户原话会原封不动漏到检索器**。这正是 rag-framework 侧要做词汇扩展的理由之一——检索服务不能假设进来的一定是合成过的好查询（见附二.1）。

---

## 4. rag-framework 侧要做的

### 4.1 更换：demo 语料

当前语料是一份 LLM 编写的干净 docx（`Wake_Apparel_MAEDA_Business_Context.docx`），它有两个问题：内容偏「公司介绍」类的泛背景（按推论二属于零价值那类），且**背后没有任何 ground truth**。

替换为**多来源脏语料**：若干封邮件、一份会议纪要、一页政策备忘、一份促销日历——分散、口语化、部分过时、部分互相矛盾。

内容严格限定为第 1 节定义的检索层辖区：业务事件、促销、政策变更、断货、库存流转说明。**不包含**指标口径与列名解释（那些归 MAEDA 的策展层）。

副产物：rag-framework 现有的 email connector 与 connector_sync_pipeline 到这里才第一次有真实的使用场景。

### 4.2 新增：检索层评测

在脏语料上测埋点事件的召回率——针对每个埋点事件构造查询，检查是否召回了对应的语料块。这是纯检索指标，不依赖 MAEDA，可独立跑、独立出结论。

**判定依据必须是 `event_id` 或明确的 relevance label，禁止用关键词/字符串命中判相关**（理由见 7.5 的同源措辞泄漏控制）。

**报告的指标**：recall@k、precision@k、MRR 或 nDCG，以及延迟。

**可复现可审计**：每次评测结果都要记录事件脚本版本 + 哈希、索引 manifest（见 4.4）、以及本次运行配置（provider/model/reranker/top_k）。否则跑出来的数字事后无法归因到某一次具体的配置。

**dev/test 隔离**：调参（分块、top-k、融合权重、reranker、查询构造）只在 `dev` split 上做，`test` split 冻结后只用于最终报告（见 7.4）。同时作为附二.1 词汇扩展的判定依据——该判定同样**只在 dev split 上做，不碰 test**。

### 4.3【R1】修改：检索配置公共契约（方案已定案，用显式枚举）

见 2.2。`_run_retrieval` 当前写死 `reranker=None`，且只有显式传入 `embedding_provider="openai"` 才启用向量检索。**定案采用服务端配置默认值 + 显式枚举**：

```python
embedding_provider: Literal["server_default", "none", "openai"] = "server_default"
reranker_provider: Literal["server_default", "none", "crossencoder", "voyage"] = "server_default"
```

**为什么不用 `None` / 布尔哨兵**（这一版推翻了本文档上一稿的三态方案）：布尔 `reranker` 表达不了「用哪个重排器」——rag-framework 已同时支持 CrossEncoder 与 Voyage，二值开关既选不了具体实现，也没法对它们做独立评测；同时用 `None` 兼表「服务端默认」、用字符串兼表「关闭」，是把两种不同语义压进一个字段，读代码时无法自解释。改为显式枚举后，每个取值的含义写在类型里。

**配套规则**：

- provider 与 model 分开配置，不要把 model 名塞进 provider 字段。
- 旧客户端未传该字段时，归一化为 `server_default`；服务端环境变量未配置时保持 BM25-only，**向后兼容不破坏**。
- **显式请求参数优先级高于服务端默认。**
- 环境变量的布尔解析**不得把字符串 `"false"` 当成真值**（Python 里 `bool("false") is True` 这个坑要显式挡掉）。

改动落在 `rag/app/mcp_server/schemas.py` 与 `server.py`。**MAEDA 侧零代码改动即可获得完整管线**——见 5.4。

### 4.4【R2】新增：索引兼容性校验（rag 侧最高优先级）

**这一项是 4.3 生效的前提，优先级高于 4.3 本身。** 只改查询端配置**不足以**让向量检索真的生效：向量查询成立还要求语料在**摄取时就已经生成了向量**，且查询端的 provider / model / dimension 与建索引时完全一致。

不校验的后果是一种最难发现的失败模式：**假 hybrid**——配置和日志都显示「完整管线已开启」，实际因为没有向量索引（或维度对不上）而仍然是 BM25-only。它不会报错、不会告警，但会直接让 5.5 的 A/B 评测结论作废（测出来的「RAG 有用/没用」根本不是在测你以为的那条管线）。

要求：

- ingest、connector sync、`retrieve`、`retrieve_with_metadata`、评测入口**共用同一套检索配置解析**，不允许各自实现一份。
- 索引保存 **manifest**，至少含 `schema_version` / `embedding_provider` / `embedding_model` / `vector_dimension`。
- 查询前校验：FAISS 索引存在且非空、provider/model/dimension 与 manifest 一致、API key 或本地模型依赖可用。
- 不兼容时**默认明确失败**（走 4.5 的错误码 `INDEX_CONFIG_MISMATCH` / `INDEX_NOT_FOUND`）；确需降级到 BM25 时，**必须在结构化返回值和日志里记录降级原因，禁止无痕降级**。

### 4.5【R3】新增：MCP 错误语义正式修复

当前 `retrieve` / `retrieve_with_metadata` 的内部异常被 `try/except` 兜住，返回 `RetrieveOutput(error=..., chunks=[])`，而 MCP 协议层是**成功**。调用方只要漏查 `error` 字段，就会把「检索服务故障」读成「没搜到相关内容」——两者在下游的处置完全不同，却长得一模一样。

**目标态**：

- **合法零命中** → 成功响应 + 空 `chunks`（这是正常结果，不是错误）。
- **内部故障** → 抛真正的 MCP tool error（`isError=True`），带**稳定错误码**：`INDEX_NOT_FOUND` / `INDEX_CONFIG_MISMATCH` / `EMBEDDING_PROVIDER_UNAVAILABLE` / `RERANKER_UNAVAILABLE` / `INVALID_COLLECTION` / `RETRIEVAL_INTERNAL_ERROR`。
- 异常原文只作**调试详情**，不作长期公共契约——调用方不许对着异常文本做字符串匹配来分支。

**迁移期双轨**：在本项落地之前，MAEDA 侧保留对 `error` 字段的检查（对应权威计划的 M5）；rag-framework 侧修复落地后**主动通知 MAEDA**移除该检查。这是一处双方共享的契约脆弱点，与附一的 `{"input"}` 包装同级，改动前必须知会对方。

### 4.6【R4】新增：配置矩阵自动回归测试

覆盖组合：（未指定 / 显式 `none` / 显式 provider）×（服务端默认有无）×（索引兼容 / 无向量索引 / 维度不匹配），另含非法值校验。同时验证两个 retrieve 工具**确实共用同一套配置解析**（4.4 的要求需要被测试固定住，否则日后很容易各自漂移）。

trace 需记录本次实际执行的 retrieval mode、provider、model、reranker，以及降级原因（若发生降级）。

**验收口径**：TB1 要看**真实的向量候选与 reranker 调用**（看候选列表与融合过程），**不是只看「已启用」的配置日志**——这正是 4.4 那个假 hybrid 问题在验收环节的对应防线。

### 4.7 需明确的 reranker 运维事项

以下几项需在实现时定下来并写进 README，它们直接影响 TB1 的体感与可复现性：

- 默认 reranker provider 与 model 是什么。
- 模型缓存目录在哪。
- 离线环境下的行为（拿不到模型时是什么表现）。
- 首次启动是否预热。
- 加载失败时是**明确报错**还是**带标记降级**（对应 4.5 的 `RERANKER_UNAVAILABLE`）。

特别注意：**CrossEncoder 首次下载有冷启动成本**，联调时如果没预热，第一次调用的延迟会显著偏高，容易被误读成检索本身慢。

---

## 5. MAEDA 侧要做的

### 5.1 新增：口径词表（策展层）

**形态：YAML 存储 + 按真实 schema 过滤后注入。**

我们评估过三个方案，结论如下。真正决定这题的不是存储格式，而是**词表长大之后怎么办**——三十条时怎么存都能跑，差别发生在它涨到两百条、跨五个数据源之后，那时它塞不进 prompt，必须做**选择**。

- **方案 A：硬编在 `src/config/agent_prompts.py` 的 planner prompt 里。** 十分钟能做完，零加载逻辑。代价是业务知识与 prompt 工程焊死——改一条口径要动代码、走 code review，而口径恰恰是业务方最常改、也最该由业务方自己改的东西。涨大后没有任何选择机制，只能整块塞。属于原型解。
- **方案 B：每数据源一个独立 YAML，按数据源加载。** 这是 Snowflake semantic model 的形状。业务方可直接改文件、可版本控制、可 review，天然按数据源隔离。代价是多了加载与校验层，且引入一个静默失败模式：YAML 声明 `net_revenue` 列存在，实际数据里叫 `revenue_net`，词表与真实 schema 悄悄漂移，模型拿着一份关于不存在的列的解释去做计划。
- **方案 C（选定）：YAML 存储 + 按真实 schema 过滤注入。** 在 B 的基础上加一步：注入前用 `schema` 中**实际存在的列名**筛词表，只注入命中的条目。这一步是确定性字符串匹配，不用 LLM、不用检索、不引入延迟。

选 C 的理由：比 B 只多二三十行代码，但同时解决两个问题——词表可以长到几百条而 prompt 只装当前这张表用得上的十几条；schema 漂移从静默错误变成**可检测的信号**。

**同一套匹配逻辑白送三种状态，都应写入 state：**

| 状态 | 含义 | 给谁看 |
|---|---|---|
| `full` | 词表条目基本都在 schema 中命中 | 正常，无需提示 |
| `partial` | 词表里有若干条在 schema 中找不到 | **漂移告警，给开发者**——词表过时或数据源换了 |
| `absent` | 未知数据源，无词表 | **caveat，给用户**（见 5.2） |

**注入点**：第 [3] 步 `plan_analysis` 的 prompt，以及第 [2] 步的 intent payload（三方审核新增，见下方「注入点扩展」）。确定性注入，不走检索。

**为什么不把词表丢进 RAG 检索**：那等于让口径解释承担检索漏召的风险，一次漏召直接导致 planner 用错列、算出错数。只有当词表大到按 schema 过滤后仍塞不下 prompt（单表上百列且每列口径复杂）时，检索才重新成为候选——预计不会遇到这个规模。

**内容范围**：列名语义、指标口径、以及第 1 节推论三划归策展层的那类数据注意事项（编码变更、单位、废弃值）。

**注入点扩展（三方审核新增的联合设计议题）**：data-cleaner 的交接文档（`agentic-data-cleaner-v2/HANDOFF_TO_MAEDA.md`「跨项目事项」一节）要求领域知识在清洗（第 [2] 步）时就可用，并建议「把领域知识查询提前」。但 cleaner 需要的内容——这列是营收还是 ID、null 是缺失还是合法状态、大小写差异是部门代码还是拼写错误——全部是列名语义与口径，按第 1 节推论一属于策展层辖区，不该走 RAG 检索。正确解法不是把检索前移（8.2 的暂缓结论不变），而是把本节口径词表的注入点从「只在第 [3] 步」扩展到「第 [2] 步的 intent payload」：词表按 schema 过滤的时机相应提前到第 [2] 步节点内部，过滤命中的条目顺着 cleaner 已在接收的 `intent` 参数传给 `get_cleaning_plan`/`clean_dataset`——cleaner 侧无需新增管线，只是 intent 对象变得更丰富。三方已在审核中对齐这个方向，具体 intent 字段结构待与 cleaner 联合定义。

> **已知扩展边界，暂不处理**：以上方案隐含一个前提——单个数据源的 schema 小到能整个塞进第 [3] 步的 LLM 调用，词表按 schema 过滤时也是对着这份完整 schema 线性扫描。这个前提在当前 demo（CSV/SQLite，一两张表）成立，但在接入生产级数据库（如 Snowflake，可能有数百张表、数千个列）时会失效——那时连"把完整 schema 交给 LLM 看一眼"这一步本身就已经超预算，更谈不上过滤词表。
>
> 需要拆开看的是两件扩展曲线完全不同的事：词表的存储和过滤（确定性字符串匹配）本身不受规模影响，涨到几百条也便宜；真正撑不住的是**被过滤的对象**——schema 发现这件事，要从「每次查询时现枚举全量 schema」演进为「数据源接入时一次性建索引，查询时只检索出相关的一小撮表和列」，也就是 text-to-SQL 文献里的 **schema linking**。这本质上是策展层内部在大规模下也需要一层轻量检索来做筛选，不再是无脑全量注入——但它筛选的是「这张表这列该不该出现在这次分析里」，错了的后果是选错列、算错数，因此仍然属于策展层辖区，不应与检索层的脏语料共用同一个 collection 或评测。
>
> 这条不在本次集成范围内，此处仅作为显式声明的适用范围记录，供 MAEDA 未来接入更大数据源时参考。它与第 6.1 节的 `tables[0]` 硬编码问题同源——那是当前小规模下"选表"偷懒的表现，大规模下会变成"没有索引就无法选表"的硬约束。

### 5.2 新增：未知上传数据源的处理

MAEDA 接受 `data/uploads` 下的任意上传文件，这是主要入口之一，此时词表不存在。处理原则是**降级但出声，且不许瞎编**。注意这是**两件不同的事**，混在一起就会做成「只有警告、没有防护」：

**(a) 防瞎编 —— 靠 prompt 约束，不是警告能解决的。**

词表状态为 `absent` 时，planner prompt 中显式加入约束：当前数据源没有经过审核的列语义，不得基于列名推断业务含义，只能按列的字面内容与类型作计划。

这句话必须加在**瞎编会发生的那个点**上。否则模型看到一列叫 `revenue` 就默认它是净收入，报告照样写得斩钉截铁，而用户看到的警告只是页面角落一行小字。

**(b) 警告 —— 必须跟着报告走，不能只在上传时弹一次。**

报告会被导出、截图、转发给别人看，那一刻上传时的 UI 提示早已不在。所以这条 caveat 要写进报告本身。**修正一处初稿的断言**：初稿写的是「MAEDA 的 insight 已有 `confidence` 字段与 caveat 机制，挂在那里最自然」，MAEDA 侧核实后指出这只对了一半——Insight dataclass 有 `confidence` 字段，但**没有结构化的 caveat 字段**；`src/agents/insight_agent.py:400` 处只是把工具级警告拼进 evidence 字符串，不是可复用的 caveat 机制。因此 (b) 需要 MAEDA 新增一个结构化字段（如 `caveats: list[str]`）并让报告渲染层带上它，工作量比初稿估计的大。

**两件明确不要做的事：**

- **不要因为词表缺失就机械地压低 insight 的 `confidence`。** 它看起来很对，但 confidence 会流进 `_score_to_impact` 和 eval 打分，而我们之后要用 eval 跑 RAG on/off 的 A/B——人为压低一个输入会污染那次测量。让 confidence 由内容决定，caveat 作为并列字段单独存在。
- **不要做成 guardrail 的阻断检查。** 词表缺失不是不安全输出，是低保证输出，应放行并标注，不应拦截。

### 5.3 修改：`generate_demo_data.py` 读取事件脚本

见第 7 节。当前 demo 数据与语料之间没有任何对应关系，导致无法评测。

### 5.4 MCP 调用与 embedding_provider（已定案：MAEDA 零改动）

见 2.2。`src/mcp_client/rag_server.py` 当前只传 `{query, top_k, collection}`，导致检索退化为纯 BM25。初稿在此悬置了两个方案（MAEDA 显式传参 / rag-framework 服务端配置默认值），**已定案采用后者**（具体契约见 4.3 的显式枚举）——检索配置属于检索服务的职责，不该让每个调用方都懂。

**MAEDA 侧不需要改任何检索配置代码**：未传的字段由服务端归一化为 `server_default`，在 rag-framework 配好服务端默认值后自动获得向量检索与重排。需要 MAEDA 知晓的两点：

1. 默认值生效后，同样的调用会从纯 BM25 变为向量 + 重排，延迟与 embedding/重排的 API 成本相应上升，跑 5.5 评测时注意保持配置口径一致；且在 4.4 的索引兼容性校验落地前，这个「已启用」可能是假的（见 4.4 的假 hybrid），A/B 数字要等 R2 完成后才可信。
2. **迁移期仍需保留 `error` 字段检查**（权威计划的 M5）——rag 侧 4.5 的错误语义修复落地后会主动通知 MAEDA 移除。

### 5.5 新增：端到端评测

RAG on 对 off，三个指标（ground truth 来自第 7 节的事件脚本）：

- **归因准确率**：insight 对数据异常的归因是否命中脚本中的正确事件
- **错误归因率**：是否引用了脚本中埋设的**干扰事件**（语料里有、数据里无对应信号）
- **幻觉率**：RAG-off 时，面对**无解释异常**（数据里有、语料里无对应记载），模型是编造理由还是承认不知道

跑在 MAEDA 的 eval harness 上——它测的是 insight 质量，不是检索质量（检索质量归 4.2）。

---

## 6. 顺带发现的缺陷（与本次 RAG 集成无关，仅顺带报告）

在梳理第 [2] 步「connect_and_profile_data」时发现两处问题。**这两处都在 MAEDA / Data Cleaner 的范畴内，rag-framework 侧不涉及，也不依赖它们被修复。** 单独列出，是为了不让它们被误读成本次集成的设计分歧——它们是既有 bug，不是待拉扯的决策项。之所以在这份文档里提，是因为其中一处会污染 5.5 的 A/B 评测结果，值得 MAEDA 侧知晓。

### 6.1 多表数据源的表选择是硬编码的 `tables[0]`

`src/tools/data_connector.py:209`：

```python
if not table_name:
    tables = inspector.get_table_names()
    table_name = tables[0]
    logger.info("Auto-selected table: %s", table_name)
```

未指定 `table_name` 时，直接取 `inspector.get_table_names()` 的第一张表作为 active table，**没有参照用户的问题**。只有这张表会被 `read_sql_table` 整个读入 DataFrame 用于后续分析；其余表只在第 [3] 步以「表名+列名类型」清单的形式提供给 planner 参考（`src/agents/analysis_agent.py:164` 的 `_related_tables_manifest`），供其决定要不要 JOIN，但读进内存做计算的从始至终只有第一张。

后果：用户问客户流失，若 `tables[0]` 恰好是 `promotions` 表，则整条流水线从第 [2] 步就已经跑偏，第 [3] 步的 planner 再准确也救不回来——它拿到的本来就是错误的数据。

**为什么与本次 A/B 评测（5.5）相关**：demo 数据里 `data/demo/ecommerce_orders.db` 是多表的。若某次评测跑用错了表，那一轮的归因准确率是噪声，会分不清是「RAG 没帮上忙」还是「数据源本来就选错了」。建议 MAEDA 在跑 5.5 之前先确认或修复此项，否则评测结果需要额外核查每次跑选中的是哪张表。

### 6.2 `parse_intent`（第 [1] 步）读取的 `schema_summary` 恒为空

`schema_summary` 直到第 [2] 步才被赋值（`src/graph/nodes.py:154`），初始值是空串（`src/state/graph_state.py:87`）。但 `parse_intent`（第 [1] 步，含 `ask_clarification` 回环）在 `schema_summary` 参数上做判断（`src/agents/intent_parser.py:96,114,194`）——这一步在图里排在第 [2] 步之前，因此第一次进入意图解析时读到的永远是空串。

也就是说，这段依赖 `schema_summary` 来辅助判断「revenue 指的是哪一列」之类逻辑的代码，实际上从未拿到过真实 schema。是否需要把该依赖后移到第 [2] 步之后，或在选表后补一次意图对齐，请 MAEDA 侧自行判断——这纯粹是时序问题，不涉及 RAG。

---

## 7. 共享产物：事件脚本（归属已定，见 7.6）

这是整个方案里**唯一被两个仓库同时消费**的产物，因此必须先说定归属与变更流程。

### 7.1 它是什么

**它不是语料，是语料的上游规格。** 原先那份 LLM 编写的 docx 只是一份文档，背后没有任何 ground truth——没有一张表说明「文档里提到的事，分别应该在数据里长成什么样」。事件脚本是一个**以前不存在的新产物**。

内容形如：12 个月里发生 N 个业务事件。**每个事件必须包含以下字段**（外部评审后扩充，初稿只有「业务描述 / 数据信号 / 是否干扰项」三项，不够支撑可复现的评测）：

| 字段 | 含义 |
|---|---|
| `event_id` | **稳定标识符**，评测判定的唯一依据（见 7.5） |
| `valid_from` / `valid_to` | 时间语义。没有它就无法表达「这条信息在某个时间点之后已过时」 |
| `business_description` | 业务描述，用于渲染语料 |
| `data_signal` | 应在数据中产生的信号形态：`metric` / `segment` / `expected_direction` |
| `corpus_evidence` | 该事件在语料中的证据：relevance label + `source_types`（邮件 / 会议纪要 / 政策备忘 / 促销日历） |
| `is_distractor` | 是否为埋设的干扰事件（语料里有、数据里无对应信号） |
| `has_explanation_in_corpus` | 数据里有异常但语料无记载的负例标记，用于测幻觉率 |

### 7.2 为什么必须两边独立渲染

**这是防止自证的关键机制。** 如果流程是「先看数据（三月跌了），再照着数据编文档（三月断货）」，那么 A/B 测试中 RAG 必赢，因为答案被抄进了考卷。面试官或评审只要问一句「你的语料是怎么来的」，整个评测的可信度就塌了。

正确流程是**事件脚本先行**：

1. 先写事件脚本作为 ground truth
2. MAEDA 的数据生成器读脚本 → 只渲染出数字形态
3. rag-framework 的语料生成器读脚本 → 只渲染出业务语言
4. 两边**各自独立渲染，互不参照对方的产物**

再加两类刻意的不对称，让 RAG-on 不必赢、RAG-off 不必输：

- **干扰事件**：语料里写了，数据里没有对应信号 → 测错误归因率
- **无解释异常**：数据里有，语料里没记载 → 测幻觉率

### 7.3 硬约束

**脚本必须先写完，数据和文档才能生成。** 这个次序一旦被违反，评测的可信度就没了。这是最容易在执行中被无意违反的一条。

### 7.4 事件划分：dev / test / stress

事件必须划分成三组，**这是防过拟合的机制**：

- **`dev`**：调参用——分块策略、top-k、融合权重、reranker、查询构造方式，都只在这组上调。
- **`test`**：冻结后**只用于最终报告**，中途不许拿来看效果。
- **`stress`（可选）**：冲突信息、过时信息、无解释异常、大量干扰项。

理由很直接：如果不分组，反复在同一组事件上调检索器，调出来的就是**对这份固定脚本的过拟合**，报出来的 recall 是自己喂给自己的答案。这一条对 4.2 的检索层评测和附二.1 的词汇扩展判定同样适用。

### 7.5 同源措辞泄漏控制

**这是本次外部评审提出的、最容易被忽略的一个评测陷阱。** 同一份脚本同时生成数据、评测查询和语料——如果三者沿用相同的关键词，BM25 会因为字面重合而虚高，测出来的「检索有效」其实是「同一个词出现了两次」。

语料与评测查询必须刻意覆盖以下差异：

- 同义词与业务别名（查询说 GMV、语料写总成交额）
- 缩写与全称
- 口语化邮件与正式政策文本的措辞差异
- **不直接出现指标名、但语义相关**的证据
- 相似但无关的干扰事件
- 过时或互相矛盾的文档
- 数据有异常但语料无解释的负例

**判定规则**：必须基于 `event_id` 或明确的 relevance label，**禁止用关键词/字符串命中判相关**。字符串命中判定会把上面所有刻意制造的差异全部判成「没召回」，同时把巧合的词面重合判成「召回了」，两个方向都错。

### 7.6 归属与版本（已定案）

它同时是双方的输入、又兼任评测的标准答案。三方已定：

- **放 rag-framework 仓库**（建议路径 `eval/fixtures/event_script/`），YAML 格式，文件内只含**内容与 `version` 字段**；每次修改打 git tag。
- **SHA-256 放 sidecar manifest**，不写进 YAML 本身——把哈希字段写进被哈希的文件会造成自引用（改哈希字段就改变了文件内容，哈希随即失效）。同时固定 canonicalization、编码、换行规则，并提供校验脚本，否则两边算出的哈希会因为行尾或编码差异而对不上。
- MAEDA 侧**只读引用**，消费前校验哈希，不一致即报错，而不是静默用旧版；**并且要固定 rag 仓库的 commit/tag**——只靠脚本自报哈希防不住同一路径下的内容漂移（有人改了文件也改了 manifest，哈希自洽但内容已经变了）。
- **归属方与变更权限是同一方**：改动由 rag-framework 发起并打新 tag，MAEDA 升级引用即视为确认。

7.3 的硬约束（脚本先行、两边独立渲染）不受归属影响，继续有效。

---

## 8. 需要 MAEDA 重新决定 / 双方拉扯的事项

### 8.1 【最高优先级】连通性（已解决，结论如下）

初稿此处请 MAEDA 确认「是否仍不可达、具体表现」。三方审核已给出答案，本项从「待确认」改为记录结论：

- **联调事实**：MAEDA `docs/eval_report.md` 记录了 2026 年 7 月在 8002 端口的多次真实端到端 MCP 联调（20/20 case 的 `mcp_modes` 含 `'mcp'`）。README 的「0% 从未可达」是过时表述，待 MAEDA 侧更新。
- **协议层兼容**：MAEDA 的握手 bug 已在 commit `b5242ea` 修复；rag-framework 的 HTTP 可达性已在 commit `b0cb4d1` 修复。
- **真实根因**：三层配置问题（两个子系统默认 stdio 而非 HTTP；HTTP 默认端口互相冲突且与 MAEDA 假设不符；MAEDA docker-compose 引用不存在的镜像与目录），详见 2.1。

**已达成的端口协定**：rag-framework 起 **8002**、data-cleaner 起 **8001**、MAEDA 配置**零改动**（`settings.py` 默认值 8001/8002 就是最终值）。

rag-framework 侧待办，**采用方案 A：容器内外统一 8002**——

```yaml
ports: "8002:8002"
MCP_PORT: "8002"
```

不用「容器内 8000、对外映射 8002」那种内外不一致的写法：它会让 README 写的、启动命令里敲的、健康检查探的、compose 里配的成为四个需要各自换算的数字，而这次连通性问题的根因之一恰恰就是「文档各自正确但互相矛盾」。**README、启动命令、健康检查、compose 必须描述同一个事实。** README §7/§8 里写的 8000 同步改为 8002。

初稿的退路方案（写模拟 MAEDA 调用的评测脚本）作废——连通性已不构成未知阻塞，剩下的是上述可枚举的配置修复。

### 8.2 RAG 注入点是否前移

**当前**：RAG 只在第 [6] 步介入，即数据已算完、图已画完之后。检索结果只用于给已有数字配解说词。

**问题**：这是价值最弱的用法。行业验证过的高价值注入点在**上游**——planner 决定用哪列、怎么过滤、"活跃用户"算不算测试账号的那一刻。口径错了，后面算得再对都是错的，解说词写得再好也救不回来。

**若前移**，会变成**两次检索**：

- 第 [3] 步前：用「用户问题 + schema」检索 → 喂 planner
- 第 [6] 步：用「问题 + 已得发现」检索 → 喂 insight（保留现状，因为 3.1 说的那个优点是真的）

**连带影响**：两次检索的语料类别不同，多半要拆成两个 collection，`settings.rag_collection`（MAEDA `src/config/settings.py:40`，单值 `Optional[str]`）就不够用了，需要改成按用途分别配置。rag-framework 侧近期刚修过 collection 隔离问题（commit `91bf15d`），拆分是可行的。

**代价**：要动 MAEDA 的 graph（加节点或改 `plan_analysis` 的输入），比只改语料重不少。

**注意**：这一项与 5.1 的口径词表有部分职责重叠——如果口径已经由确定性的词表覆盖，前移检索的边际价值就下降了。建议先把词表做出来，再评估前移是否还必要。**这是本项待定的主要原因。**

### 8.3 检索查询的构造方式

见 3.1。当前 `build_retrieval_query` 是确定性字符串拼接，产出一个混着指标名、维度名和数字摘要的长串。这在干净单文档上无所谓，在脏语料里可能因噪声词过多而稀释关键词。

备选：让 LLM **有意识地拟一句检索问题**，例如「华东区 2024 年三季度收入为何下滑」。这是形状完全不同的查询，检索结果也会不同。

（更正一处此前的口误：当前检索发生在 insight 阶段，写查询的是 insight agent，不是 planner。只有 8.2 的前移真的发生，planner 才会参与写查询。）

这属于调用方（MAEDA）的职责，但直接影响 rag-framework 侧测得的召回率——**召回率是在 rag-framework 里测量的指标，但它的输入完全由 MAEDA 决定**，这个耦合是双方都要意识到的。

建议：4.2 的检索层评测同时跑两种查询构造方式，用数据决定。

### 8.4 事件脚本归属（已定案）

见 7.6——放 rag-framework 仓库、打 tag、SHA-256 走 sidecar manifest、MAEDA 只读引用并同时固定 commit/tag，本项关闭。

---

## 9. 验收标准

**整个集成的成败以第 5.5 节的三个指标衡量。如果 RAG on 与 off 在这三个指标上测不出可测量的差异，就承认这个集成对当前场景不成立，并把结论写进两边的文档。**

这一条是认真的，不是保险话术。rag-framework 此前有过一个先例：LLM 元数据增强实测 +0.0pt，据此砍掉了该功能。我们要交付的不是「RAG 有必要」这个结论，而是**一套能对任何组件下「值不值」判断的评测方法论**——那才是可迁移的资产。

需要注意的是，本次评测只能回答「对这个语料、这个任务，检索层值不值回成本」，**不能**推广为「RAG 在一般意义上是否必要」。后者不是一个可回答的命题，工业界也不问它。

---

## 10. 建议的执行顺序

依赖关系决定了顺序，不能并行的地方已标明。**最终执行顺序以权威计划 `~/ECOSYSTEM_INTEGRATION_PLAN.md`（v3）的阶段划分与 TB 节点为准**，本节是 rag 视角的依赖摘要。

1. **落实连通性配置修复**（按 8.1 已定的诊断与端口协定：rag-framework 按方案 A 改 docker-compose/README 到 8002，MAEDA 更新 README 表述与自身 docker-compose 的镜像/目录引用）—— 阻塞项，其余全部依赖它
2. **修复检索管线**（R1 4.3 + **R2 4.4** + R3 4.5 + R4 4.6 + 5.4）—— 其中 **R2 索引兼容性是 rag 侧最高优先级**：R2 未完成时，4.3 报告的「已启用完整管线」可能是假 hybrid，第 6 步的 A/B 数字不可信
3. **写事件脚本并冻结**（7，含 7.4 的 dev/test/stress 划分与 7.5 的泄漏控制）—— 阻塞 4 和 5，且必须早于两者
4. 并行：**MAEDA 改数据生成器**（5.3）+ **rag-framework 生成脏语料**（4.1）—— 两边独立渲染，互不参照
5. 并行：**MAEDA 做口径词表**（5.1 + 5.2）+ **rag-framework 跑检索层评测**（4.2，只在 dev split 上调参）
6. **跑端到端 A/B 评测**（5.5）—— 前提是第 2 步已完成，否则测的不是你以为的那条管线
7. 依据评测结果决定：词汇扩展（附二.1）是否启用、注入点是否前移（8.2）、查询构造方式（8.3）

---

## 附一：MCP 契约备忘

`retrieve` 与 `retrieve_with_metadata` 的参数需包装为 `{"input": {...}}` 而非扁平的 `{query, top_k}`。原因是 rag-framework 基于 FastMCP，这两个工具声明为接收单个名为 `input` 的 Pydantic 模型参数，FastMCP 将 MCP 工具调用参数映射到 Python 参数名，因此参数字典的顶层 key 必须是 `input`。MAEDA 侧已正确处理（`src/mcp_client/rag_server.py`）。

若将来 rag-framework 改为扁平关键字参数，此处需同步修改——这是一处**双方共享的契约脆弱点**，改动前请知会对方。

---

## 附二：背景——rag-framework 侧同期在做的自主决定（仅供参考，无需 MAEDA 回应）

以下两项是 rag-framework 自己的设计决定，不属于本次集成的要求，MAEDA 不需要为此采取任何行动或给出回应。它们源自一个更早、更通用的问题——"我的项目是否缺 memory、query 改写、上下文压缩这些常见 RAG 功能"——而不是集成本身提出的需求。列在这里仅为提供背景，帮助理解 rag-framework 侧同期还在做什么，其中附二.1 因为在正文里被引用（4.2、3.2、执行顺序步骤 7）而需要保留编号。

### 附二.1 新增：词汇扩展（multi-query / HyDE）

**为什么归 rag-framework 而不是 MAEDA**：这里必须区分两件常被混为一谈的事。

- **会话改写**（"那他的呢？" → 自包含问句）依赖**对话历史** → 只有 MAEDA 有 → 归 MAEDA
- **词汇扩展**（查询说「GMV」而语料写「总成交额」）依赖**对语料用词的了解** → 只有 rag-framework 有（它握着 BM25 词频统计和索引）→ 归 rag-framework

MAEDA 对语料里究竟用什么词说话一无所知，它做扩展只能是盲扩、凭 LLM 猜同义词；rag-framework 做扩展可以是**语料感知的**，扩出的词能先验证索引中是否真实存在。

边界线一句话：**MAEDA 负责把一段对话变成一个自包含的信息需求；rag-framework 负责把一个信息需求变成能匹配上自己语料的若干查询。**

**实现位置**：`rag/pipelines/query_pipeline.py:320` 的 `_run` 顶部。当前结构是「一个 query → BM25 一路 + 向量一路 → RRF 融合」，RRF 本就是融合多路排序表的算法，扩展为「N 个 query 变体 → 2N 路 → RRF 融合」不需要改架构。

**验收**：做成配置开关，默认关。在 4.2 的脏语料评测上命中率不涨就不启用——**判定只在 `dev` split 上做，不碰 `test`**（见 7.4）。

**对 MAEDA 的影响**：无需改动。这是被调用方对查询质量的兜底能力，也顺带兜住 3.2 那个原话泄漏的漏洞。

### 附二.2 新增：README 的 Non-goals 一节

明确声明**不做**以下三项，并写清理由，避免被误解为欠账：

- **对话 memory** —— 这是 MCP 边界后的无状态检索服务，会话状态属于调用方，MAEDA 自己有 multi-turn memory。无状态是刻意的设计（可水平扩展、可缓存、易测试），不是遗漏。
- **会话式指代改写** —— 同上，职责在调用方。
- **上下文压缩** —— 在 top_k=5 且启用重排的前提下，重排本身已是有效的上下文预算控制。压缩只省 token 不涨质量，不值得引入。
