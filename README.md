# rag-agent-eval

一个基于 FastAPI + Celery + Redis 的大模型应用实践项目，用于验证 RAG 检索增强、Agent 工具调用（Function Calling）、异步任务队列与 SSE 进度推送的工程化落地，并附带一套可重复执行的评测体系。

## 技术栈

- **后端框架**：Python 3.10+、FastAPI、Celery、Redis
- **RAG 编排**：LangChain、ChromaDB（cosine）、BAAI/bge-small-zh-v1.5（Embedding）、BAAI/bge-reranker-base（Rerank）
- **大模型**：OpenAI 兼容接口，通过环境变量切换模型
- **部署与评测**：Docker Compose、Ragas（LLM-as-a-Judge）

## 已实现的核心功能

### 1. 异步调度与流式交互

- 文档解析与向量化通过 Celery 异步解耦，上传接口立即返回 `task_id`，避免阻塞请求链路。
- Agent 的执行状态与步骤进度通过 SSE 实时推送，使用 `asyncio.to_thread` 包装同步 Celery 查询，避免阻塞事件循环。

### 2. 两阶段检索与防幻觉

- **粗排**：Chroma 向量检索（显式指定 cosine 距离），配合阈值过滤低相关片段。
- **精排**：引入 CrossEncoder（BGE Reranker）对候选片段二次打分，取 top3 送入 Prompt。
- **兜底**：Prompt 强约束 + 检索为空时直接拒答，不调用大模型，抑制幻觉。

### 3. Agent 工作流与记忆管理

- 基于标准 Function Calling 与 JSON Schema 实现工具路由（天气查询 + 本地知识库），采用多步循环：决策 → 调用工具 → 观察结果 → 再决策。
- 支持并行工具调用，逐个回填 `tool` 消息，保证 `tool_call_id` 一一对应。
- 通过 Redis 维护多轮对话上下文，截断时按完整轮次切分，避免出现悬空的 `tool` 消息。
- 使用 Pydantic 校验工具入参，Tenacity 处理可重试的网络异常。

### 4. 工程化与安全防御

- **并发控制**：Worker 使用单进程池（`--pool=solo`），避免多个进程各自加载 Embedding 与 Rerank 模型导致内存翻倍。
- **文件安全**：`basename` + 扩展名白名单拦截路径穿越；仅接收 `.md` / `.txt`，单文件限制 10MB。
- **写入可靠性**：临时文件 + `os.replace` 原子替换，避免写入中断产生半截文件。
- **幂等去重**：写入临时文件时流式计算 MD5，重复文件直接拒绝并清理；配合 Redis `SETNX` 原子占位，防止并发重复入库。文件名前缀为 8 位随机串（非内容哈希），同一文件重传会生成新文件名，去重依赖 MD5 判定。
- **熔断机制**：Agent 执行配置最大步数（5 步）与超时（60 秒）双重限制，同时为每次模型调用设置独立超时。

### 5. 自动化评测体系

- 引入 **Ragas**，构建 8 道题的测试集（6 道可回答 + 2 道诱导性拒答）。
- 外层 **3 次重采样**计算均值与标准差，降低 LLM-as-a-Judge 的随机性。
- 另建 Agent 专用评测，覆盖工具选择准确率、步数合规率、事实命中率与越界拒答率。

## 评测结果

最近一次运行（3 次重采样）：

| 指标 | 均值 | 有效样本 |
|---|---|---|
| Faithfulness | 0.95 | n=5 |
| Answer Relevancy | 0.94 | n=5 |
| Context Precision | 1.00 | n=5 |
| Context Recall | 1.00 | n=5 |
| 拒答准确率 | 2/2 | n=2 |

> 说明：当前测试集规模较小且语料单一，检索几乎必然命中，Context 指标接近饱和；拒答题仅 2 条，统计意义有限。上表用于回归对比，不代表通用场景下的绝对水平。

## 检索设计

| 环节 | 参数 | 说明 |
|---|---|---|
| 切块 | 400 字符 / 80 重叠 | 控制在 bge 的 512 token 上限内 |
| 粗排 | top5 | `similarity_search_with_score`，cosine 距离 |
| 阈值 | `THRESHOLD = 0.5` | 距离上限，等价余弦相似度 ≥ 0.5 |
| 精排 | top3 | BGE Reranker 二次排序 |

> Chroma 的 cosine 空间返回 `1 - 余弦相似度`，值越小越相关。阈值是对距离的上限约束。

## 接口说明

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/api/upload` | 上传文档，返回 `task_id` |
| POST | `/api/chat` | 同步问答，返回答案与引用片段 |
| POST | `/api/agent` | 异步 Agent，返回 `task_id` |
| GET | `/api/task/{task_id}` | 查询任务状态 |
| GET | `/api/agent/stream/{task_id}` | SSE 推送执行进度 |

## 快速开始

1. 配置 `.env`（填入 API Key 与 Base URL）。
2. 启动服务：

```bash
docker compose up -d --build
```

3. 访问接口文档：`http://127.0.0.1:8000/docs`

4. 上传文档：

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@你的文件.md"
```

5. 执行评测：

```bash
python eval/run_eval.py
python eval/run_agent_eval.py
```

## 项目边界与后续规划

- 当前 Agent 评测采用硬编码判定，后续计划引入 LangSmith 等工具做全链路 Trace。
- 当前 Worker 并发受限，未来考虑将 Embedding 与 Rerank 剥离为独立的 Model Serving 服务。
- 评测集计划扩充至 20–30 题，并引入主题相近的干扰文档，提升指标区分度。
- Ragas 的 judge 目前与生成模型相同，存在自评偏好，计划切换为独立模型。
- 待补充单元测试与 GitHub Actions 流水线。

## 目录结构

```text
rag-agent-eval/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI 入口与路由
│   ├── ingest.py         # 文档解析、切块、入库
│   ├── retrieve.py       # 粗排 + 阈值 + 精排
│   ├── generate.py       # Prompt 与答案生成
│   ├── agent.py          # Function Calling 循环
│   └── celery_app.py     # Celery 配置
├── data/docs/            # 原始文档与入库副本
├── eval/
│   ├── questions.jsonl
│   ├── ragas_report.json
│   ├── run_eval.py
│   └── run_agent_eval.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
