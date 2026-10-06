# Agent LLM Gateway

分层 LLM 网关（api / service / core 三层），把单文件原型演进为可配置、可部署、可测试的服务：多供应商路由、统一重试、L1-L4 失败分类、SQLite 用量账本、鉴权限流与 OpenAI 兼容接口。

## 已交付能力

**基础交付：**
- `POST /v1/chat/completions`（OpenAI Compatible）、`POST /v1/responses`、`GET /v1/models`、`GET /healthz`
- 多 Provider、公开模型别名、**优先级路由**与**加权轮询**（YAML 声明式配置）
- 普通输出、SSE Streaming、**首个内容块前**的重试与 fallback（首块后只发流内错误，不换模型拼接）
- `response_format.json_schema` / `text.format` 的本地二次校验和**一次反喂修复**
- **SQLite 用量账本**：Token、Cost、TTFT、Latency、重试、fallback 与错误（调用方只留短指纹）
- **Bearer API Key**、进程内令牌桶限流、**熔断**、Docker、健康检查、自动化测试（57 例）

**扩展能力补充：**
- **取消传播**：客户端断开 → 下游生成器显式关闭（`aclosing`）+ 唯一 `cancelled` 终态
- **业务规则校验**：JSON 合法但业务不合法（如 `steps: []`）→ `business_validation_failed` 422，绝不进入 Agent Loop（pydantic `TaskPlan` 注册表）
- **可解释 RouteDecision**：Trace 记录每个候选的 `action/reason`（`priority_first`/`weighted_pick`/`fallback_after_network_error`/`circuit_open`）
- **Prompt Bundle**：输入 Schema、输出 Schema、业务规则绑定、模型与参数默认值、内容 hash、v1 稳定版 + v2 候选版并存
- **资源治理**：每调用方并发（超限 429）、每供应商候选并发（超限排队）、**Run 预算 deadline**（重试/Fallback/修复共用同一预算，耗尽 504）、价格版本（`price.version`）
- **观测闭环**：`GET /v1/metrics` 输出 P50/P95 TTFT、Latency、错误率、429 比例，按模型/调用方/Prompt 版本聚合；`X-Run-Id`/`X-Step-Id` 头贯穿日志与账本（Run/Step/Call 关联）；供应商请求 ID、schema hash、prompt hash 进 Trace
- **base_url 白名单**：协议白名单（http/https）+ 可选主机白名单
- **账本自动迁移**：旧库自动 `ALTER TABLE` 补列，保留历史数据

两个刻意保留的边界：限流/熔断/并发守卫是单进程内存实现，多副本部署需迁移 Redis 等共享存储；本服务只负责模型执行，不维护 Agent Run 状态、不替 Harness 判断任务是否完成。

## 一次请求的完整链路

```
POST /v1/chat/completions (Bearer Key)
  → api/deps.require_caller        鉴权 + 令牌桶限流 + 写入调用方指纹
  → api/openai_schemas             协议转换（未支持字段在模型调用前 422 明确失败）
  → service/prompt_service         受控模板注入（沙箱渲染）
  → core/providers/selector        公开别名 → 候选路由（priority/weighted + 能力过滤）
  → core/retry + service 编排      L1/L2 网络重试（退避+Retry-After）；L3/L4 内容问题绝不换 provider
  → core/providers/openai_provider 上游 HTTP 请求（SDK 重试关闭，口径唯一）
  → service/ledger_service         SQLite 记账（Token/Cost/TTFT/Latency/重试/fallback/错误）
```

## 目录结构

```
llm-gateway/
├── gateway.py                    # 薄入口：uvicorn
├── providers.yaml                # 多供应商/路由策略/API Key/重试/熔断/模板 声明式配置
├── Dockerfile / docker-compose.yml / .dockerignore
├── llm_gateway/
│   ├── main.py / settings.py     # 应用装配 / 配置加载（可注入，供测试）
│   ├── api/                      # 协议适配：routes（内部+models+healthz）、routes_openai、
│   │                             #   openai_schemas、deps（鉴权限流）、errors、middleware
│   ├── service/                  # 业务编排：llm_service（L1-L4 判定链）、prompt_service（沙箱）、
│   │                             #   validation_service（L4）、ledger_service（SQLite 账本）
│   └── core/                     # 基础能力：exceptions（FailureClass 唯一事实源）、models、retry、
│                                 #   circuit_breaker、rate_limit、security、logging_setup、masking、
│                                 #   providers/（base / openai_provider / registry / selector）
└── tests/                        # Fake Adapter 测试套件（鉴权/限流/路由/重试/流式/熔断/账本/沙箱/兼容层）
```

依赖方向严格单向：`api → service → core`（api 可引用 core 的领域模型与错误类型）。

## L1-L4 失败分类（core.exceptions.FailureClass 唯一事实源）

| 层 | 检测点 | 同模型重试 | 换 provider | 熔断 | 失败出口 |
|---|---|---|---|---|---|
| L1 DNS/连接/TLS/超时 | core Adapter | 指数退避+抖动 | 允许 | 计入 | 502 `model_unavailable` |
| L2 429 | core Adapter | 优先 `Retry-After` | 允许 | 不计入 | **429**（透传 Retry-After） |
| L2 5xx / 408 | core Adapter | 指数退避 | 允许 | 计入 | 502 |
| L2 其他 4xx | core Adapter | 否 | 否 | 不计入 | 502 `provider_client_error` |
| L3 协议 choice | core Adapter 显式判定 | 否 | 否（内容问题） | 不计入 | **422** `empty_content` |
| L4 JSON/Schema | service 校验 | **反喂重修一次（同 route）** | 否（内容问题） | 不计入 | 修不好 → **422** |

策略铁律：内容问题（L3/L4）一律不换 provider；取得到文本就反喂重修一次；取不到就 422；纯文本请求完全跳过 L4。网络重试与修复重试分开计数（`network_attempts` / `repair_attempts`）。

## Usage 三态记账

| 上游行为 | tokens | usage_missing | cost_usd |
|---|---|---|---|
| 明确回 0 | 0 | false | 正常计价（0） |
| 未回 usage / 部分缺失 | 0（占位） | true + warning | `null`（未知 ≠ 0 美元） |

## 快速开始

```bash
cd llm-gateway
pip install -e ".[dev]"
export GATEWAY_API_KEY=demo-key-please-change
export DEEPSEEK_API_KEY=sk-xxx DEEPSEEK_BACKUP_API_KEY=sk-xxx
python gateway.py               # http://127.0.0.1:8000（/docs 交互文档）

pytest                          # 全量自动化测试（40 例，Fake Adapter，不打真实上游）
```

Docker：

```bash
docker compose up --build       # 需先 export 两个 DEEPSEEK key；HEALTHCHECK 探 /healthz
```

### 调用示例

```bash
AUTH='Authorization: Bearer demo-key-please-change'

# OpenAI SDK 可直连 /v1/chat/completions
curl -s localhost:8000/v1/chat/completions -H "$AUTH" -H 'content-type: application/json' -d '{
  "model": "general-primary",
  "messages": [{"role": "user", "content": "用一句话介绍网关模式"}]
}'

# Structured Output（L4 本地校验 + 一次修复）
curl -s localhost:8000/v1/chat/completions -H "$AUTH" -H 'content-type: application/json' -d '{
  "model": "general-primary",
  "messages": [{"role": "user", "content": "判断是否需要搜索"}],
  "response_format": {"type": "json_schema",
    "json_schema": {"schema": {"type":"object","properties":{"need_search":{"type":"boolean"}},"required":["need_search"],"additionalProperties":false}}}
}'

# Responses API（text.format 结构化）
curl -s localhost:8000/v1/responses -H "$AUTH" -H 'content-type: application/json' -d '{
  "model": "general-primary", "input": "回答一个整数",
  "text": {"format": {"type": "json_schema", "schema": {"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}
}'

# SSE 流式（首块前可 fallback；openai sdk: client.chat.completions.create(stream=True)）
curl -N localhost:8000/v1/chat/completions -H "$AUTH" -H 'content-type: application/json' -d '{
  "model": "general-balanced", "stream": true,
  "messages": [{"role": "user", "content": "写一首两行短诗"}]
}'

# 模型列表 / 健康检查 / 审计账本（分页+过滤）/ 观测聚合（时间窗口）
curl -s localhost:8000/v1/models
curl -s localhost:8000/healthz
curl -s 'localhost:8000/v1/traces?status=failed&model=fast&since_minutes=60&limit=20' -H "$AUTH"
curl -s 'localhost:8000/v1/metrics?window_minutes=60' -H "$AUTH"

# Run/Step/Call 关联：调用方带头，贯穿日志与账本
curl -s localhost:8000/v1/chat/completions -H "$AUTH" -H 'X-Run-Id: run-42' -H 'X-Step-Id: step-3' ...
```

### 配置要点（providers.yaml）

- `api_keys`：纯密钥字符串列表；**留空 = 开发模式不鉴权**（生产设 `GATEWAY_API_KEY`）
- `providers.<id>`：`base_url` / `api_key`（`${ENV:-默认}` 替换）/ `timeout_seconds` / `enabled`
- `models.<别名>.strategy`：`priority`（routes 顺序=主备链）或 `weighted_round_robin`（加权轮询）；`routes[].api`：`both`（chat+responses）或 `chat`
- `pricing.<供应商模型名>`：`input/output/cached_input_per_million`（美元/1M tokens）
- `retry.retry_statuses`：哪些 HTTP 状态值得同模型重试；`structured_output_retries`：L4 反喂修复次数
- 可选扩展段：`governance`（Run 预算/并发）、`security`（base_url 白名单）、`prompt_templates`（Prompt Bundle）
- `GATEWAY_CONFIG` 可指定配置路径；成本聚合记得排除 `usage_missing=true` 的行
