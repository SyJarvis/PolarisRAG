# PolarisRAG Agent 接入契约（Agent Web Contract）v1.0

> 状态：草案（待实现）
> 日期：2026-09-28
> 目标：定义一个**与具体实现无关**的「网站 ↔ Agent 服务」HTTP 契约。任何网站按本契约接入；
> 任何 Agent 后端（不限于 PolarisRAG）实现本契约即可被同一前端无缝替换（换脑测试，见 §9）。
> 配套符合性测试：`tests/contract/test_api_contract.py`

---

## 0. 设计原则（5 条铁律）

| # | 原则 | 含义 |
| --- | --- | --- |
| P1 | 核心对齐 OpenAI Chat 格式 | `choices/message/usage` 等核心字段与 OpenAI Chat Completions 逐字节兼容；事实标准，接入摩擦最低 |
| P2 | 扩展字段永不污染核心字段 | `content` 永远是纯文本答案；来源、trace 等只放扩展位（`citations`/`trace`），禁止文本内嵌引用 |
| P3 | 无状态优先 | 会话历史由**调用方回传**（`messages`），服务端不记忆会话；记忆能力不进契约 |
| P4 | 能力自描述 | 客户端先 `GET /v1/capabilities` 再决定行为；可选端点能力门控，换 agent 自动降级 |
| P5 | 宽容读者 | 客户端必须忽略不认识的字段；`/v1` 内只允许新增可选字段（additive only），破坏性变更升 `/v2` |

## 1. 通用约定

- 鉴权：`Authorization: Bearer <key>`，不引入自定义签名。
- 编码：请求/响应 UTF-8 JSON；流式为 `text/event-stream`（SSE）。
- 版本：URL 路径版本化（`/v1`）。
- 必需端点：`POST /v1/chat/completions`、`GET /v1/capabilities`。
- 可选端点（能力门控）：`POST /v1/ingest`、`GET /v1/status`。

## 2. `GET /v1/capabilities`（必需）

```jsonc
{
  "contract_version": "1.0",
  "agent_name": "polarisrag",        // 实现方自命名
  "streaming": true,
  "citations": true,                  // 是否返回 citations 扩展
  "sessions": "client",               // client=调用方管历史 | server=服务端管（本契约推荐 client）
  "max_context_messages": 12,         // 调用方应回传的最大消息条数；null=不限
  "capabilities_extra": {             // 可选端点：有才列出，没有不得假设存在
    "ingest": "/v1/ingest",
    "status": "/v1/status"
  }
}
```

约束：
- 客户端**必须**依据本响应决定是否使用流式/扩展字段/可选端点，禁止硬编码。
- `capabilities_extra` 中出现的路径仅为提示，可用性以实际 HTTP 状态为准。

## 3. `POST /v1/chat/completions`（必需）

### 3.1 请求

```jsonc
{
  "model": "polarisrag",              // 不透明字符串，实现方自定；客户端原样回显即可
  "messages": [                        // OpenAI 格式；历史由客户端回传（原则 P3）
    {"role": "user", "content": "例会是什么时候？"}
  ],
  "stream": true,                      // 默认 false
  "metadata": {"language": "zh"}       // 可选，语义化提示；实现方可忽略
}
```

### 3.2 非流式响应（`stream=false`）

```jsonc
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1791447000,
  "model": "polarisrag",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "例会为每周一上午十点。"},
    "finish_reason": "stop"            // stop | length | error
  }],
  "citations": [                       // 扩展位①：来源；capabilities.citations=false 时应为 null/缺省
    {
      "id": "c1",
      "title": "项目纪要",
      "uri": "internal://doc/1",       // 通用 URI；实现方内部细节（集合名/chunk id）不得外泄
      "snippet": "例会时间为每周一上午十点",
      "score": 0.82
    }
  ],
  "trace": {"steps": ["search(1)", "sufficiency(0.78)", "generate"]},  // 扩展位②：可观测，结构自由
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

约束：
- `choices[].message.content` 必须是**纯文本答案**。禁止把结构化数据塞进 `content`
  （如 `【来源1】…`、JSON、Markdown 脚注编号）。符合性测试会做基本模式检查（§8）。
- `citations[].uri` 不得包含实现内部细节（Milvus 集合名、数据库路径等）。

### 3.3 流式响应（`stream=true`，SSE 类型化事件）

事件流使用**类型化事件**（`event:` 行声明类型），不用裸 delta：

```
event: message
data: {"type":"message.delta","delta":"例会为"}

event: message
data: {"type":"message.delta","delta":"每周一上午十点。"}

event: sources
data: {"type":"sources","citations":[ …与 3.2 citations 同构… ]}

event: done
data: {"type":"done","finish_reason":"stop"}

event: error
data: {"type":"error","code":"upstream_unavailable","message":"LLM 服务不可用","retryable":true}
```

约束：
- 事件类型词汇表：`message`（增量正文）、`sources`（来源）、`done`（结束）、`error`（错误）。
- `sources` 事件可出现在 `done` 之前的任意位置（通常在首个 `message` 前或 `done` 前）。
- 收到 `done` 或 `error` 后事件流必须终止；`error` 时不得再发 `message`。
- HTTP 状态在鉴权/参数校验失败时直接返回 4xx（不进入事件流）；已进入事件流后的失败用 `error` 事件。

## 4. `POST /v1/ingest`（可选，能力门控）

```jsonc
// 请求
{"text": "……", "source_name": "项目纪要", "metadata": {}}

// 响应
{"chunk_ids": ["7187..."], "source_id": "src-1", "chunk_count": 3}
```

约束：仅文本入库；上限由实现方自定，超限返回 `413` + `context_overflow` 错误信封。
删除能力不在 v1 契约内（PolarisRAG 侧等 schema 升级后再议）。

## 5. `GET /v1/status`（可选，能力门控）

```jsonc
{"agent_name": "polarisrag", "healthy": true,
 "documents": {"source_count": 2, "chunk_count": 57},
 "models": {"llm": "gpt-4o-mini", "embedding": "text-embedding-3-small"}}
```

约束：**脱敏**——不得返回任何密钥、内部主机名、数据库路径。

## 6. 错误信封（对齐 OpenAI 格式）

HTTP 4xx/5xx 时返回统一信封：

```jsonc
{"error": {
  "code": "context_overflow",         // 受控词汇表，见下
  "type": "invalid_request_error",    // invalid_request_error | api_error | auth_error | rate_limit_error
  "message": "messages 超过 12 条上限",
  "retryable": false
}}
```

错误码词汇表（禁止自造码外裸文本）：

| code | HTTP | retryable | 场景 |
| --- | --- | --- | --- |
| `invalid_request` | 400 | false | 参数缺失/格式错误 |
| `auth_failed` | 401 | false | 鉴权失败 |
| `rate_limited` | 429 | true | 必须携带 `Retry-After` 头 |
| `context_overflow` | 400/413 | false | 消息数/文本超限 |
| `upstream_unavailable` | 502/503 | true | LLM/Embedding 上游故障 |
| `internal_error` | 500 | false | 其他 |

## 7. 实现纪律（写死，不许协商）

1. `content` 纯净性：任何引用信息只能出现在 `citations`（P2）。
2. 无状态：同一 `messages` 输入重复调用，结果可不同（生成随机性），但服务端不得要求先前调用过的会话存在。
3. 内部不外泄：契约字段中不出现 `top_k`、集合名、Milvus 路径、chunk 内部 id。
4. `/v1` 内 additive only：新增字段必须可选；改语义、删字段、改必选性 → 升 `/v2`。
5. 超时与重试建议（客户端侧）：非流式 60s；流式以 `done`/`error` 为终止信号；仅 `retryable=true` 自动重试。

## 8. 换脑测试（swap test）

`tests/contract/test_api_contract.py` 是可执行的契约符合性套件，对任意实现方运行：

```bash
export POLARIS_CONTRACT_BASE_URL="http://127.0.0.1:8000"
export POLARIS_CONTRACT_API_KEY="sk-..."
pytest tests/contract -v
```

断言覆盖（全部来自本契约的可机器验证子集）：

- A1 `capabilities`：`contract_version`/`agent_name` 存在，`sessions ∈ {client, server}`；
  未声明的可选端点返回 404 而非 500。
- B1 非流式：核心 OpenAI 字段齐全；`content` 为非空字符串且无内嵌引用模式（`【来源`、`[1]`）；
  `citations` 若存在，元素含 `id/title/snippet` 且 `uri` 无内部路径痕迹。
- C1 流式：事件类型属于词汇表；首个事件后必有 `done` 或 `error` 收尾；`message.delta` 均为字符串。
- D1 错误信封：空 `messages` / 非法 role → 400 且信封含受控 `code`；错误码 ∈ 词汇表。
- E1 可选端点（仅在 capabilities 声明时执行）：`ingest` 正常入库返回 `chunk_ids`；
  `status` 响应不含密钥形态字符串（`sk-`）。

通过全部断言 ≈ 该后端可被任何按本契约开发的前端无缝替换（人类可读性验收另行人工评审）。

## 9. PolarisRAG 映射（实现指引）

| 契约元素 | 现有对应物（能力层） | 改造量 |
| --- | --- | --- |
| `/v1/chat` 非流式 | `mcp_server/agent.py` 决策循环 | FastAPI 薄封装 |
| `citations` | registry `sources_used` | 字段通用化（去 Milvus 痕迹） |
| `trace` | `tool_trace` | 直通 |
| `sources` SSE 事件 | 无 | 新增 |
| `/v1/capabilities` | 无 | 新增 |
| `/v1/ingest` | `mcp_server/ingest.py` | 薄封装 |
| `/v1/status` | `_status_payload()` | 脱敏检查 |

实现顺序：**先冻结本文档 → 符合性测试红 → FastAPI 实现绿**。顺序不可反。
