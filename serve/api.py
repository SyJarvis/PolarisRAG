# -*- coding: utf-8 -*-
"""PolarisRAG Agent Web API —— docs/api-contract.md v1.0 的参考实现。

启动：
    pip install -e ".[api]"
    export POLARIS_FAKE_EMBEDDINGS=1            # 无 key 冒烟
    export POLARIS_CONTRACT_HOME=/tmp/polaris_api   # 可选，工作目录
    uvicorn serve.api:app --port 8000

验证（契约符合性）：
    export POLARIS_CONTRACT_BASE_URL=http://127.0.0.1:8000
    pytest tests/contract -v

端点（契约 v1.0）：
    GET  /v1/capabilities          必需
    POST /v1/chat/completions      必需（非流式 + 类型化 SSE 流式）
    POST /v1/ingest                可选（能力门控）
    GET  /v1/status                可选（能力门控，脱敏）

实现纪律（对应契约 §7）：
- content 纯净：答案只进 content，来源只进 citations（P2）
- 无状态：历史由调用方经 messages 回传；当前实现取最后一条 user 消息作为
  检索查询（历史融合检索属后续优化，见 PR roadmap）
- 内部不外泄：citations.uri 使用 internal:// scheme，不暴露集合名/路径/chunk id
- 流式说明：能力层（agent.run）为同步一次性返回，本实现做"重组流式"
  （sources 事件 → message.delta 分片 → done），对客户端完全符合契约；
  真token级流式待 LLM 层支持后替换。
"""
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

CONTRACT_VERSION = "1.0"
AGENT_NAME = "polarisrag"
MAX_CONTEXT_MESSAGES = 12
DELTA_CHUNK_CHARS = 48  # 重组流式的分片大小

VALID_ROLES = {"user", "assistant", "system"}

app = FastAPI(title="PolarisRAG Agent API", version=CONTRACT_VERSION)

# 懒加载能力层（import 零网络请求；首次请求才初始化 embedding/milvus/agent）
_resources = None


def _get_resources():
    global _resources
    if _resources is None:
        from mcp_server.server import _get_resources as _lazy
        _resources = _lazy()
    return _resources


def _api_key_required() -> Optional[str]:
    """设置了 POLARIS_API_KEY 即启用 Bearer 鉴权；未设置则匿名可用（本地开发）。"""
    import os
    return os.getenv("POLARIS_API_KEY") or None


def _error(code: str, http_status: int, message: str,
           retryable: bool = False, err_type: str = "invalid_request_error") -> JSONResponse:
    """契约 §6 错误信封。"""
    return JSONResponse(
        status_code=http_status,
        content={"error": {"code": code, "type": err_type,
                           "message": message, "retryable": retryable}},
    )


async def _check_auth(request: Request) -> Optional[JSONResponse]:
    key = _api_key_required()
    if key is None:
        return None
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {key}":
        return _error("auth_failed", 401, "Missing or invalid bearer token",
                      err_type="auth_error")
    return None


def _citations_from_sources(sources_used: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """sources_used → 契约 citations（通用化，含 registry 反查的 snippet）。"""
    _, _, agent = _get_resources()
    registry = agent.registry
    out = []
    for i, s in enumerate(sources_used or [], start=1):
        sid = s.get("source_id") or ""
        rec = registry.get_source(sid) if sid else None
        text = (rec or {}).get("text", "")
        out.append({
            "id": f"c{i}",
            "title": s.get("source_name") or (rec or {}).get("source_name") or sid,
            "uri": f"internal://source/{sid}" if sid else "internal://unknown",
            "snippet": text[:160],
        })
    return out


def _trace_steps(tool_trace: List[Dict[str, Any]]) -> List[str]:
    steps = []
    for e in tool_trace or []:
        if not isinstance(e, dict):
            continue
        if "tool" in e:
            steps.append(f"{e['tool']}({'ok' if e.get('ok') else 'fail'})")
        elif e.get("truncated"):
            steps.append("truncated")
    return steps


def _validate_messages(messages: Any) -> tuple[Optional[JSONResponse], Optional[str]]:
    if not isinstance(messages, list) or not messages:
        return _error("invalid_request", 400, "messages 必须为非空数组"), None
    if len(messages) > MAX_CONTEXT_MESSAGES:
        return _error(
            "context_overflow", 400,
            f"messages 超过 {MAX_CONTEXT_MESSAGES} 条上限", err_type="invalid_request_error",
        ), None
    last_user = None
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in VALID_ROLES \
                or not isinstance(m.get("content"), str):
            return _error("invalid_request", 400,
                          f"非法 message：{json.dumps(m, ensure_ascii=False)[:120]}"), None
        if m.get("role") == "user":
            last_user = m["content"]
    if last_user is None:
        return _error("invalid_request", 400, "messages 中至少需要一条 user 消息"), None
    return None, last_user


def _chat_payload(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    """能力层结果 → 契约非流式响应（核心 OpenAI 字段 + 扩展位）。"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.get("answer", "")},
            "finish_reason": "stop",
        }],
        "citations": _citations_from_sources(result.get("sources_used")),
        "trace": {"steps": _trace_steps(result.get("tool_trace"))},
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------- 必需端点

@app.get("/v1/capabilities")
def capabilities():
    return {
        "contract_version": CONTRACT_VERSION,
        "agent_name": AGENT_NAME,
        "streaming": True,
        "citations": True,
        "sessions": "client",
        "max_context_messages": MAX_CONTEXT_MESSAGES,
        "capabilities_extra": {"ingest": "/v1/ingest", "status": "/v1/status"},
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    denied = await _check_auth(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return _error("invalid_request", 400, "请求体必须是合法 JSON")
    if not isinstance(body, dict):
        return _error("invalid_request", 400, "请求体必须是 JSON 对象")

    verr, query = _validate_messages(body.get("messages"))
    if verr:
        return verr
    model = body.get("model") or AGENT_NAME
    stream = bool(body.get("stream", False))

    _, _, agent = _get_resources()
    try:
        result = await run_in_threadpool(agent.run, query)
    except Exception as e:  # AgentError / Jev / 上游故障
        return _error("upstream_unavailable", 502, f"agent 调用失败: {e}",
                      retryable=True, err_type="api_error")

    payload = _chat_payload(result, model)
    if not stream:
        return JSONResponse(content=payload)

    # 类型化 SSE（重组流式，见模块 docstring）
    async def event_stream():
        try:
            sources_event = {"type": "sources", "citations": payload["citations"]}
            yield f"event: sources\ndata: {json.dumps(sources_event, ensure_ascii=False)}\n\n"

            content = payload["choices"][0]["message"]["content"]
            for i in range(0, len(content), DELTA_CHUNK_CHARS):
                delta = content[i:i + DELTA_CHUNK_CHARS]
                ev = {"type": "message.delta", "delta": delta}
                yield f"event: message\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            done = {"type": "done", "finish_reason": "stop"}
            yield f"event: done\ndata: {json.dumps(done, ensure_ascii=False)}\n\n"
        except Exception as e:  # 已进入事件流后的失败：契约要求 error 事件而非断流
            ev = {"type": "error", "code": "internal_error",
                  "message": str(e), "retryable": False}
            yield f"event: error\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------- 可选端点

@app.post("/v1/ingest")
async def ingest(request: Request):
    denied = await _check_auth(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return _error("invalid_request", 400, "请求体必须是合法 JSON")
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str) or not text.strip():
        return _error("invalid_request", 400, "text 必须为非空字符串")
    source_name = body.get("source_name") or ""
    if len(source_name) > 200:
        return _error("invalid_request", 400, "source_name 不能超过 200 字符")
    if len(text) > 100000:
        return _error("context_overflow", 413, "text 超过 100000 字符上限",
                      err_type="invalid_request_error")

    _, ingestor, agent = _get_resources()
    try:
        rec = await run_in_threadpool(ingestor.add_text, text, source_name, "fast")
    except Exception as e:
        return _error("upstream_unavailable", 502, f"入库失败: {e}",
                      retryable=True, err_type="api_error")

    stored = agent.registry.get_source(rec["source_id"]) or {}
    chunk_ids = [str(c) for c in stored.get("chunk_ids", [])]
    return {
        "chunk_ids": chunk_ids,
        "source_id": rec["source_id"],
        "chunk_count": rec.get("chunks", len(chunk_ids)),
    }


@app.get("/v1/status")
async def status(request: Request):
    denied = await _check_auth(request)
    if denied:
        return denied
    from mcp_server.server import _status_payload  # 已脱敏（无密钥/路径）
    stats = _status_payload()
    return {
        "agent_name": AGENT_NAME,
        "healthy": True,
        "documents": {"source_count": stats["sources"], "chunk_count": stats["chunks"]},
        "models": {"llm": stats["decision_model"], "embedding": stats["embedding_model"]},
        "test_mode": stats["test_mode"],
    }


@app.delete("/v1/ingest/{source_id}")
async def delete_source(source_id: str, request: Request):
    """契约 v1.0 明确不含删除；显式 501 提示而非 404，便于客户端区分。"""
    return _error("invalid_request", 501, "删除能力不在契约 v1.0 内",
                  err_type="api_error")