# -*- coding: utf-8 -*-
"""契约符合性测试（swap test）——对 docs/api-contract.md v1.0 的可机器验证子集。

对任意实现了该契约的 Agent 后端运行：

    export POLARIS_CONTRACT_BASE_URL="http://127.0.0.1:8000"
    export POLARIS_CONTRACT_API_KEY="sk-..."        # 可选
    pytest tests/contract -v

未设置 POLARIS_CONTRACT_BASE_URL 时整组跳过（不影响常规单测）。
只依赖 httpx（项目 venv 已有），不引入新依赖。
"""
import json
import os
import re

import pytest
import httpx

# ---------------------------------------------------------------- 环境与夹具

BASE_URL = os.environ.get("POLARIS_CONTRACT_BASE_URL", "")
API_KEY = os.environ.get("POLARIS_CONTRACT_API_KEY", "")
TIMEOUT = httpx.Timeout(60.0, read=90.0)

pytestmark = pytest.mark.skipif(
    not BASE_URL, reason="POLARIS_CONTRACT_BASE_URL 未设置，跳过契约测试"
)

CONTRACT_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")

ALLOWED_ERROR_CODES = {
    "invalid_request",
    "auth_failed",
    "rate_limited",
    "context_overflow",
    "upstream_unavailable",
    "internal_error",
}

# 事件类型词汇表：SSE event 名 + data.type（message.delta 是 message 事件的子类型）
ALLOWED_SSE_EVENT_TYPES = {"message", "message.delta", "sources", "done", "error"}

EMBEDDED_CITATION_PATTERNS = [re.compile(p) for p in (r"【来源", r"\[1\]", r"\[2\]", r"\[来源")]
INTERNAL_LEAK_PATTERNS = [re.compile(p) for p in (r"milvus_data\.db", r"/Users/", r"sk-")]


@pytest.fixture(scope="module")
def client():
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="module")
def capabilities(client):
    """必需端点：所有测试共享；A1 失败时后续全部跳过。"""
    resp = client.get("/v1/capabilities")
    if resp.status_code != 200:
        pytest.fail(f"/v1/capabilities 不可用：HTTP {resp.status_code}（契约必需端点）", pytrace=False)
    cap = resp.json()
    if not CONTRACT_VERSION_PATTERN.match(str(cap.get("contract_version", ""))):
        pytest.fail(f"contract_version 非法：{cap.get('contract_version')!r}", pytrace=False)
    return cap


def _assert_error_envelope(payload: dict):
    """D1：错误信封结构与受控错误码。"""
    assert "error" in payload, f"非契约错误信封：{payload}"
    err = payload["error"]
    assert isinstance(err.get("message"), str) and err["message"]
    code = err.get("code")
    assert code in ALLOWED_ERROR_CODES, f"错误码 {code!r} 不在受控词汇表 {sorted(ALLOWED_ERROR_CODES)}"


def _assert_chat_core_schema(payload: dict):
    """B1：核心 OpenAI 字段齐全 + content 纯净 + citations 通用化。"""
    assert payload.get("object") == "chat.completion"
    assert isinstance(payload.get("id"), str) and payload["id"]
    choices = payload.get("choices")
    assert isinstance(choices, list) and choices, "choices 为空"
    choice = choices[0]
    assert choice.get("index") == 0
    msg = choice.get("message") or {}
    assert msg.get("role") == "assistant"
    content = msg.get("content")
    assert isinstance(content, str) and content.strip(), "content 必须是非空纯文本"
    for pat in EMBEDDED_CITATION_PATTERNS:
        assert not pat.search(content), f"content 疑似内嵌引用（违反 P2）：匹配 {pat.pattern}"
    assert choice.get("finish_reason") in {"stop", "length", "error"}
    citations = payload.get("citations")
    if citations:
        assert isinstance(citations, list)
        for cit in citations:
            assert cit.get("id") and cit.get("snippet") is not None, \
                f"citations 元素缺 id/snippet：{cit}"
            uri = str(cit.get("uri", ""))
            for pat in INTERNAL_LEAK_PATTERNS[:2]:
                assert not pat.search(uri), f"citations.uri 疑似泄露内部细节：{uri}"


def _parse_sse(text: str):
    """极简 SSE 解析：(event, data_json) 列表；忽略注释与空行。"""
    events = []
    for block in text.split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if event or data:
            try:
                data = json.loads(data) if data else None
            except json.JSONDecodeError:
                data = {"_raw": data}
            events.append((event, data))
    return events


# ---------------------------------------------------------------- A1 capabilities

class TestCapabilities:
    def test_a1_schema(self, capabilities):
        assert isinstance(capabilities.get("agent_name"), str) and capabilities["agent_name"]
        assert capabilities.get("sessions") in {"client", "server"}
        assert capabilities.get("max_context_messages") is None or isinstance(
            capabilities["max_context_messages"], int
        )
        extra = capabilities.get("capabilities_extra") or {}
        assert isinstance(extra, dict)

    def test_a1_undeclared_optional_endpoints_return_404(self, client, capabilities):
        extra = capabilities.get("capabilities_extra") or {}
        # extra 形如 {"ingest": "/v1/ingest"}：键与值都视为"已声明"
        declared = set(extra.keys()) | set(extra.values())
        for name in ("/v1/ingest", "/v1/status"):
            if name in declared:
                continue
            resp = client.post(name, json={}) if name.endswith("ingest") else client.get(name)
            assert resp.status_code in (404, 405), (
                f"未声明的能力门控端点 {name} 返回了 {resp.status_code}（应为 404/405）"
            )


# ---------------------------------------------------------------- B1 非流式

class TestChatCompletions:
    def test_b1_non_streaming(self, client, capabilities):
        payload = {
            "model": "contract-test",
            "messages": [{"role": "user", "content": "请用一句话回答：1+1等于几？"}],
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        _assert_chat_core_schema(body)
        if capabilities.get("citations"):
            assert body.get("citations") is not None, "capabilities.citations=true 时应返回 citations"

    def test_d1_empty_messages_rejected(self, client):
        resp = client.post(
            "/v1/chat/completions", json={"model": "x", "messages": [], "stream": False}
        )
        assert resp.status_code == 400
        _assert_error_envelope(resp.json())

    def test_d1_invalid_role_rejected(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "wizard", "content": "hi"}]},
        )
        assert resp.status_code == 400
        _assert_error_envelope(resp.json())

    def test_d1_auth_enforced_when_key_configured(self, client):
        if not API_KEY:
            pytest.skip("未配置 POLARIS_CONTRACT_API_KEY，跳过鉴权用例")
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as anon:
            resp = anon.post(
                "/v1/chat/completions",
                json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 401
        _assert_error_envelope(resp.json())


# ---------------------------------------------------------------- C1 流式

class TestStreaming:
    def test_c1_typed_event_stream(self, client, capabilities):
        if not capabilities.get("streaming"):
            pytest.skip("实现方声明 streaming=false")
        events_seen, terminated, deltas = [], False, []
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "contract-test",
                "messages": [{"role": "user", "content": "请用一句话回答：1+1等于几？"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200, "流式请求应返回 200（校验类错误应走 4xx 而非事件流）"
            buffer = ""
            for chunk in resp.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    for event, data in _parse_sse(block + "\n\n"):
                        events_seen.append((event, data))
                        etype = (data or {}).get("type") or event
                        assert etype in ALLOWED_SSE_EVENT_TYPES, f"非法事件类型 {etype!r}"
                        if etype in {"message", "message.delta"}:
                            assert isinstance((data or {}).get("delta"), str)
                            deltas.append(data["delta"])
                        elif etype in {"done", "error"}:
                            terminated = True
        assert terminated, "事件流必须以 done 或 error 收尾"
        if any(e in {"message", "message.delta"} for e, _ in events_seen):
            assert "".join(deltas).strip(), "message.delta 拼接应为非空答案"

    def test_c1_error_event_after_stream_start(self, client, capabilities):
        """已进入事件流后的失败必须用 error 事件而非断流。此用例尽力而为：仅校验正常流不触发。"""
        if not capabilities.get("streaming"):
            pytest.skip("实现方声明 streaming=false")
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "contract-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            text = "".join(chunk for chunk in resp.iter_text())
        events = _parse_sse(text)
        assert events, "SSE 事件流为空"
        assert any(e in {"done", "error"} for e, _ in events), "缺少终止事件"


# ---------------------------------------------------------------- E1 可选端点（能力门控）

class TestOptionalEndpoints:
    def test_e1_ingest(self, client, capabilities):
        extra = capabilities.get("capabilities_extra") or {}
        if "ingest" not in extra:
            pytest.skip("实现方未声明 ingest 能力")
        resp = client.post(
            "/v1/ingest",
            json={"text": "契约测试文本：PolarisRAG swap test。" * 10, "source_name": "contract-test"},
        )
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert isinstance(body.get("chunk_ids"), list) and body["chunk_ids"]
        assert isinstance(body.get("chunk_count"), int)

    def test_e1_status_no_secret_leak(self, client, capabilities):
        extra = capabilities.get("capabilities_extra") or {}
        if "status" not in extra:
            pytest.skip("实现方未声明 status 能力")
        resp = client.get("/v1/status")
        assert resp.status_code == 200
        text = resp.text
        for pat in INTERNAL_LEAK_PATTERNS:
            assert not pat.search(text), f"/v1/status 疑似泄露敏感信息：匹配 {pat.pattern}"
