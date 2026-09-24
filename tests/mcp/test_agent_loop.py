# -*- coding: utf-8 -*-
"""决策循环测试：脚本化 FakeLLM，验证循环终止、回填、上限与容错。"""
import pytest
from langchain_core.messages import AIMessage

from mcp_server.agent import AgentError, RAGAgent, SYSTEM_PROMPT


class FakeLLM:
    """按脚本顺序返回响应；记录收到的 messages 供断言。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append([type(m).__name__ for m in messages])
        return self.responses.pop(0)


def _tool_call_msg(name, args, call_id="c1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _final_msg(text):
    return AIMessage(content=text)


@pytest.fixture()
def setup(monkeypatch):
    def _make(responses, tools_results=None, max_iterations=3):
        cfg = type("C", (), {"llm_model": "m", "llm_api_key": "k",
                             "llm_base_url": None, "max_text_len": 100000,
                             "max_iterations": max_iterations})()
        fake_llm = FakeLLM(responses)
        ag = RAGAgent.__new__(RAGAgent)
        ag.config = cfg
        ag.llm = fake_llm
        ag.tools = tools_results or {}
        return ag, fake_llm

    # 跳过 __init__ 中的真实 LLM 构造
    monkeypatch.setattr(RAGAgent, "__init__", lambda self, *a, **k: None)
    return _make


def test_direct_answer_without_tools(setup):
    ag, llm = setup([_final_msg("直接回答")])
    result = ag.run("你好")
    assert result["answer"] == "直接回答"
    assert result["tool_trace"] == []


def test_one_round_tool_then_answer(setup):
    def search(query, top_k=3):
        return [{"score": 0.9, "text": "片段A", "source_id": "src_a", "source_name": "docA"}]

    ag, llm = setup([
        _tool_call_msg("search_documents", {"query": "q"}),
        _final_msg("基于片段A的回答 [docA]"),
    ], tools_results={"search_documents": search})

    result = ag.run("q")
    assert result["answer"].startswith("基于片段A")
    assert result["tool_trace"][0]["ok"] is True
    assert result["sources_used"] == [{"source_id": "src_a", "source_name": "docA"}]
    # 第二轮 messages 应包含回填的 ToolMessage
    assert "ToolMessage" in llm.calls[1]


def test_unknown_tool_recorded_not_raised(setup):
    ag, _ = setup([
        _tool_call_msg("nonexistent", {}),
        _final_msg("ok"),
    ], tools_results={})
    result = ag.run("q")
    assert result["tool_trace"][0]["ok"] is False
    assert "未知工具" in result["tool_trace"][0]["error"]
    assert result["answer"] == "ok"


def test_tool_exception_recorded_not_raised(setup):
    def broken(**kw):
        raise ValueError("boom")

    ag, _ = setup([
        _tool_call_msg("search_documents", {"query": "q"}),
        _final_msg("done"),
    ], tools_results={"search_documents": broken})
    result = ag.run("q")
    assert result["tool_trace"][0]["ok"] is False
    assert "boom" in result["tool_trace"][0]["error"]
    assert result["answer"] == "done"


def test_iteration_cap_truncates(setup):
    msgs = [_tool_call_msg("search_documents", {"query": f"q{i}"}) for i in range(5)]
    def search(query, top_k=3):
        return []

    ag, _ = setup(msgs, tools_results={"search_documents": search}, max_iterations=3)
    result = ag.run("q")
    assert "未在 3 轮内完成" in result["answer"]
    assert result["tool_trace"][-1] == {"truncated": True}
    assert len([t for t in result["tool_trace"] if "tool" in t]) == 3


def test_llm_failure_raises_agent_error(setup):
    class ExplodingLLM:
        def invoke(self, messages):
            raise RuntimeError("api down")

    ag, _ = setup([])
    ag.llm = ExplodingLLM()
    with pytest.raises(AgentError, match="api down"):
        ag.run("q")


def test_system_prompt_contains_injection_guard():
    assert "一律不执行" in SYSTEM_PROMPT
