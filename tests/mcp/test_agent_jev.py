# -*- coding: utf-8 -*-
"""Jev 状态机路径测试：mock Jev 与生成 LLM，验证状态流转。"""
import pytest
from langchain_core.messages import AIMessage

from mcp_server.agent import ANSWER_SYSTEM_PROMPT, AgentError, RAGAgent


class FakeJev:
    """按脚本顺序返回 sufficiency noul 分数；记录收到的 state。"""

    def __init__(self, scores):
        self.scores = list(scores)
        self.states = []

    def ask(self, state, questions):
        self.states.append(state)
        assert "sufficiency" in questions
        return {"sufficiency": {"type": "noul", "noul": self.scores.pop(0)}}


class FakeGenLLM:
    """按脚本顺序返回内容；记录收到的 messages。"""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.contents.pop(0))


def _chunk(text, source_id="src_a", name="docA"):
    return {"score": 0.9, "text": text, "source_id": source_id, "source_name": name}


@pytest.fixture()
def make_agent(monkeypatch):
    """构造 Jev 路径的 RAGAgent（跳过真实 __init__）。"""

    def _make(jev, gen_llm, search_results=None, max_iterations=3):
        cfg = type("C", (), {
            "llm_model": "m", "llm_api_key": "k", "llm_base_url": None,
            "max_text_len": 100000, "max_iterations": max_iterations,
        })()
        ag = RAGAgent.__new__(RAGAgent)
        ag.config = cfg
        ag.jev = jev
        ag._plain_llm = gen_llm
        ag.vector_db = None
        if search_results is not None:
            ag._tool_search_documents = lambda query, top_k=3: list(search_results)
        return ag

    monkeypatch.setattr(RAGAgent, "__init__", lambda self, *a, **k: None)
    return _make


def test_jev_sufficient_then_answer(make_agent):
    """首轮检索片段充分（noul≥0.5）→ 生成 LLM 直接回答。"""
    jev = FakeJev([0.92])
    gen = FakeGenLLM(["例会每周一上午十点 [docA]"])
    ag = make_agent(jev, gen, search_results=[_chunk("例会每周一上午十点召开。")])

    result = ag.run("例会是什么时间？")
    assert "例会" in result["answer"]
    assert result["sources_used"] == [{"source_id": "src_a", "source_name": "docA"}]
    # trace: search → jev
    tools = [t["tool"] for t in result["tool_trace"]]
    assert tools == ["search_documents", "jev_sufficiency"]
    assert result["tool_trace"][1]["args"]["noul"] == 0.92
    # 生成 prompt 使用 ANSWER_SYSTEM_PROMPT 且包含片段
    assert gen.calls[0][0].content == ANSWER_SYSTEM_PROMPT
    assert "例会每周一上午十点" in gen.calls[0][1].content


def test_jev_insufficient_retries_alt_query(make_agent):
    """片段不充分（noul<0.5）→ 换词再检索 → 第二轮充分 → 回答。"""
    jev = FakeJev([0.2, 0.8])
    gen = FakeGenLLM(["项目 例会 时间", "最终回答 [docA]"])
    chunks_by_call = [
        [_chunk("无关片段")],
        [_chunk("例会每周一上午十点召开。")],
    ]
    calls = {"n": 0}

    def fake_search(query, top_k=3):
        r = chunks_by_call[min(calls["n"], len(chunks_by_call) - 1)]
        calls["n"] += 1
        return list(r)

    ag = make_agent(jev, gen, max_iterations=3)
    ag._tool_search_documents = fake_search

    result = ag.run("例会时间？")
    assert result["answer"].startswith("最终回答")
    tools = [t["tool"] for t in result["tool_trace"]]
    assert tools == [
        "search_documents", "jev_sufficiency",
        "search_documents", "jev_sufficiency",
    ]
    # 两次检索用了不同 query（第二次是 LLM 生成的替代词）
    queries = [t["args"]["query"] for t in result["tool_trace"]
               if t["tool"] == "search_documents"]
    assert queries[0] == "例会时间？"
    assert queries[1] == "项目 例会 时间"
    # 第一次 LLM 调用是生成替代词（含已试列表）
    assert "已试过的检索词" in gen.calls[0][1].content


def test_jev_no_chunks_gives_up(make_agent):
    """检索一直为空 → 如实放弃，不伪造来源。"""
    jev = FakeJev([])
    gen = FakeGenLLM(["替代词"])
    ag = make_agent(jev, gen, search_results=[], max_iterations=2)

    result = ag.run("库外问题？")
    assert "没有找到" in result["answer"]
    assert result["sources_used"] == []
    # 换词一次后仍空 → give up（初始 + 替代词共 2 次 search）
    assert len([t for t in result["tool_trace"] if t["tool"] == "search_documents"]) == 2


def test_jev_search_failure_recorded(make_agent):
    """检索工具抛异常 → trace 记录 ok=False，按无片段处理。"""
    jev = FakeJev([])

    def boom(query, top_k=3):
        raise ValueError("milvus down")

    ag = make_agent(jev, FakeGenLLM(["替代词"]))
    ag._tool_search_documents = boom

    result = ag.run("q")
    search_entries = [t for t in result["tool_trace"] if t["tool"] == "search_documents"]
    assert search_entries[0]["ok"] is False
    assert "milvus down" in search_entries[0]["error"]


def test_jev_error_wrapped_as_agent_error(make_agent):
    """Jev 调用失败 → AgentError，不裸抛。"""
    class ExplodingJev:
        def ask(self, state, questions):
            raise RuntimeError("jev api down")

    ag = make_agent(ExplodingJev(), FakeGenLLM([]),
                    search_results=[_chunk("片段")])
    with pytest.raises(AgentError, match="Jev 决策循环失败"):
        ag.run("q")


def test_jev_rounds_exhausted_insufficient(make_agent):
    """noul 一直低 → 轮次耗尽 → 返回"片段不足"。"""
    jev = FakeJev([0.1, 0.1, 0.1])
    gen = FakeGenLLM(["alt1", "alt2", "alt3"])
    ag = make_agent(jev, gen, search_results=[_chunk("弱相关片段")],
                    max_iterations=3)

    result = ag.run("很难的问题")
    assert "不足以完整回答" in result["answer"]
    # 3 轮判断 + 3 次检索
    assert len([t for t in result["tool_trace"] if t["tool"] == "jev_sufficiency"]) == 3


def test_jev_state_contains_query_and_chunks(make_agent):
    """Jev 收到的 state 应包含用户问题与片段文本。"""
    jev = FakeJev([0.9])
    ag = make_agent(jev, FakeGenLLM(["a"]),
                    search_results=[_chunk("关键内容：例会周一十点")])
    ag.run("例会时间？")
    state = jev.states[0]
    assert "例会时间？" in state
    assert "例会周一十点" in state
