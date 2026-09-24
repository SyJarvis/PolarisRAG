# -*- coding: utf-8 -*-
"""JevClient 单测：全部 mock requests，无 key 可跑。"""
import pytest
import requests as requests_mod

from mcp_server.jev import JevClient, JevError


def _questions():
    return {
        "action": {
            "type": "choice",
            "instructions": "下一步做什么？",
            "criteria": {"answer": "直接回答", "search": "再次检索", "give_up": "放弃"},
        },
        "sufficiency": {"type": "noul", "instructions": "片段是否充分？"},
    }


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (str(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _ok_payload():
    return {
        "model": "jev-1.13.0",
        "answers": {
            "action": {
                "type": "choice",
                "choice": "answer",
                "confidence": 1.0,
                "probabilities": {"answer": 1.0, "search": 0.0, "give_up": 0.0},
            },
            "sufficiency": {"type": "noul", "noul": 0.99},
        },
        "usage": {"input_tokens": 434, "output_tokens": 59},
    }


def _client():
    return JevClient(api_key="apikey_test", model="jev-latest",
                     base_url="https://api.typesafe.ai/v1/systemone", timeout=5.0)


def test_ask_parses_choice_and_noul(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return _FakeResponse(payload=_ok_payload())

    monkeypatch.setattr(requests_mod, "post", fake_post)
    result = _client().ask("state 文本", _questions())

    assert result["action"]["type"] == "choice"
    assert result["action"]["choice"] == "answer"
    assert result["action"]["confidence"] == 1.0
    assert result["sufficiency"] == {"type": "noul", "noul": 0.99}

    # 请求结构
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer apikey_test"
    assert captured["json"] == {
        "state": "state 文本",
        "model": "jev-latest",
        "questions": _questions(),
    }
    assert captured["timeout"] == 5.0


def test_ask_empty_state_rejected():
    with pytest.raises(JevError, match="state"):
        _client().ask("  ", _questions())


def test_ask_empty_questions_rejected():
    with pytest.raises(JevError, match="questions"):
        _client().ask("state", {})


def test_init_empty_key_rejected():
    with pytest.raises(JevError, match="api_key"):
        JevClient(api_key="")


def test_http_error_raises(monkeypatch):
    monkeypatch.setattr(
        requests_mod, "post",
        lambda *a, **k: _FakeResponse(status_code=401, text="unauthorized"),
    )
    with pytest.raises(JevError, match="401"):
        _client().ask("s", _questions())


def test_network_error_raises(monkeypatch):
    def boom(*a, **k):
        raise requests_mod.ConnectionError("conn refused")

    monkeypatch.setattr(requests_mod, "post", boom)
    with pytest.raises(JevError, match="请求失败"):
        _client().ask("s", _questions())


def test_non_json_response_raises(monkeypatch):
    monkeypatch.setattr(
        requests_mod, "post", lambda *a, **k: _FakeResponse(payload=None)
    )
    with pytest.raises(JevError, match="非 JSON"):
        _client().ask("s", _questions())


def test_missing_answers_raises(monkeypatch):
    monkeypatch.setattr(
        requests_mod, "post", lambda *a, **k: _FakeResponse(payload={"model": "x"})
    )
    with pytest.raises(JevError, match="answers"):
        _client().ask("s", _questions())


def test_missing_question_answer_raises(monkeypatch):
    payload = _ok_payload()
    del payload["answers"]["sufficiency"]
    monkeypatch.setattr(
        requests_mod, "post", lambda *a, **k: _FakeResponse(payload=payload)
    )
    with pytest.raises(JevError, match="sufficiency"):
        _client().ask("s", _questions())


def test_choice_outside_criteria_raises(monkeypatch):
    payload = _ok_payload()
    payload["answers"]["action"]["choice"] = "unknown_action"
    monkeypatch.setattr(
        requests_mod, "post", lambda *a, **k: _FakeResponse(payload=payload)
    )
    with pytest.raises(JevError, match="criteria"):
        _client().ask("s", _questions())


def test_noul_non_numeric_raises(monkeypatch):
    payload = _ok_payload()
    payload["answers"]["sufficiency"]["noul"] = "high"
    monkeypatch.setattr(
        requests_mod, "post", lambda *a, **k: _FakeResponse(payload=payload)
    )
    with pytest.raises(JevError, match="noul"):
        _client().ask("s", _questions())


def test_unsupported_question_type_raises(monkeypatch):
    questions = {"bad": {"type": "essay", "instructions": "..."}}
    payload = {"answers": {"bad": {"essay": "some text"}}}
    monkeypatch.setattr(
        requests_mod, "post", lambda *a, **k: _FakeResponse(payload=payload)
    )
    with pytest.raises(JevError, match="essay"):
        _client().ask("s", questions)


def test_base_url_trailing_slash_stripped():
    c = JevClient(api_key="k", base_url="https://x.example/v1/systemone/")
    assert c.base_url == "https://x.example/v1/systemone"
