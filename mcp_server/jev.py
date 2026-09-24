# -*- coding: utf-8 -*-
"""Jev（typesafe systemone）API 客户端。

Jev 是确定性判断模型：state + questions → 每问题一个确定性答案。
不是 chat 接口，无自由文本输出，用于 RAG 决策循环中的"判断"节点。

问题类型（探测于 2026-09-21，jev-1.13.0）：
- choice: {type, instructions, criteria: {<key>: <描述>}} → {choice, confidence, probabilities}
- noul:   {type, instructions}                            → {noul: float}
"""
import logging
from typing import Any, Dict, Optional

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 30.0


class JevError(Exception):
    """Jev 调用失败（网络、非 200、响应结构异常）。"""


class JevClient:
    """Jev systemone API 封装：ask() 一次请求可含多个问题。"""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key:
            raise JevError("Jev api_key 不能为空")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ask(self, state: str, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """执行一次判断请求。

        Args:
            state: 待判断的状态文本（问题描述 + 已有检索结果摘要）
            questions: 问题定义；key 为问题名，value 形如
                {"type": "choice", "instructions": "...", "criteria": {...}}
                或 {"type": "noul", "instructions": "..."}

        Returns:
            {问题名: {"type": "choice", "choice": str, "confidence": float,
                      "probabilities": {...}}}
            或 {问题名: {"type": "noul", "noul": float}}

        Raises:
            JevError: 网络/超时、HTTP 非 200、响应缺 answers、问题键缺失、
                      choice 值不在 criteria 中
        """
        if not state or not state.strip():
            raise JevError("state 不能为空")
        if not questions:
            raise JevError("questions 不能为空")

        payload = {"state": state, "model": self.model, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                self.base_url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise JevError(f"Jev 请求失败: {e}") from e

        if resp.status_code != 200:
            raise JevError(
                f"Jev HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise JevError(f"Jev 响应非 JSON: {resp.text[:200]}") from e

        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise JevError(f"Jev 响应缺少 answers 对象: {data}")

        return {
            name: self._parse_answer(name, questions[name], answers.get(name))
            for name in questions
        }

    @staticmethod
    def _parse_answer(
        name: str, question: Dict[str, Any], raw: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise JevError(f"Jev 响应缺少问题 {name!r} 的答案: {raw!r}")

        qtype = question.get("type")
        if qtype == "choice":
            criteria = question.get("criteria", {})
            choice = raw.get("choice")
            if choice not in criteria:
                raise JevError(
                    f"Jev 问题 {name!r} 的 choice {choice!r} 不在 criteria {list(criteria)} 中"
                )
            return {
                "type": "choice",
                "choice": choice,
                "confidence": raw.get("confidence"),
                "probabilities": raw.get("probabilities", {}),
            }
        if qtype == "noul":
            noul = raw.get("noul")
            if not isinstance(noul, (int, float)) or isinstance(noul, bool):
                raise JevError(f"Jev 问题 {name!r} 的 noul 非数值: {noul!r}")
            return {"type": "noul", "noul": float(noul)}

        raise JevError(f"问题 {name!r} 类型不支持: {qtype!r}")
