# -*- coding: utf-8 -*-
"""官方 MCP Client 协议验证（无 key：POLARIS_FAKE_EMBEDDINGS=1）。

参照 MCP 手册样例 examples/python-sdk-v2/validate.py。
启动 mcp_server/server.py 子进程（stdio），断言发现、列表、调用与失败路径。
stdout 输出单个 JSON 摘要；退出码 0=passed / 1=failed。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters

LOGGER = logging.getLogger("polaris-mcp-validation")
SERVER_PATH = Path(__file__).resolve().parents[2] / "mcp_server" / "server.py"


class ValidationFailure(RuntimeError):
    pass


def text_values(items: Any) -> list[str]:
    out = []
    for item in items:
        t = getattr(item, "text", None)
        if t is not None:
            out.append(str(t))
    return out


async def validate() -> dict[str, Any]:
    home = tempfile.mkdtemp(prefix="polaris_mcp_validate_")
    env = dict(os.environ)
    env.update({
        "POLARIS_FAKE_EMBEDDINGS": "1",
        "POLARIS_MCP_HOME": home,
        "LLM_API_KEY": "",      # fake 模式应跳过
        "EMBEDDING_API_KEY": "",
    })
    try:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env=env,
            cwd=str(SERVER_PATH.parents[1]),
        )
        async with Client(params) as client:
            if client.session.discover_result is None:
                raise ValidationFailure("server/discover result is missing")

            tools = (await client.list_tools()).tools
            tool_names = sorted(t.name for t in tools)
            for expected in ("rag_add_text", "rag_query", "rag_status"):
                if expected not in tool_names:
                    raise ValidationFailure(f"tool {expected} missing: {tool_names!r}")

            resources = (await client.list_resources()).resources
            uris = [str(r.uri) for r in resources]
            if "polaris://status" not in uris:
                raise ValidationFailure(f"resource polaris://status missing: {uris!r}")

            # rag_status
            status_call = await client.call_tool("rag_status", {})
            status_text = text_values(status_call.content)
            if status_call.is_error or not status_text:
                raise ValidationFailure(f"rag_status failed: {status_text!r}")
            status = json.loads(status_text[0])
            if not status.get("test_mode"):
                raise ValidationFailure(f"test_mode 未标记: {status!r}")

            # rag_add_text（fake embedding 入库）
            add_call = await client.call_tool("rag_add_text", {
                "text": "PolarisRAG MCP validation 文档。BERT 是双向编码器表示模型，"
                        "通过掩码语言建模预训练。" * 5,
                "source_name": "validation-doc",
            })
            add_text = text_values(add_call.content)
            if add_call.is_error or not add_text:
                raise ValidationFailure(f"rag_add_text failed: {add_text!r}")
            added = json.loads(add_text[0])
            if added.get("mode") != "fast" or added.get("chunks", 0) < 1:
                raise ValidationFailure(f"rag_add_text 结果异常: {added!r}")

            # rag_query（FakeAgent）
            query_call = await client.call_tool("rag_query", {"query": "什么是 BERT"})
            query_textv = text_values(query_call.content)
            if query_call.is_error or not query_textv:
                raise ValidationFailure(f"rag_query failed: {query_textv!r}")
            answer = json.loads(query_textv[0])
            if "answer" not in answer:
                raise ValidationFailure(f"rag_query 无 answer: {answer!r}")

            # resource 读取
            res_read = await client.read_resource("polaris://status")
            res_text = text_values(res_read.contents)
            if not res_text or "test_mode" not in res_text[0]:
                raise ValidationFailure(f"resource 读取异常: {res_text!r}")

            # 非法参数：空 query
            try:
                bad = await client.call_tool("rag_query", {"query": ""})
            except Exception as exc:
                bad_summary: dict[str, Any] = {
                    "outcome": "exception", "error_type": type(exc).__name__}
            else:
                bad_summary = {"outcome": "result",
                               "is_error": bool(bad.is_error),
                               "content": text_values(bad.content)}
                if not bad.is_error:
                    raise ValidationFailure("空 query 被接受为成功调用")

            return {
                "status": "passed",
                "sdk": "mcp==2.2.0",
                "transport": "stdio",
                "protocol_version": client.protocol_version,
                "tools": tool_names,
                "resources": uris,
                "rag_status": status,
                "rag_add_text": added,
                "rag_query": {"answer_head": answer["answer"][:80],
                              "sources_used": answer.get("sources_used")},
                "invalid_argument": bad_summary,
            }
    finally:
        shutil.rmtree(home, ignore_errors=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    try:
        summary = anyio.run(validate)
    except Exception as exc:
        LOGGER.exception("validation failed")
        print(json.dumps({"status": "failed",
                          "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
