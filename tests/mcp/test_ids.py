# -*- coding: utf-8 -*-
import re

from mcp_server.ids import new_source_id, next_chunk_id


def test_chunk_ids_unique():
    ids = [next_chunk_id() for _ in range(1000)]
    assert len(set(ids)) == 1000


def test_chunk_id_positive_int64():
    v = next_chunk_id()
    assert isinstance(v, int)
    assert 0 < v < 2 ** 63


def test_source_id_format():
    sid = new_source_id()
    assert re.fullmatch(r"src_[0-9a-f]{6}", sid)


def test_source_ids_unique():
    assert len({new_source_id() for _ in range(200)}) == 200
