# -*- coding: utf-8 -*-
"""Shared fixture for the v19 tests: ONE fake market session per test module (building it takes
~2 s), and fresh copies of ai_tools' module-level sample stores for every test, so no test sees
another test's samples. The old test files build their own tiny state and are not affected."""
import copy
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import ai_tools as T   # noqa: E402

_STORES = ("_SAMPLES", "_ISAMPLES", "_PACE", "_SAMPLE_META", "EXTRA_INDICES")


@pytest.fixture(scope="module")
def market_base():
    from fake_market import Market
    saved = {k: getattr(T, k) for k in _STORES}
    for k in _STORES:
        setattr(T, k, {})
    T._PACE.update({"day": None, "last": {}, "events": {}, "runs": [], "loaded": False, "hb": 0, "hl": {}})
    T._SAMPLE_META.update({"since": None, "last": None, "n": 0})
    m = Market(tempfile.mkdtemp(prefix="v19_"))
    snap = {k: copy.deepcopy(getattr(T, k)) for k in _STORES}
    yield m, snap
    for k, v in saved.items():
        setattr(T, k, v)


@pytest.fixture
def mkt(market_base, monkeypatch):
    m, snap = market_base
    for k in _STORES:
        monkeypatch.setattr(T, k, copy.deepcopy(snap[k]))
    return m
