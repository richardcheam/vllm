# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import Counter

import torch

from vllm.v1.simple_kv_offload import copy_backend


class FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cuda_stream = 0


class FakeEvent:
    def __init__(self) -> None:
        self.stream: FakeStream | None = None

    def record(self, stream: FakeStream) -> None:
        self.stream = stream


def test_dma_copy_backend_uses_independent_load_store_workers(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def fake_set_device(device: torch.device) -> None:
        calls.append(("set_device", device))

    def fake_build_params(src_caches, dst_caches, stream: FakeStream) -> str:
        return f"params-{stream.name}"

    def fake_copy_blocks(src_blocks, dst_blocks, params) -> None:
        calls.append(("copy", (tuple(src_blocks), tuple(dst_blocks), params)))

    monkeypatch.setattr(copy_backend.current_platform, "set_device", fake_set_device)
    monkeypatch.setattr(copy_backend, "build_params", fake_build_params)
    monkeypatch.setattr(copy_backend, "copy_blocks", fake_copy_blocks)
    monkeypatch.setattr(copy_backend.torch, "Event", FakeEvent)

    backend = copy_backend.DmaCopyBackend()
    load_stream = FakeStream("load")
    store_stream = FakeStream("store")
    backend.init(
        {"layer": object()},
        {"layer": object()},
        torch.device("cpu"),
        load_stream,  # type: ignore[arg-type]
        store_stream,  # type: ignore[arg-type]
    )

    load_events = []
    store_events = []
    backend.launch_copy(
        [1], [2], is_store=False, event_idx=10, events_list=load_events
    )
    backend.launch_copy(
        [3], [4], is_store=True, event_idx=11, events_list=store_events
    )
    backend.shutdown()

    assert load_events == [(10, load_events[0][1])]
    assert load_events[0][1].stream is load_stream
    assert store_events == [(11, store_events[0][1])]
    assert store_events[0][1].stream is store_stream

    copy_calls = [payload for kind, payload in calls if kind == "copy"]
    assert Counter(copy_calls) == Counter(
        [
            (((1,), (2,), "params-load")),
            (((3,), (4,), "params-store")),
        ]
    )
    assert sum(1 for kind, _ in calls if kind == "set_device") == 2


def test_inline_copy_backend_submits_immediately_without_threads(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def fake_build_params(src_caches, dst_caches, stream: FakeStream) -> str:
        return f"params-{stream.name}"

    def fake_copy_blocks(src_blocks, dst_blocks, params) -> None:
        calls.append(("copy", (tuple(src_blocks), tuple(dst_blocks), params)))

    monkeypatch.setattr(copy_backend, "build_params", fake_build_params)
    monkeypatch.setattr(copy_backend, "copy_blocks", fake_copy_blocks)
    monkeypatch.setattr(copy_backend.torch, "Event", FakeEvent)

    backend = copy_backend.InlineCopyBackend()
    load_stream = FakeStream("load")
    store_stream = FakeStream("store")
    backend.init(
        {"layer": object()},
        {"layer": object()},
        torch.device("cpu"),
        load_stream,  # type: ignore[arg-type]
        store_stream,  # type: ignore[arg-type]
    )

    load_events = []
    store_events = []
    backend.launch_copy(
        [5], [6], is_store=False, event_idx=20, events_list=load_events
    )
    backend.launch_copy(
        [7], [8], is_store=True, event_idx=21, events_list=store_events
    )
    backend.shutdown()

    assert load_events == [(20, load_events[0][1])]
    assert load_events[0][1].stream is load_stream
    assert store_events == [(21, store_events[0][1])]
    assert store_events[0][1].stream is store_stream
    assert calls == [
        ("copy", ((5,), (6,), "params-load")),
        ("copy", ((7,), (8,), "params-store")),
    ]
