# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SuperInfer benchmark report helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_prompt import tokenize_to_length
from scripts.report_superinfer_bench import (
    metric_counter_deltas,
    parse_metric_samples,
    transfer_summary,
)


def test_parse_metric_samples_retains_labels(tmp_path: Path):
    metrics = tmp_path / "metrics.txt"
    metrics.write_text(
        "\n".join(
            [
                "# HELP ignored ignored",
                'vllm:simple_cpu_offload_offload_load_bytes_total{engine="0"} 12',
                'vllm:simple_cpu_offload_offload_load_bytes_total{engine="1"} 34',
                "vllm:simple_cpu_offload_offload_load_queue_depth 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = parse_metric_samples(metrics)

    assert samples[
        'vllm:simple_cpu_offload_offload_load_bytes_total{engine="0"}'
    ] == 12
    assert samples[
        'vllm:simple_cpu_offload_offload_load_bytes_total{engine="1"}'
    ] == 34
    assert samples["vllm:simple_cpu_offload_offload_load_queue_depth"] == 2


def test_transfer_summary_strips_counter_suffix_and_keeps_max():
    summary = transfer_summary(
        {
            'vllm:simple_cpu_offload_offload_load_bytes_total{engine="0"}': 12,
            'vllm:simple_cpu_offload_offload_load_bytes_total{engine="1"}': 34,
            "vllm:simple_cpu_offload_offload_load_queue_depth": 2,
        }
    )

    assert summary == {
        "offload_load_bytes": 34,
        "offload_load_queue_depth": 2,
    }


def test_metric_counter_deltas_uses_before_after_repeat_snapshots(tmp_path: Path):
    before = tmp_path / "run_1.before.metrics.txt"
    after = tmp_path / "run_1.after.metrics.txt"
    before.write_text(
        'vllm:simple_cpu_offload_offload_store_bytes_total{engine="0"} 10\n'
        'vllm:simple_cpu_offload_offload_store_queue_depth{engine="0"} 1\n',
        encoding="utf-8",
    )
    after.write_text(
        'vllm:simple_cpu_offload_offload_store_bytes_total{engine="0"} 42\n'
        'vllm:simple_cpu_offload_offload_store_queue_depth{engine="0"} 0\n',
        encoding="utf-8",
    )

    deltas = metric_counter_deltas(tmp_path)

    assert deltas == {
        'vllm:simple_cpu_offload_offload_store_bytes_total{engine="0"}': 32,
    }


def test_metric_counter_deltas_excludes_cpu_allocation_counters(tmp_path: Path):
    before = tmp_path / "run_1.before.metrics.txt"
    after = tmp_path / "run_1.after.metrics.txt"
    before.write_text(
        'vllm:simple_cpu_offload_offload_cpu_kv_capacity_bytes_total{engine="0"} 10\n'
        'vllm:simple_cpu_offload_offload_store_events_total{engine="0"} 1\n',
        encoding="utf-8",
    )
    after.write_text(
        'vllm:simple_cpu_offload_offload_cpu_kv_capacity_bytes_total{engine="0"} 42\n'
        'vllm:simple_cpu_offload_offload_store_events_total{engine="0"} 4\n',
        encoding="utf-8",
    )

    deltas = metric_counter_deltas(tmp_path)

    assert deltas == {
        'vllm:simple_cpu_offload_offload_store_events_total{engine="0"}': 3,
    }


def test_tokenize_to_length_returns_exact_target():
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            del add_special_tokens
            return list(range(len(text)))

    tokens = tokenize_to_length(
        "abc",
        prompt_len=7,
        tokenizer=FakeTokenizer(),
        filler="xy",
    )

    assert len(tokens) == 7
    assert tokens[:3] == [0, 1, 2]
