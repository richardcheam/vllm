# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for staged serving-soak report aggregation."""

from pathlib import Path

from scripts.report_serving_soak import phase_summary


def test_phase_summary_aggregates_throughput_and_transfer_deltas():
    results = [
        {
            "status": "PASS",
            "successful": 8,
            "failed": 0,
            "transfer_deltas": {
                "store_bytes": 100,
                "load_bytes": 20,
                "store_events": 1,
                "load_events": 1,
                "native_store_submissions": 0,
                "native_load_submissions": 0,
            },
            "final_gauges": {"dirty_blocks": 2, "invalid_blocks": 0},
            "transfer_quiescent_at_capture": True,
            "run": {"api_completion_tokens_per_second": 80},
        },
        {
            "status": "PASS",
            "successful": 8,
            "failed": 0,
            "transfer_deltas": {
                "store_bytes": 120,
                "load_bytes": 30,
                "store_events": 2,
                "load_events": 1,
                "native_store_submissions": 0,
                "native_load_submissions": 0,
            },
            "final_gauges": {"dirty_blocks": 3, "invalid_blocks": 0},
            "transfer_quiescent_at_capture": False,
            "run": {"api_completion_tokens_per_second": 100},
        },
    ]

    summary = phase_summary(results)

    assert summary["status"] == "PASS"
    assert summary["successful"] == 16
    assert summary["throughput_median"] == 90
    assert summary["transfer_deltas"]["store_bytes"] == 220
    assert summary["transfer_deltas"]["load_bytes"] == 50
    assert summary["max_dirty_blocks"] == 3
    assert summary["quiescent_captures"] == 1


def test_phase_summary_handles_reload_validation():
    summary = phase_summary(
        [
            {
                "status": "PASS",
                "validation": {
                    "status": "PASS",
                    "metric_deltas": {
                        "store_bytes": 100,
                        "load_bytes": 200,
                    },
                },
                "phase_timings_s": {"reload": 1.5},
                "reload": {"status": "PASS"},
            }
        ]
    )

    assert summary["status"] == "PASS"
    assert summary["transfer_deltas"] == {
        "load_bytes": 200,
        "store_bytes": 100,
    }
    assert summary["phase_timings_s"] == {"reload": 1.5}
