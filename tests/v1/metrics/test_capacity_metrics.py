import pytest

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import SchedulerStats


@pytest.mark.cpu_test
def test_prometheus_logger_records_kv_capacity_metrics() -> None:
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m"),
        cache_config=CacheConfig(block_size=16),
        scheduler_config=SchedulerConfig(max_model_len=2048, is_encoder_decoder=False),
        observability_config=ObservabilityConfig(),
    )
    logger = PrometheusStatLogger(config)
    logger.record(
        SchedulerStats(
            num_waiting_reqs=5,
            num_capacity_waiting_reqs=1,
            num_skipped_waiting_reqs=2,
            kv_cache_total_blocks=99,
            kv_cache_used_blocks=12,
            kv_cache_free_blocks=87,
        ),
        iteration_stats=None,
    )

    assert logger.kv_cache_capacity_metrics["vllm:kv_cache_total_blocks"][0]._value.get() == 99
    assert logger.kv_cache_capacity_metrics["vllm:kv_cache_used_blocks"][0]._value.get() == 12
    assert logger.kv_cache_capacity_metrics["vllm:kv_cache_free_blocks"][0]._value.get() == 87
    assert (
        logger.gauge_waiting_by_reason["capacity"][0]._value.get()
        == 1
    )
    assert logger.gauge_waiting_by_reason["queue"][0]._value.get() == 4
    assert logger.gauge_waiting_by_reason["deferred"][0]._value.get() == 2
