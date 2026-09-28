# SPDX-License-Identifier: Apache-2.0
"""Opt-in SuperInfer connector entry point for the vLLM V1 KV API.

This first integration layer deliberately reuses the v0.30 CPU offload
implementation. SuperInfer-specific policy and topology extensions can be
added behind ``kv_connector_extra_config`` without changing the vanilla path.
"""

from math import isfinite
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.superinfer_stats import (
    SuperInferConnectorStats,
    SuperInferPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.superinfer_topology import (
    GpuCpuLocalityMapping,
    discover_gpu_cpu_locality,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_VALID_COPY_BACKENDS = {"dma"}
_VALID_TOPOLOGY_MODES = {"disabled", "auto", "required"}


def validate_superinfer_config(extra_config: dict[str, Any]) -> None:
    """Validate connector-owned SuperInfer options before construction."""
    schema_version = int(extra_config.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"Unsupported SuperInfer config schema: {schema_version}")

    capacity_gb = extra_config.get("capacity_gb")
    cpu_bytes_to_use = extra_config.get("cpu_bytes_to_use")
    cpu_bytes_to_use_per_rank = extra_config.get("cpu_bytes_to_use_per_rank")
    if cpu_bytes_to_use is not None and int(cpu_bytes_to_use) <= 0:
        raise ValueError("SuperInfer cpu_bytes_to_use must be positive")
    if (
        cpu_bytes_to_use_per_rank is not None
        and int(cpu_bytes_to_use_per_rank) <= 0
    ):
        raise ValueError("SuperInfer cpu_bytes_to_use_per_rank must be positive")
    if capacity_gb is not None:
        capacity_bytes = float(capacity_gb) * (1024**3)
        if not isfinite(capacity_bytes) or capacity_bytes <= 0:
            raise ValueError("SuperInfer capacity_gb must be positive")
        if cpu_bytes_to_use_per_rank is not None or (
            cpu_bytes_to_use is not None
            and int(cpu_bytes_to_use) != int(capacity_bytes)
        ):
            raise ValueError(
                "Specify only one capacity option, except equivalent "
                "capacity_gb and cpu_bytes_to_use values"
            )

    allowed_options = {
        "schema_version",
        "capacity_gb",
        "cpu_bytes_to_use",
        "cpu_bytes_to_use_per_rank",
        "copy_backend",
        "allocation",
        "topology",
        "policy",
        "kv_offload_backend",
        "cpu_kv_allocation_mode",
        "lazy_offload",
    }
    unsupported_options = set(extra_config) - allowed_options
    unsupported_options.update(
        {"transfer_queue_depth", "metrics"}.intersection(extra_config)
    )
    if unsupported_options:
        raise ValueError(
            "Unsupported SuperInfer options in v0.30: "
            f"{', '.join(sorted(unsupported_options))}"
        )

    copy_backend = str(extra_config.get("copy_backend", "dma"))
    if copy_backend not in _VALID_COPY_BACKENDS:
        raise ValueError(
            f"Unsupported SuperInfer copy_backend: {copy_backend!r}; "
            f"expected one of {sorted(_VALID_COPY_BACKENDS)}"
        )

    allocation = extra_config.get("allocation", {})
    if not isinstance(allocation, dict):
        raise ValueError("SuperInfer allocation must be an object")
    _reject_unknown_keys(allocation, {"mode"}, "allocation")
    allocation_mode = str(allocation.get("mode", "auto"))
    if allocation_mode not in {"auto", "zero", "empty"}:
        raise ValueError(f"Unsupported SuperInfer allocation mode: {allocation_mode}")
    translated_allocation = "zero" if allocation_mode == "auto" else allocation_mode
    if (
        "cpu_kv_allocation_mode" in extra_config
        and extra_config["cpu_kv_allocation_mode"] != translated_allocation
    ):
        raise ValueError("cpu_kv_allocation_mode conflicts with allocation.mode")

    topology = extra_config.get("topology", {})
    if not isinstance(topology, dict):
        raise ValueError("SuperInfer topology must be an object")
    _reject_unknown_keys(
        topology,
        {
            "mode",
            "local_cpu_pool_fraction",
            "local_bandwidth_bytes_per_s",
            "remote_bandwidth_bytes_per_s",
        },
        "topology",
    )
    topology_mode = str(topology.get("mode", "auto"))
    if topology_mode not in _VALID_TOPOLOGY_MODES:
        raise ValueError(f"Unsupported SuperInfer topology mode: {topology_mode}")

    policy = extra_config.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("SuperInfer policy must be an object")
    _reject_unknown_keys(policy, {"mode", "proactive_budget"}, "policy")
    policy_mode = str(policy.get("mode", "conservative"))
    if policy_mode != "conservative":
        raise ValueError(
            "Only policy.mode='conservative' is supported; scheduler policy "
            "actions are not implemented in v0.30"
        )
    budget = int(policy.get("proactive_budget", 0))
    if budget != 0:
        raise ValueError("SuperInfer proactive_budget must be 0 in v0.30")
    if extra_config.get("kv_offload_backend", "cpu") != "cpu":
        raise ValueError(
            "SuperInfer v0.30 currently supports kv_offload_backend='cpu'"
        )


def _reject_unknown_keys(
    values: dict[str, Any], allowed: set[str], section: str
) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"Unsupported SuperInfer {section} options: {', '.join(sorted(unknown))}"
        )


def prepare_superinfer_extra_config(extra_config: dict[str, Any]) -> dict[str, Any]:
    """Validate and translate supported options to the v0.30 CPU connector."""
    prepared = dict(extra_config)
    validate_superinfer_config(prepared)

    capacity_gb = prepared.get("capacity_gb")
    if capacity_gb is not None:
        prepared["cpu_bytes_to_use"] = int(float(capacity_gb) * (1024**3))
    prepared.setdefault("kv_offload_backend", "cpu")
    allocation_mode = prepared.get("allocation", {}).get("mode", "auto")
    prepared["cpu_kv_allocation_mode"] = (
        "zero" if allocation_mode == "auto" else allocation_mode
    )
    return prepared


class SuperInferConnector(SimpleCPUOffloadConnector):
    """Explicit SuperInfer entry point backed by vLLM's V1 CPU connector."""

    def __init__(self, vllm_config, role, kv_cache_config):
        raw_extra_config = dict(
            vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        )
        extra_config = prepare_superinfer_extra_config(raw_extra_config)
        vllm_config.kv_transfer_config.kv_connector_extra_config = extra_config

        logger.info(
            "SuperInferConnector enabled: policy=%s copy_backend=%s topology=%s",
            extra_config.get("policy", {}).get("mode", "conservative"),
            extra_config.get("copy_backend", "dma"),
            extra_config.get("topology", {}).get("mode", "auto"),
        )
        self.superinfer_config = extra_config
        topology = extra_config.get("topology", {})
        self.locality_mapping: GpuCpuLocalityMapping = discover_gpu_cpu_locality(
            mode=str(topology.get("mode", "auto")),
            local_cpu_pool_fraction=float(
                topology.get("local_cpu_pool_fraction", 0.75)
            ),
            local_bandwidth_bytes_per_s=topology.get(
                "local_bandwidth_bytes_per_s"
            ),
            remote_bandwidth_bytes_per_s=topology.get(
                "remote_bandwidth_bytes_per_s"
            ),
        )
        super().__init__(vllm_config, role, kv_cache_config)

    @property
    def superinfer_policy_enabled(self) -> bool:
        """Whether SuperInfer currently changes scheduler policy."""
        return False

    @property
    def superinfer_proactive_budget(self) -> int:
        """Return the configured policy budget without changing scheduling."""
        return int(
            self.superinfer_config.get("policy", {}).get("proactive_budget", 0)
        )

    def get_superinfer_capabilities(self) -> dict[str, object]:
        """Expose connector-owned policy state for a later scheduler adapter."""
        topology = self.superinfer_config.get("topology", {})
        return {
            "policy_mode": self.superinfer_config.get("policy", {}).get(
                "mode", "conservative"
            ),
            "proactive_budget": self.superinfer_proactive_budget,
            "copy_backend": self.superinfer_config.get("copy_backend", "dma"),
            "topology_mode": topology.get("mode", "auto"),
            "topology": self.locality_mapping.snapshot(),
            "proactive_policy_active": self.superinfer_policy_enabled,
        }

    def get_superinfer_telemetry(self) -> dict[str, int | float]:
        """Return interval observations from whichever connector role is active."""
        telemetry: dict[str, int | float] = {}
        if self.scheduler_manager is not None:
            telemetry.update(self.scheduler_manager.take_superinfer_telemetry())
        if self.worker_handler is not None:
            telemetry.update(self.worker_handler.take_superinfer_telemetry())
        return telemetry

    def get_kv_connector_stats(self):
        telemetry = self.get_superinfer_telemetry()
        if not telemetry:
            return None
        stats = SuperInferConnectorStats()
        stats.record(telemetry)
        return stats

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        return SuperInferConnectorStats(data=data or {})

    @classmethod
    def build_prom_metrics(
        cls, vllm_config, metric_types, labelnames, per_engine_labelvalues
    ):
        return SuperInferPromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )
