# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestRotaryState, RequestStatus


def _make_request() -> Request:
    return Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
    )


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_rotary_state_syncs_with_status():
    req = _make_request()
    assert req.rotary_state == RequestRotaryState.WAITING

    req.status = RequestStatus.RUNNING
    assert req.rotary_state == RequestRotaryState.RUNNING

    req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req.rotary_state == RequestRotaryState.ROTARY_PENDING_IN

    req.status = RequestStatus.FINISHED_STOPPED
    assert req.rotary_state == RequestRotaryState.FINISHED


def test_request_rotary_state_transition_guard():
    req = _make_request()

    req.set_rotary_state(RequestRotaryState.ROTARY_PENDING_OUT)
    req.set_rotary_state(RequestRotaryState.ROTARY_SWAPPED)

    with pytest.raises(AssertionError):
        req.set_rotary_state(RequestRotaryState.RUNNING)

    req.set_rotary_state(RequestRotaryState.ROTARY_PENDING_IN)
    req.set_rotary_state(RequestRotaryState.WAITING)


def test_request_finished_rotary_state_guard():
    req = _make_request()
    req.status = RequestStatus.FINISHED_ABORTED

    with pytest.raises(AssertionError):
        req.set_rotary_state(RequestRotaryState.WAITING)
