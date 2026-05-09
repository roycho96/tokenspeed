# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""``update_status`` must keep ``Failed`` sticky once set.

When a request enters ``Failed`` state — whether from a heartbeat-detected
node failure or a deterministic bootstrap-info fetch failure in
``MooncakeKVReceiver.__init__`` — any subsequent non-``Failed`` call to
``update_status`` must not overwrite it.  The previous implementation used
``max(current, status)`` which silently flipped ``Failed (=0)`` to any
later non-``Failed`` value, letting the consumer read a never-filled KV
buffer and produce garbage tokens with no error surfaced.
"""

import collections
import os
import random
import sys
import threading
import types
import unittest
from unittest.mock import patch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.pd.base.conn import KVPoll  # noqa: E402
from tokenspeed.runtime.pd.mooncake.decode import (  # noqa: E402
    MooncakeKVManagerDecode,
    PrefillParallelInfo,
)
from tokenspeed.runtime.pd.mooncake.receiver import MooncakeKVReceiver  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 1024
_N_TOKENS = 8


def _make_manager() -> MooncakeKVManagerDecode:
    """Construct a manager without running its network ``__init__``.

    Only the state-machine attributes touched by ``update_status`` and
    ``record_failure`` are populated; ZMQ sockets and threads are not needed.
    """
    mgr = object.__new__(MooncakeKVManagerDecode)
    mgr.request_status = {}
    mgr.request_status_lock = threading.Lock()
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    return mgr


def _decode(kv_bytes: bytes, seed: int) -> list:
    """Deterministic mini-decode: same KV bytes + seed produce the same tokens."""
    kv_hash = hash(kv_bytes) & 0xFFFFFFFF
    rng = random.Random(seed ^ kv_hash)
    return [rng.randint(0, _VOCAB_SIZE - 1) for _ in range(_N_TOKENS)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpdateStatusFailedSticky(unittest.TestCase):
    SEED = 12345
    ROOM = 42
    KV_SIZE = 1024

    def test_normal_transfer_reaches_success(self):
        """Clean prefill-to-Success path sets status to Success."""
        mgr = _make_manager()
        for s in (
            KVPoll.Bootstrapping,
            KVPoll.Bootstrapped,
            KVPoll.WaitingForInput,
            KVPoll.Transferring,
        ):
            mgr.update_status(self.ROOM, s)
        mgr.update_status(self.ROOM, KVPoll.Success)
        self.assertEqual(mgr.request_status[self.ROOM], KVPoll.Success)

    def test_failed_status_is_sticky_against_later_success(self):
        """Failed must not be overwritten by a later non-Failed update."""
        mgr = _make_manager()
        for s in (KVPoll.Bootstrapping, KVPoll.Bootstrapped, KVPoll.Transferring):
            mgr.update_status(self.ROOM, s)
        mgr.record_failure(self.ROOM, "simulated heartbeat node failure")
        mgr.update_status(self.ROOM, KVPoll.Failed)
        mgr.update_status(self.ROOM, KVPoll.Success)

        self.assertEqual(
            mgr.request_status[self.ROOM],
            KVPoll.Failed,
            "Failed must be sticky; max(Failed=0, Success=5) silently flips it.",
        )

    def test_failed_kv_transfer_does_not_corrupt_decoded_output(self):
        """A consumer that gates decode on Success must abort when the transfer failed."""
        mgr = _make_manager()
        for s in (KVPoll.Bootstrapping, KVPoll.Bootstrapped, KVPoll.Transferring):
            mgr.update_status(self.ROOM, s)
        kv_buffer = b"\x00" * self.KV_SIZE
        mgr.record_failure(self.ROOM, "simulated heartbeat node failure")
        mgr.update_status(self.ROOM, KVPoll.Failed)
        mgr.update_status(self.ROOM, KVPoll.Success)

        if mgr.request_status[self.ROOM] == KVPoll.Success:
            decoded = _decode(kv_buffer, self.SEED)
        else:
            decoded = None

        self.assertIsNone(
            decoded,
            "A Failed transfer must not be silently advertised as Success — "
            "the consumer would otherwise decode an unfilled KV buffer. "
            f"failure_records: {mgr.failure_records.get(self.ROOM)!r}",
        )


def _make_manager_for_receiver(
    bootstrap_addr: str = "127.0.0.1:8080",
) -> MooncakeKVManagerDecode:
    """Extend ``_make_manager`` with attributes touched by ``MooncakeKVReceiver.__init__``.

    Adds the extra dict/attr state that the receiver constructor reads before
    (and between) the two early-return sites, without opening any sockets or
    threads.
    """
    mgr = _make_manager()
    # Attributes read by _calc and the bootstrap-key lookup.
    mgr.world_size = 1
    mgr.dp_size = 1
    mgr.is_mla_backend = False
    mgr.draft_is_mla_backend = False
    # kv_args: only engine_rank is needed by _calc and the failure message.
    mgr.kv_args = types.SimpleNamespace(engine_rank=0)
    # Dicts consulted between the two return sites.
    mgr.required_prefill_response_num_table = {}
    mgr.connection_pool = {}
    mgr.prefill_parallel_info = {}
    # defaultdict(set) matches production so a regression that drops the early
    # return surfaces as the actual Failed->Bootstrapped flip rather than a
    # masking KeyError on the tracker access.
    mgr.addr_to_rooms_tracker = collections.defaultdict(set)
    # get_session_id() is called unconditionally before either return site.
    mgr.get_session_id = lambda: "test-session-0"
    return mgr


class TestReceiverInitFailureIsSticky(unittest.TestCase):
    """``__init__`` failure branches must leave the room in Failed and abort early."""

    ROOM = 99
    ADDR = "127.0.0.1:9999"

    def test_failed_status_sticky_when_prefill_parallel_info_is_none(self):
        """A missing prefill parallel info aborts the constructor with Failed latched."""
        mgr = _make_manager_for_receiver(self.ADDR)
        with patch.object(
            MooncakeKVReceiver, "_get_prefill_parallel_info", return_value=None
        ):
            receiver = MooncakeKVReceiver(mgr, self.ADDR, self.ROOM)

        self.assertEqual(mgr.check_status(self.ROOM), KVPoll.Failed)
        self.assertTrue(mgr.failure_records.get(self.ROOM))
        self.assertEqual(receiver.poll(), KVPoll.Failed)

    def test_failed_status_sticky_when_bootstrap_infos_is_none(self):
        """A missing bootstrap-info fetch aborts before the trailing Bootstrapped update."""
        mgr = _make_manager_for_receiver(self.ADDR)
        stub_ppi = PrefillParallelInfo(
            tp_size=1, dp_size=1, enable_mla_l1_5_cache=False
        )
        with patch.object(
            MooncakeKVReceiver,
            "_get_prefill_parallel_info",
            return_value=stub_ppi,
        ), patch.object(MooncakeKVReceiver, "_get_bootstrap_infos", return_value=None):
            receiver = MooncakeKVReceiver(mgr, self.ADDR, self.ROOM)

        self.assertEqual(mgr.check_status(self.ROOM), KVPoll.Failed)
        self.assertTrue(mgr.failure_records.get(self.ROOM))
        self.assertEqual(receiver.poll(), KVPoll.Failed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
