r"""Erasure protection for priority packets via a simple repetition code.

On a memoryless packet-erasure channel, repeating a logical packet `r`
times lowers the probability that *all* copies are lost from `p` to
`p**r` (you only need one surviving replica).

This is a transparent stand-in for stronger codes (RS, LDPC, etc.) while
keeping the simulator honest: **every physical packet sees the same
drop probability** `p` — priority is expressed through **redundancy**,
not through biasing `p` by tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import numpy as np

from .priority_packetizer import Packet


@dataclass(frozen=True)
class PhysicalPacket:
    """One transmission on the wire (may be a replica of a logical packet)."""

    physical_index: int
    logical_index: int
    tier: int
    start: int
    end: int
    payload: np.ndarray


def expand_tier1_repetition(
    packets: Sequence[Packet], tier1_copies: int
) -> Tuple[PhysicalPacket, ...]:
    """Emit ``tier1_copies`` physical copies for each Tier-1 logical packet.

    Tier-2 logical packets are sent once.  ``tier1_copies == 1`` recovers
    the original one-physical-packet-per-logical-packet mapping.
    """
    if tier1_copies < 1:
        raise ValueError(f"tier1_copies must be >= 1, got {tier1_copies}")

    out: List[PhysicalPacket] = []
    phys = 0
    for pkt in packets:
        n_rep = int(tier1_copies) if pkt.tier == 1 else 1
        for _ in range(n_rep):
            out.append(
                PhysicalPacket(
                    physical_index=phys,
                    logical_index=int(pkt.index),
                    tier=int(pkt.tier),
                    start=int(pkt.start),
                    end=int(pkt.end),
                    payload=np.asarray(pkt.payload, dtype=np.int32).copy(),
                )
            )
            phys += 1
    return tuple(out)


def logical_received_from_physical(
    physical: Sequence[PhysicalPacket], received_physical: Set[int]
) -> List[int]:
    """Logical packet ``k`` is received if *any* replica of ``k`` arrived."""
    ok: Set[int] = set()
    for ph in physical:
        if ph.physical_index in received_physical:
            ok.add(ph.logical_index)
    return sorted(ok)
