from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence

import numpy as np

from .priority_packetizer import Packet, PacketTier


LossModel = Literal["random", "gilbert_elliott"]


@dataclass(frozen=True)
class BurstConfig:
    """Gilbert–Elliott two-state burst loss model.

    States:
      - G (good): packet loss probability p_loss_good
      - B (bad):  packet loss probability p_loss_bad

    Transitions per packet:
      - G -> B with probability p_gb
      - B -> G with probability p_bg
    """

    p_gb: float = 0.02
    p_bg: float = 0.20
    p_loss_good: float = 0.02
    p_loss_bad: float = 0.50

    def __post_init__(self) -> None:
        for name in ("p_gb", "p_bg", "p_loss_good", "p_loss_bad"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")


def simulate_packet_loss(
    packets: Sequence[Packet],
    *,
    model: LossModel = "random",
    p_loss: float = 0.10,
    burst: BurstConfig | None = None,
    rng: np.random.Generator | None = None,
) -> List[int]:
    """Return indices of packets that are successfully received."""
    if rng is None:
        rng = np.random.default_rng()
    if not (0.0 <= p_loss <= 1.0):
        raise ValueError(f"p_loss must be in [0, 1], got {p_loss}")

    if model == "random":
        received: List[int] = []
        for pkt in packets:
            if rng.random() >= p_loss:
                received.append(pkt.index)
        return received

    if model == "gilbert_elliott":
        if burst is None:
            burst = BurstConfig()
        received = []
        state_bad = False  # start in Good
        for pkt in packets:
            # Transition
            if not state_bad:
                if rng.random() < burst.p_gb:
                    state_bad = True
            else:
                if rng.random() < burst.p_bg:
                    state_bad = False

            p = burst.p_loss_bad if state_bad else burst.p_loss_good
            if rng.random() >= p:
                received.append(pkt.index)
        return received

    raise ValueError(f"unknown model {model!r}")


def tier_stats(packets: Sequence[Packet], received_indices: Iterable[int]) -> dict:
    """Compute received/lost counts split by tier."""
    rec = set(int(i) for i in received_indices)
    out = {
        "tier1_total": 0,
        "tier1_received": 0,
        "tier2_total": 0,
        "tier2_received": 0,
    }
    for pkt in packets:
        if pkt.tier == 1:
            out["tier1_total"] += 1
            out["tier1_received"] += int(pkt.index in rec)
        else:
            out["tier2_total"] += 1
            out["tier2_received"] += int(pkt.index in rec)
    out["tier1_lost"] = out["tier1_total"] - out["tier1_received"]
    out["tier2_lost"] = out["tier2_total"] - out["tier2_received"]
    return out

