"""The twin's suit versions (Step 3 brief §5.4), as overrides of ``configs/twin.yaml``.

* ``suit_2026_11_14``: the new suit, the config's defaults: the binary wire, the page at ``87d7c3b`` and its recorder,
  the iOS app, 14 pods plus HipsAlt.
* ``suit_2026_11_13``: the old suit: the JSON pods and dongle (``twin.wire.mode: json``), the older page's recorder
  (1/30 s asserted, no ';' lines), ten pods plus HipsAlt (no hands or feet). The old phone is not documented in the
  repository; the twin uses the iOS app for it too (`assumption`). Nov 13's five takes are recordings of this suit.
* ``suit_2026_11_04``: Nov 4's suit, a third version the recording itself showed (S3-M6): the JSON wire and the older
  recorder as on Nov 13, but 13 pods plus HipsAlt (the hands and feet worn, the right hand not: `measured`, the pod
  bones of its BVH), and its own wire precision, :data:`NOV04_JSON_DECIMALS` (the still suit repeats 86 % of its
  frames, as only a 2-decimal wire does in the twin, while Nov 13's takes repeat 3-48 %, as only 4 or more do).
"""

from __future__ import annotations

__all__ = ["SUITS", "OLD_SUIT_PODS", "NOV04_PODS", "suit_overrides"]

OLD_SUIT_PODS = [0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 13]
"""Head, Spine, HipsAlt, both upper arms and forearms, both thighs and shins (Step 1 ids)."""
NOV04_PODS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
"""Nov 4: every pod but the right hand."""
NOV04_JSON_DECIMALS = 2
"""Nov 4's wire precision (`fitted`: S3-M6's calibration keeps 2 decimals in 8 of its 20 closest draws, and the other
precisions miss Nov 4's exact repeats by 55 points or more at every tilt constant; also `firmware`:
``Device code/Dongle__JSON.ino:59`` prints 2)."""

SUITS = {
    "suit_2026_11_14": {},
    "suit_2026_11_13": {"twin.wire.mode": "json", "twin.page.recorder": "old", "twin.suit.pods": OLD_SUIT_PODS},
    "suit_2026_11_04": {"twin.wire.mode": "json", "twin.page.recorder": "old", "twin.suit.pods": NOV04_PODS,
                        "twin.wire.json_decimals": NOV04_JSON_DECIMALS},
}


def suit_overrides(version: str) -> dict:
    """The overrides that make ``configs/twin.yaml`` describe suit ``version``."""
    if version not in SUITS:
        raise ValueError(f"unknown suit version {version!r}; known: {sorted(SUITS)}")
    return {"twin.suit.version": version, **SUITS[version]}
