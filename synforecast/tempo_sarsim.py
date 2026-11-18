"""Pool members and stage shares of the TempoPFN regime SDE and SarSim SARIMA-2 families.

``pretraining_pool(include_tempo_sarsim=True)`` appends
:data:`TEMPO_SARSIM_MEMBERS` after every other entry. A stream draws them at
explicit shares from :func:`member_weights` and gives the rest of its draws
to its existing members, so either family can be added or removed (share 0)
at matched exposure.
"""

from synforecast.simons_parity import STAGES

TEMPO_SARSIM_VERSION = "tempo-sarsim-v1"
"""Version tag of the two priors and the default shares below."""

TEMPO_SARSIM_MEMBERS = ("TempoSDEGenerator", "SarSimGenerator")
"""Generators appended by ``pretraining_pool(include_tempo_sarsim=True)``, in pool order."""

DEFAULT_SHARES = {
    "stage1": {"TempoSDEGenerator": 0.06, "SarSimGenerator": 0.06},
    "stage3": {"TempoSDEGenerator": 0.03, "SarSimGenerator": 0.03},
}
"""Proposed shares of synthetic draws (stage 2 uses stage 1's, stages 4 and 4-long stage 3's)."""


def member_weights(
    stage: str,
    tempo_sde_share: float | None = None,
    sarsim_share: float | None = None,
) -> dict[str, float]:
    """Share of synthetic draws of each appended generator at ``stage``, by class name.

    Args:
        stage (str): One of :data:`synforecast.simons_parity.STAGES`.
        tempo_sde_share (float | None): Share of TempoSDEGenerator; None takes :data:`DEFAULT_SHARES`.
        sarsim_share (float | None): Share of SarSimGenerator; None takes :data:`DEFAULT_SHARES`.

    Returns:
        Mapping of class name to share; the rest of the stream is the caller's.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    defaults = DEFAULT_SHARES["stage1" if stage in ("stage1", "stage2") else "stage3"]
    shares = {
        "TempoSDEGenerator": defaults["TempoSDEGenerator"]
        if tempo_sde_share is None
        else float(tempo_sde_share),
        "SarSimGenerator": defaults["SarSimGenerator"]
        if sarsim_share is None
        else float(sarsim_share),
    }
    if min(shares.values()) < 0 or sum(shares.values()) >= 1:
        raise ValueError(f"shares must be non-negative and sum below 1, got {shares}")
    return shares
