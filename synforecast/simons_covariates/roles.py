"""Covariate roles on coupled synthetic groups and synthetic covariate-effect rows.

Port of ``simons/training/covariates/roles.py`` (``settings``, ``RolePlanner``,
``effect_row``) without torch: rows are numpy arrays. The numpy random stream
is consumed exactly as the reference does, so for the same generator state and
the same base windows the outputs are identical.
"""

import math

import numpy as np

from synforecast.simons_covariates import families, generic

WIDTH = 32
PATCH = 32
EFFECT_ID = 50_000_000
MECHANISMS = ("identity", "mix", "lag", "coint", "func", "event")
MECH = {name: i for i, name in enumerate(MECHANISMS)}

DEFAULTS = {
    "boundaries": [0.27, 0.53],
    "p_role": [0.10, 0.20, 0.40],
    "p_eff": [0.0, 0.0, 0.0],
    "root_p": 0.8,
    "sibling_p": [0.2, 0.5],
    "max_cov": 8,
    "partial_future": 0.15,
    "end_block": 0.5,
    "end_block_max": 16,
    "effect_workers": 3,
    "effect_horizons": [1, 2, 3, 4, 6, 8, 12, 16],
    "effect_pad": 0.4,
    "effect_min_observed": 1024,
    "effect_flip": 0.3,
    "base_families": "kernel,sarima,sde,step,spike,cauker,rhythm,wave,fpfn,ets,levy,pulse",
    "base_weights": "0.21056,0.21056,0.12024,0.05512,0.05512,0.08016,0.04008,0.03008,0.03008,0.068,0.015,0.015",
    "synth_calendar": 0.3,
    "effect_mix": None,
}


def settings(spec):
    """The role settings: DEFAULTS updated by ``spec`` (validated)."""
    s = {**DEFAULTS, **(spec or {})}
    unknown = set(spec or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"covariate roles: unknown fields {sorted(unknown)}")
    if len(s["p_role"]) != len(s["boundaries"]) + 1 or len(s["p_eff"]) != len(
        s["p_role"]
    ):
        raise ValueError("role_share and effect_share need one value per role phase")
    if max(s["p_eff"]) > 0 and s["effect_mix"] is None:
        raise ValueError("effect rows (effect_share > 0) need effect_mix")
    if s["effect_mix"] is not None:
        families.parse_mix(s["effect_mix"])
    return s


def phase_value(values, boundaries, fraction):
    """The value of the phase that ``fraction`` of the run falls in."""
    return values[
        int(np.searchsorted(np.asarray(boundaries, float), fraction, side="right"))
    ]


def structure_from_recipes(group, recipes):
    """Simons ``plan`` fields (group, mech, root) for coupled slots from Multivariatizer recipes.

    ``group`` [n] holds each slot's group id (slots of a group contiguous);
    ``recipes`` maps a group id to its Multivariatizer recipe (``_last_recipe``)
    or None (GroupPrior ``CoupledSpec`` recipes use the same ``leadlag`` pairs).
    A group with lead-lag pairs is a ``lag`` group whose roots are its leaders:
    channels some pair reads from and no pair writes to (Simons' observed
    drivers that followers lag); any other group of two or more channels is
    ``mix`` (no roots); a single channel is ``identity``.
    """
    group = np.asarray(group, np.int64)
    n = len(group)
    mech = np.zeros(n, np.int64)
    root = np.zeros(n, bool)
    if n == 0:
        return {"group": group, "mech": mech, "root": root}
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts.tolist(), ends.tolist(), strict=False):
        q = b - a
        if q < 2:
            continue
        recipe = recipes.get(int(group[a])) or {}
        pairs = recipe.get("leadlag") or []
        if pairs:
            followers = {int(p["dst"]) for p in pairs}
            leaders = {int(p["src"]) for p in pairs} - followers
            mech[a:b] = MECH["lag"]
            root[a:b] = [i in leaders for i in range(q)]
        else:
            mech[a:b] = MECH["mix"]
    return {"group": group, "mech": mech, "root": root}


def role_meta_from_roles(channel_roles, length, origin):
    """Simons role metadata [k, 3] (role, end block, cut) for one group's channel roles (GroupPrior ``Group.roles``).

    'target' slots get no role; 'future' covariates are known to the end;
    'past' covariates are known up to ``origin`` (a future cut at the origin).
    """
    meta = np.zeros((len(channel_roles), 3), np.int64)
    if not 0 < int(origin) <= int(length):
        raise ValueError("origin must be in (0, length]")
    for i, role in enumerate(channel_roles):
        if role == "target":
            continue
        if role not in ("future", "past"):
            raise ValueError(f"unknown channel role {role!r}")
        meta[i, 0] = 1
        if role == "past":
            meta[i, 2] = int(origin)
    return meta


class RolePlanner:
    """Assign covariate roles and future visibility to coupled synthetic groups."""

    def __init__(self, s, seed: int, stage_steps: int):
        self.s = s
        self.rng = np.random.default_rng([int(seed), 4417])
        self.stage_steps = max(1, int(stage_steps))
        self.step = 0
        self.stats = {"pools": 0, "groups": 0, "role_groups": 0, "cov_slots": 0, "target_slots": 0, "end_blocks": 0,
                      "cuts": 0, "by_mech": {}}  # fmt: skip

    def fraction(self):
        return self.step / self.stage_steps

    def p_role(self):
        return float(
            phase_value(self.s["p_role"], self.s["boundaries"], self.fraction())
        )

    def plan_pool(self, structure, kinds, length, n_pool):
        """[n_pool, 3] int64: role, target end-block patches, future cut step (0: none); ``kinds`` marks coupled rows."""
        s, rng = self.s, self.rng
        group, mech, root = structure["group"], structure["mech"], structure["root"]
        n_c = len(group)
        meta_c = np.zeros((n_c, 3), np.int64)
        p = self.p_role()
        self.stats["pools"] += 1
        if n_c:
            starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
            ends = np.r_[starts[1:], n_c]
        else:
            starts = ends = np.zeros(0, np.int64)
        for a, b in zip(starts.tolist(), ends.tolist(), strict=False):
            q = b - a
            if q < 2:
                continue
            self.stats["groups"] += 1
            if rng.random() >= p:
                continue
            m = int(mech[a])
            r = root[a:b]
            if m in (MECH["lag"], MECH["event"]) or (m == MECH["func"] and r.any()):
                cov = r & (rng.random(q) < s["root_p"])
            else:
                share = rng.uniform(*s["sibling_p"])
                cov = rng.random(q) < share
            if cov.all():
                cov[int(rng.integers(q))] = False
            if cov.sum() > s["max_cov"]:
                on = np.flatnonzero(cov)
                cov[:] = False
                cov[rng.choice(on, s["max_cov"], replace=False)] = True
            if not cov.any():
                continue
            block = meta_c[a:b]
            block[cov, 0] = 1
            if rng.random() < s["end_block"]:
                block[~cov, 1] = int(rng.integers(1, s["end_block_max"] + 1))
                self.stats["end_blocks"] += 1
            cut = cov & (rng.random(q) < s["partial_future"])
            if cut.any():
                block[cut, 2] = rng.integers(
                    int(0.75 * length), length, size=int(cut.sum())
                )
                self.stats["cuts"] += int(cut.sum())
            self.stats["role_groups"] += 1
            self.stats["cov_slots"] += int(cov.sum())
            self.stats["target_slots"] += int((~cov).sum())
            key = MECHANISMS[m]
            self.stats["by_mech"][key] = self.stats["by_mech"].get(key, 0) + 1
        meta = np.zeros((n_pool, 3), np.int64)
        c = 0
        for k, is_coupled in enumerate(kinds):
            if is_coupled:
                meta[k * WIDTH : (k + 1) * WIDTH] = meta_c[c * WIDTH : (c + 1) * WIDTH]
                c += 1
        return meta


def effect_row(s, rng, window, length):
    """One 32-slot effect row: values [32, L] float32, meta [32, 3], group [32], first [32], family counts.

    ``window(n)`` returns a float64 base window of ``n`` steps (NaN = missing);
    metadata holds (role, end block, cut); group -1 marks empty slots; first is
    the first observed patch.
    """
    values = np.full((WIDTH, length), np.nan, np.float32)
    meta = np.zeros((WIDTH, 3), np.int64)
    group = np.full(WIDTH, -1, np.int64)
    slot, g, tries = 0, 0, 0
    names, p = families.parse_mix(s["effect_mix"])
    counts = np.zeros(len(families.FAMILIES), np.int64)
    while slot < WIDTH - 1 and tries < 64:
        tries += 1
        family = names[int(rng.choice(len(names), p=p))]
        k = (min(generic.MAX_COV, int(rng.geometric(0.35)), WIDTH - slot - 1)
             if family == "generic" else min(generic.MAX_COV, WIDTH - slot - 1))  # fmt: skip
        H = families.horizon(family, rng, s["effect_horizons"]) * PATCH
        observed = length
        if rng.random() < s["effect_pad"]:
            lo = max(s["effect_min_observed"], H + 256)
            observed = int(math.exp(rng.uniform(math.log(lo), math.log(length))))
            observed = min(length, max(lo, observed // PATCH * PATCH))
        hist = observed - H
        b = window(observed)
        if np.isfinite(b[:hist]).sum() < 64 or np.isfinite(b[hist:]).sum() < H // 4:
            continue
        y, c = families.make_task(family, b, rng, hist, H, k, window)
        if len(c) == 0 or y.shape[-1] != observed or c.shape[-1] != observed:
            continue
        k = len(c)
        if not np.isfinite(y[:hist]).any():
            continue
        counts[families.FAMILIES.index(family)] += 1
        if rng.random() < s["effect_flip"]:
            y = -y
        start = length - observed
        values[slot, start:] = y
        values[slot + 1 : slot + 1 + k, start:] = c
        meta[slot, 1] = H // PATCH
        meta[slot + 1 : slot + 1 + k, 0] = 1
        for i in range(k):
            if rng.random() < s["partial_future"]:
                meta[slot + 1 + i, 2] = start + hist + int(rng.integers(1, H))
        group[slot : slot + 1 + k] = g
        g += 1
        slot += 1 + k
    seen = np.isfinite(values).reshape(WIDTH, -1, PATCH).any(-1)
    patches = seen.shape[1]
    first = np.where(seen.any(1), seen.argmax(1), patches).astype(np.int64)
    return {
        "values": values,
        "meta": meta,
        "group": group,
        "first": first,
        "families": counts,
    }
