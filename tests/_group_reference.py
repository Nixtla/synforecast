"""Priors and hashing for the overlays=None identity fixture (tests/fixtures/group_draws_3b33546e.json).

Uses only API that existed at 3b33546e, so the fixture can be regenerated there:
``python -c "import tests._group_reference as r; r.write('tests/fixtures/group_draws_3b33546e.json')"``.
"""

import hashlib
import json

import numpy as np

from synforecast import (
    CoupledSpec,
    FixedRegressionSpec,
    GroupPrior,
    RegressionSpec,
    VARXSpec,
    probe_suite,
)


def reference_priors() -> dict[str, GroupPrior]:
    """Every mechanism, several lengths, frequencies and composition modes."""
    probes = probe_suite(context=64, horizon=32)
    priors = {
        "regression": GroupPrior(
            mechanism=RegressionSpec(), length=(32, 300), freq="h"
        ),
        "regression/positive": GroupPrior(
            mechanism=RegressionSpec(p_multiplicative=1.0, p_interaction=1.0),
            length=(200, 200),
            freq="D",
        ),
        "coupled/one": GroupPrior(
            mechanism=CoupledSpec(targets="one", n_channels=(2, 6)),
            length=(64, 512),
            freq="h",
        ),
        "coupled/all": GroupPrior(
            mechanism=CoupledSpec(), length=(128, 128), freq="MS"
        ),
        "varx": GroupPrior(
            mechanism=VARXSpec(n_targets=(2, 3), n_future=1, n_past=1, n_shared=1),
            length=(100, 400),
        ),
        "varx/int": GroupPrior(mechanism=VARXSpec(), length=(64, 64), freq=1),
    }
    for name in ("sum/phi0.3/past/d2/smooth", "nonlinear/phi0.0/future/d0/iid"):
        priors[f"probe/{name}"] = probes[name]
    assert all(
        isinstance(p.mechanism, FixedRegressionSpec)
        for k, p in priors.items()
        if k.startswith("probe")
    )
    return priors


def digest(prior: GroupPrior) -> dict:
    """Hashes of 8 groups' values, roles and recipes, and the prior fingerprint."""
    batch = prior.sample(8, seed=5, start_group=3, n_jobs=2)
    values = hashlib.sha256(
        b"".join(np.ascontiguousarray(g.values).tobytes() for g in batch)
    )
    recipes = hashlib.sha256(
        json.dumps([g.recipe for g in batch], sort_keys=True).encode()
    )
    return {
        "values": values.hexdigest(),
        "roles": [list(g.roles) for g in batch],
        "recipes": recipes.hexdigest(),
        "fingerprint": prior.fingerprint,
    }


def write(path: str) -> None:
    out = {name: digest(prior) for name, prior in reference_priors().items()}
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
