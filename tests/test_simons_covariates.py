"""Tests for the Simons covariate mechanisms and curriculum (synforecast.simons_covariates)."""

import hashlib
import json
import platform
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

from synforecast import CoupledSpec, GroupPrior, VARXSpec, pretraining_pool  # noqa: E402
from synforecast import simons_covariates as sc  # noqa: E402
from synforecast.simons_covariates import families, generic, roles  # noqa: E402

STAGE2_MIX = {
    "generic": 0.2,
    "classical": 0.3,
    "energy": 0.2,
    "retail": 0.15,
    "lagged": 0.15,
}
LENGTH = 2048


def _base(seed, n=4 * LENGTH, nan_prefix=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.cumsum(rng.normal(size=n)) * 0.1 + np.sin(2 * np.pi * t / 24) + 5
    x[:nan_prefix] = np.nan
    return x


def _window_fn(seed):
    rng = np.random.default_rng(seed)

    def window(n):
        x = _base(int(rng.integers(1 << 30)), n=max(n, 64))
        return x[len(x) - n :]

    return window


def _stage2_settings():
    return roles.settings(
        sc.CovariateCurriculum.from_simons_stage("stage2").role_spec()
    )


def _digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# Hashes of existing draws: main (8fbcac1) pool members and PR #107 GroupPrior mechanisms; the subpackage must not move them.
GOLDEN_MAIN = {
    "pool[0]:SARIMAGenerator": "930d1ccf8b4f9385a5cf4386e33585345cc5b251281cb4dde41953e4cd34bbd6",
    "pool[20]:CyclicGenerator": "4c187031cdca3f1ffd8e06d7981e857cc2ab077af0acc764a7d24ef7fbb0f32f",
    "pool[3]:RegimeSwitchingGenerator": "7c99b0dded011522163fb4372c4b99821bf25ac964cd591f47eccab7883d91bc",
    "pool[64]:StochasticVolatilityGenerator": "f994f34e463fd4891afed85dad0bb77b9fd8df0843289c55cf30782a07beb41e",
    "pool[7]:EnergyLoadGenerator": "e81e7ea59028d43c8de59b7d02223c91f4af62331c85835dcb556d73c53ba331",
}
GOLDEN_PR107 = {
    "group:coupled_one": "52062ac7ac1ba2df7cf35344ce4ac71f6b2cf82eeb90d34060a4e697d903d428|f3102d546d6ad7db",
    "group:regression": "d92ffff091e0111aefb9e138a5ce3be79ea0816d3a32b2ed350224c81bcb4837|36ba04be347f4d40",
    "group:varx": "9a34df2f673b0114090fa24c11df818a415065abb98648dd93909a962b013f91|7391fd0835ad0915",
}


class TestGolden:
    @pytest.mark.skipif(
        sys.platform != "linux"
        or platform.machine() != "x86_64"
        or sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 13),
        reason="the fixture hashes draws on linux x86_64 with Python 3.13",
    )
    def test_main_draws_byte_identical(self) -> None:
        import synforecast.simons_covariates  # noqa: F401

        out = {}
        pool = pretraining_pool(
            min_length=256, max_length=256, freq="D", seed=11, include_extended=True
        )
        for i in (0, 3, 7, 20, len(pool) - 1):
            out[f"pool[{i}]:{type(pool[i]).__name__}"] = _digest(
                pool[i].generate_values(3, n_jobs=1)
            )
        assert out == GOLDEN_MAIN

    @pytest.mark.skipif(
        sys.platform != "linux"
        or platform.machine() != "x86_64"
        or sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 13),
        reason="the fixture hashes draws on linux x86_64 with Python 3.13",
    )
    def test_group_prior_draws_byte_identical(self) -> None:
        import synforecast.simons_covariates  # noqa: F401

        priors = {
            "regression": GroupPrior(length=(128, 128), freq="D", burn_in=96),
            "coupled_one": GroupPrior(
                mechanism=CoupledSpec(targets="one"),
                length=(128, 128),
                freq="D",
                burn_in=160,
            ),
            "varx": GroupPrior(
                mechanism=VARXSpec(), length=(128, 128), freq="D", burn_in=96
            ),
        }
        out = {}
        for name, prior in priors.items():
            batch = prior.sample(4, seed=7, n_jobs=1)
            roles_digest = hashlib.sha256(
                json.dumps([list(g.roles) for g in batch]).encode()
            )
            out[f"group:{name}"] = (
                _digest([g.values for g in batch]) + "|" + roles_digest.hexdigest()[:16]
            )
        assert out == GOLDEN_PR107

    def test_group_roles_to_meta(self) -> None:
        prior = GroupPrior(
            mechanism=CoupledSpec(targets="one"),
            length=(128, 128),
            freq="D",
            burn_in=160,
        )
        for group in prior.sample(4, seed=3, n_jobs=1):
            meta = roles.role_meta_from_roles(group.roles, 128, 96)
            target = np.asarray(group.roles) == "target"
            assert (meta[target] == 0).all() and (meta[~target, 0] == 1).all()
            assert (meta[np.asarray(group.roles) == "past", 2] == 96).all()

    def test_package_does_not_import_subpackage(self) -> None:
        import synforecast

        assert "simons_covariates" not in getattr(synforecast, "__all__", [])


class TestCurriculum:
    def test_stage2(self) -> None:
        c = sc.CovariateCurriculum.from_simons_stage("stage2")
        assert c.boundaries == [0.27, 0.53]
        assert c.role_share == [0.1, 0.2, 0.4]
        assert c.effect_share == [0.01, 0.02, 0.04]
        assert c.effect_mix == STAGE2_MIX
        assert c.pass_weight == 0.0 and c.long_weight == 0.0
        assert c.enabled

    @pytest.mark.parametrize(
        ("stage", "long"), [("stage3", 0.0), ("stage4", 0.025), ("stage4-long", 0.025)]
    )
    def test_late_stages(self, stage, long) -> None:
        c = sc.CovariateCurriculum.from_simons_stage(stage)
        assert c.role_share == [0.4] and c.effect_share == [0.0]
        assert c.pass_weight == 0.025
        assert c.pass_mix == {"synthetic": 0.63, "sibling": 0.25, "real": 0.12}
        assert c.long_weight == long
        assert c.role_spec() == {
            "boundaries": [],
            "p_role": [0.4],
            "p_eff": [0.0],
            "effect_mix": None,
            "effect_workers": 4,
        }

    def test_unknown_stage(self) -> None:
        with pytest.raises(ValueError):
            sc.CovariateCurriculum.from_simons_stage("stage1")

    def test_off(self) -> None:
        c = sc.CovariateCurriculum.off()
        assert not c.enabled
        assert c.role_spec() is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"role_share": [1.5]},
            {"boundaries": [0.5, 0.2], "role_share": [0.1, 0.2, 0.3]},
            {"boundaries": [0.5], "role_share": [0.1, 0.2, 0.3]},
            {"effect_share": [0.1]},
            {"effect_share": [0.1], "effect_mix": {"nope": 1.0}},
            {"pass_weight": 0.1},
            {"pass_mix": {"synthetic": 1.0}},
            {"pass_mix": {"synthetic": -1.0, "sibling": 1.0, "real": 1.0}},
        ],
    )
    def test_validation(self, kwargs) -> None:
        with pytest.raises(ValueError):
            sc.CovariateCurriculum(**kwargs)

    def test_phase_boundaries(self) -> None:
        c = sc.CovariateCurriculum.from_simons_stage("stage2")
        cases = [
            (0.0, 0.1, 0.01),
            (0.2699, 0.1, 0.01),
            (0.27, 0.2, 0.02),
            (0.5299, 0.2, 0.02),
            (0.53, 0.4, 0.04),
            (1.0, 0.4, 0.04),
        ]
        for fraction, pr, pe in cases:
            assert c.p_role(fraction) == pr
            assert c.p_effect(fraction) == pe
        spec = c.role_spec(effect_workers=4)
        assert (
            spec["p_role"] == [0.1, 0.2, 0.4]
            and spec["p_eff"] == [0.01, 0.02, 0.04]
            and spec["effect_workers"] == 4
        )
        assert roles.settings(spec)["effect_mix"] == STAGE2_MIX

    def test_single_value_broadcasts(self) -> None:
        c = sc.CovariateCurriculum(
            boundaries=[0.5], role_share=[0.3], effect_share=[0.0]
        )
        assert c.role_spec()["p_role"] == [0.3, 0.3]

    def test_effect_rows_rounding(self) -> None:
        c = sc.CovariateCurriculum.from_simons_stage("stage2")
        rng = np.random.default_rng(0)
        draws = np.array([c.effect_rows(100, 0.9, rng) for _ in range(4000)])
        assert set(np.unique(draws)) == {4}
        draws = np.array([c.effect_rows(30, 0.9, rng) for _ in range(20000)])
        assert set(np.unique(draws)) <= {1, 2}
        assert abs(draws.mean() - 1.2) < 0.02
        assert c.effect_rows(0, 0.9, rng) == 0

    def test_effect_rows_capped(self) -> None:
        c = sc.CovariateCurriculum(effect_share=[1.0], effect_mix=STAGE2_MIX)
        assert c.effect_rows(5, 0.0, np.random.default_rng(0)) == 5


class TestSettings:
    def test_unknown_field(self) -> None:
        with pytest.raises(ValueError):
            roles.settings({"nope": 1})

    def test_phase_count(self) -> None:
        with pytest.raises(ValueError):
            roles.settings(
                {"boundaries": [0.5], "p_role": [0.1, 0.2, 0.3], "p_eff": [0, 0, 0]}
            )

    def test_effects_need_mix(self) -> None:
        with pytest.raises(ValueError):
            roles.settings({"p_eff": [0.1, 0.1, 0.1]})


class TestEffectRow:
    @pytest.mark.parametrize("seed", range(6))
    def test_invariants(self, seed) -> None:
        s = _stage2_settings()
        row = roles.effect_row(s, np.random.default_rng(seed), _window_fn(seed), LENGTH)
        values, meta, group, first = (
            row["values"],
            row["meta"],
            row["group"],
            row["first"],
        )
        assert values.shape == (32, LENGTH) and values.dtype == np.float32
        assert meta.shape == (32, 3) and group.shape == (32,) and first.shape == (32,)
        assert row["families"].sum() == group.max() + 1
        empty = group < 0
        assert np.isnan(values[empty]).all() and (meta[empty] == 0).all()
        used = np.flatnonzero(~empty)
        assert (used == np.arange(len(used))).all()
        for g in range(group.max() + 1):
            slots = np.flatnonzero(group == g)
            target, covs = slots[0], slots[1:]
            assert len(covs) >= 1
            assert meta[target, 0] == 0 and meta[target, 1] > 0 and meta[target, 2] == 0
            assert (meta[covs, 0] == 1).all() and (meta[covs, 1] == 0).all()
            horizon = int(meta[target, 1]) * roles.PATCH
            observed = np.isfinite(values[slots]).any(0)
            start = int(np.argmax(observed)) if observed.any() else LENGTH
            hist_end = LENGTH - horizon
            assert start < hist_end
            for c in covs:
                cut = int(meta[c, 2])
                if cut:
                    assert hist_end < cut < LENGTH
        seen = np.isfinite(values).reshape(32, -1, roles.PATCH).any(-1)
        expect = np.where(seen.any(1), seen.argmax(1), seen.shape[1])
        assert (first == expect).all()

    def test_deterministic(self) -> None:
        s = _stage2_settings()
        a = roles.effect_row(s, np.random.default_rng(3), _window_fn(9), LENGTH)
        b = roles.effect_row(s, np.random.default_rng(3), _window_fn(9), LENGTH)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_stream_determinism(self) -> None:
        spec = sc.CovariateCurriculum.from_simons_stage("stage2").role_spec()
        a = sc.EffectRowStream(spec, np.random.default_rng(5), _window_fn(1), LENGTH)
        b = sc.EffectRowStream(spec, np.random.default_rng(5), _window_fn(1), LENGTH)
        c = sc.EffectRowStream(spec, np.random.default_rng(6), _window_fn(1), LENGTH)
        ra, rb, rc = (
            [next(a) for _ in range(3)],
            [next(b) for _ in range(3)],
            [next(c) for _ in range(3)],
        )
        for x, y in zip(ra, rb, strict=True):
            np.testing.assert_array_equal(x["values"], y["values"])
            np.testing.assert_array_equal(x["meta"], y["meta"])
        assert not all(
            np.array_equal(x["values"], y["values"], equal_nan=True)
            for x, y in zip(ra, rc, strict=True)
        )

    def test_stream_needs_mix(self) -> None:
        with pytest.raises(ValueError):
            sc.EffectRowStream(
                {"p_role": [0.1, 0.1, 0.1]},
                np.random.default_rng(0),
                _window_fn(0),
                LENGTH,
            )

    def test_single_family_mix(self) -> None:
        s = roles.settings({"p_eff": [0.1, 0.1, 0.1], "effect_mix": {"retail": 1.0}})
        row = roles.effect_row(s, np.random.default_rng(1), _window_fn(2), LENGTH)
        counts = dict(zip(families.FAMILIES, row["families"].tolist(), strict=True))
        assert (
            counts["retail"] > 0
            and sum(v for k, v in counts.items() if k != "retail") == 0
        )


def _plan(structure_group, mech, root):
    return {
        "group": np.asarray(structure_group),
        "mech": np.asarray(mech),
        "root": np.asarray(root, bool),
    }


class TestStructure:
    def test_leadlag_roots(self) -> None:
        group = np.r_[np.zeros(4, int), np.ones(3, int), [2]]
        recipes = {0: {"couplings": ["mixing", "leadlag"], "leadlag": [{"src": 0, "dst": 2}, {"src": 1, "dst": 3}]},
                   1: {"couplings": ["mixing"], "mixing": {}}, 2: None}  # fmt: skip
        s = sc.structure_from_recipes(group, recipes)
        assert s["mech"].tolist() == [2, 2, 2, 2, 1, 1, 1, 0]
        assert s["root"].tolist() == [
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        chain = sc.structure_from_recipes(
            np.zeros(4, int),
            {0: {"leadlag": [{"src": 0, "dst": 1}, {"src": 1, "dst": 2}]}},
        )
        assert chain["root"].tolist() == [True, False, False, False]
        assert (s["group"] == group).all()

    def test_role_meta_from_roles(self) -> None:
        meta = roles.role_meta_from_roles(("target", "future", "past"), 512, 448)
        assert meta.tolist() == [[0, 0, 0], [1, 0, 0], [1, 0, 448]]
        with pytest.raises(ValueError):
            roles.role_meta_from_roles(("target", "other"), 512, 448)
        with pytest.raises(ValueError):
            roles.role_meta_from_roles(("target",), 512, 0)

    def test_missing_recipe_is_mix(self) -> None:
        s = sc.structure_from_recipes(np.zeros(3, int), {})
        assert s["mech"].tolist() == [1, 1, 1] and not s["root"].any()

    def test_empty(self) -> None:
        s = sc.structure_from_recipes(np.zeros(0, int), {})
        assert len(s["mech"]) == 0


class TestRolePlanner:
    def _structure(self, rng, mech_name):
        sizes = []
        while sum(sizes) < 32:
            sizes.append(min(int(rng.integers(2, 17)), 32 - sum(sizes)))
        group = np.repeat(np.arange(len(sizes)), sizes)
        mech = np.full(32, roles.MECH[mech_name])
        root = np.zeros(32, bool)
        for g, q in enumerate(sizes):
            a = int(np.flatnonzero(group == g)[0])
            root[a : a + min(2, q)] = True
        return _plan(group, mech, root)

    @pytest.mark.parametrize("mech_name", ["mix", "lag", "identity", "coint", "event"])
    def test_invariants(self, mech_name) -> None:
        s = roles.settings({"p_role": [1.0, 1.0, 1.0], "max_cov": 3})
        planner = roles.RolePlanner(s, seed=1, stage_steps=100)
        rng = np.random.default_rng(0)
        roled = 0
        for _ in range(40):
            st = self._structure(rng, mech_name)
            kinds = np.array([True, False, True])
            full = {"group": np.r_[st["group"], st["group"] + 100], "mech": np.r_[st["mech"], st["mech"]],
                        "root": np.r_[st["root"], st["root"]]}  # fmt: skip
            meta = planner.plan_pool(full, kinds, 512, 96)
            assert meta.shape == (96, 3)
            assert (meta[32:64] == 0).all()
            for rows in (meta[:32], meta[64:]):
                for g in np.unique(st["group"]):
                    m = rows[st["group"] == g]
                    r = st["root"][st["group"] == g]
                    cov = m[:, 0] == 1
                    if len(m) >= 2:
                        assert not cov.all()
                    assert cov.sum() <= 3
                    if mech_name in ("lag", "event"):
                        assert not (cov & ~r).any()
                    assert (m[cov, 1] == 0).all()
                    assert (m[~cov, 2] == 0).all()
                    cut = m[cov, 2]
                    assert ((cut == 0) | ((cut >= 384) & (cut < 512))).all()
                    roled += int(cov.any())
        assert roled > 0
        assert planner.stats["role_groups"] == roled

    def test_zero_share_assigns_nothing(self) -> None:
        planner = roles.RolePlanner(
            roles.settings({"p_role": [0.0, 0.0, 0.0]}), seed=1, stage_steps=10
        )
        st = self._structure(np.random.default_rng(0), "mix")
        meta = planner.plan_pool(st, np.array([True]), 512, 32)
        assert (meta == 0).all() and planner.stats["groups"] > 0

    def test_phase_from_step(self) -> None:
        planner = roles.RolePlanner(_stage2_settings(), seed=0, stage_steps=100)
        planner.step = 26
        assert planner.p_role() == 0.1
        planner.step = 27
        assert planner.p_role() == 0.2
        planner.step = 60
        assert planner.p_role() == 0.4


class TestFamilies:
    @pytest.mark.parametrize("name", families.FAMILIES)
    @pytest.mark.parametrize("seed", range(4))
    def test_make_task(self, name, seed) -> None:
        rng = np.random.default_rng(seed)
        hist, H, kmax = 1024, 128, 6
        b = _base(seed, n=hist + H, nan_prefix=100)
        y, c = families.make_task(name, b, rng, hist, H, kmax, _window_fn(seed))
        assert y.dtype == np.float32 and c.dtype == np.float32
        assert y.shape == (hist + H,)
        assert c.ndim == 2 and 1 <= len(c) <= kmax and c.shape[1] == hist + H
        assert not np.isinf(y).any() and not np.isinf(c).any()
        assert np.isfinite(y[:hist]).any()
        if name != "classical":
            assert np.isnan(y[:100]).all()

    def test_generic_k_covariates(self) -> None:
        y, c = generic.make_task(_base(0, n=600), np.random.default_rng(0), 512, 64, 5)
        assert c.shape == (5, 576) and y.shape == (576,)

    def test_parse_mix(self) -> None:
        names, p = families.parse_mix({"generic": 1.0, "lagged": 3.0, "energy": 0.0})
        assert names == ("generic", "lagged") and np.allclose(p, [0.25, 0.75])
        with pytest.raises(ValueError):
            families.parse_mix({"generic": 0.0})


class TestSyntheticWindow:
    @pytest.mark.parametrize("seed", range(5))
    def test_shapes(self, seed) -> None:
        base = sc.BaseWindows(
            [
                (_base(seed, n=3000).astype(np.float32), "H"),
                (_base(seed + 1, n=500).astype(np.float32), "D"),
            ]
        )
        y, c = sc.synthetic_window(base, np.random.default_rng(seed), 1024, 256)
        assert y.shape == (1280,) and y.dtype == np.float32
        assert (
            c.ndim == 2
            and 1 <= len(c) <= 6
            and c.shape[1] == 1280
            and c.dtype == np.float32
        )

    def test_deterministic(self) -> None:
        base = sc.BaseWindows([(_base(1, n=3000).astype(np.float32), "H")])
        a = sc.synthetic_window(base, np.random.default_rng(7), 512, 64)
        b = sc.synthetic_window(base, np.random.default_rng(7), 512, 64)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
