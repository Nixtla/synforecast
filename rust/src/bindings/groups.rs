//! Python bindings for the grouped-task kernels (`synforecast._lib.groups`).

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::groups;

fn offsets_ok(offsets: &[i64], n: usize, total: usize, name: &str) -> PyResult<()> {
    if offsets.len() != n + 1
        || offsets.first() != Some(&0)
        || offsets.last().copied() != Some(total as i64)
        || offsets.windows(2).any(|w| w[1] < w[0])
    {
        return Err(PyValueError::new_err(format!(
            "{name} must be {} non-decreasing offsets from 0 to {total}",
            n + 1
        )));
    }
    Ok(())
}

fn vec_f64(list: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f64>>> {
    list.iter()
        .map(|item| {
            let arr: PyReadonlyArray1<'_, f64> = item.extract()?;
            Ok(arr.as_slice()?.to_vec())
        })
        .collect()
}

/// Draw one series per request from a list of generator configurations.
///
/// Returns ``(values, status, messages)``: the concatenated values in request
/// order, a ``uint8`` status per request (0 ok, 1 generator noise fallback,
/// 2 error) and the error message per request (empty unless status is 2).
#[pyfunction]
#[pyo3(signature = (gen_types, scalar_params, array_params, spec_index, lengths, seeds, n_workers=0))]
pub fn draw_series<'py>(
    py: Python<'py>,
    gen_types: PyReadonlyArray1<'py, i32>,
    scalar_params: &Bound<'py, PyList>,
    array_params: &Bound<'py, PyList>,
    spec_index: PyReadonlyArray1<'py, u32>,
    lengths: PyReadonlyArray1<'py, i64>,
    seeds: PyReadonlyArray1<'py, u64>,
    n_workers: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let types = gen_types.as_slice()?;
    if scalar_params.len() != types.len() || array_params.len() != types.len() {
        return Err(PyValueError::new_err(
            "gen_types, scalar_params and array_params must have equal lengths",
        ));
    }
    let scalars = vec_f64(scalar_params)?;
    let mut specs = Vec::with_capacity(types.len());
    for ((gen_type, sp), ap) in types.iter().zip(scalars).zip(array_params.iter()) {
        let ap_list: Bound<'_, PyList> = ap.cast_into()?;
        specs.push(groups::SeriesSpec {
            gen_type: *gen_type,
            scalar_params: sp,
            array_params: vec_f64(&ap_list)?,
        });
    }
    let lens: Vec<usize> = lengths
        .as_slice()?
        .iter()
        .map(|&l| usize::try_from(l).map_err(|_| PyValueError::new_err("negative length")))
        .collect::<PyResult<_>>()?;
    let si = spec_index.as_slice()?.to_vec();
    let sd = seeds.as_slice()?.to_vec();
    let (values, statuses) = py
        .detach(|| groups::draw_series(&specs, &si, &lens, &sd, n_workers))
        .map_err(PyValueError::new_err)?;
    let codes: Vec<u8> = statuses
        .iter()
        .map(|s| match s {
            groups::DrawStatus::Ok => 0,
            groups::DrawStatus::Fallback => 1,
            groups::DrawStatus::Error(_) => 2,
        })
        .collect();
    let messages = PyList::empty(py);
    for status in &statuses {
        match status {
            groups::DrawStatus::Error(msg) => messages.append(msg)?,
            _ => messages.append("")?,
        }
    }
    PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, values).into_any(),
            PyArray1::from_vec(py, codes).into_any(),
            messages.into_any(),
        ],
    )
}

/// Compose many groups' targets in parallel (see ``groups::compose``).
///
/// Returns ``(y, effect, cal_mean, cal_std, center, scale, calibration,
/// composition, ok, realized)``; ``y`` and ``effect`` follow ``base_offsets``,
/// ``cal_mean`` and ``cal_std`` hold one value per driver in group order.
#[pyfunction]
#[pyo3(signature = (
    base, base_offsets, drivers, n_drivers, term_offsets, kind, driver, partner, lag,
    partner_lag, coef, param, adstock, persistence, target_ar, composition, share, log_std,
    burn_in, calibrate, n_workers=0
))]
pub fn compose_batch<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, f64>,
    base_offsets: PyReadonlyArray1<'py, i64>,
    drivers: PyReadonlyArray1<'py, f64>,
    n_drivers: PyReadonlyArray1<'py, i64>,
    term_offsets: PyReadonlyArray1<'py, i64>,
    kind: PyReadonlyArray1<'py, u8>,
    driver: PyReadonlyArray1<'py, i64>,
    partner: PyReadonlyArray1<'py, i64>,
    lag: PyReadonlyArray1<'py, i64>,
    partner_lag: PyReadonlyArray1<'py, i64>,
    coef: PyReadonlyArray1<'py, f64>,
    param: PyReadonlyArray1<'py, f64>,
    adstock: PyReadonlyArray1<'py, f64>,
    persistence: PyReadonlyArray1<'py, f64>,
    target_ar: PyReadonlyArray1<'py, f64>,
    composition: PyReadonlyArray1<'py, i32>,
    share: PyReadonlyArray1<'py, f64>,
    log_std: PyReadonlyArray1<'py, f64>,
    burn_in: usize,
    calibrate: bool,
    n_workers: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let base = base.as_slice()?;
    let bo = base_offsets.as_slice()?;
    let drivers = drivers.as_slice()?;
    let nd = n_drivers.as_slice()?;
    let to = term_offsets.as_slice()?;
    let g = nd.len();
    offsets_ok(bo, g, base.len(), "base_offsets")?;
    let n_terms = kind.len()?;
    offsets_ok(to, g, n_terms, "term_offsets")?;
    let per_group = [
        persistence.len()?,
        target_ar.len()?,
        composition.len()?,
        share.len()?,
        log_std.len()?,
    ];
    if per_group.iter().any(|&n| n != g) {
        return Err(PyValueError::new_err(
            "per-group arrays must have one entry per group",
        ));
    }
    let per_term = [
        driver.len()?,
        partner.len()?,
        lag.len()?,
        partner_lag.len()?,
        coef.len()?,
        param.len()?,
        adstock.len()?,
    ];
    if per_term.iter().any(|&n| n != n_terms) {
        return Err(PyValueError::new_err(
            "per-term arrays must have one entry per term",
        ));
    }

    // Driver offsets follow from per-group lengths and driver counts.
    let mut driver_offsets = Vec::with_capacity(g + 1);
    driver_offsets.push(0usize);
    for i in 0..g {
        let t_len = (bo[i + 1] - bo[i]) as usize;
        let k = usize::try_from(nd[i]).map_err(|_| PyValueError::new_err("negative n_drivers"))?;
        let next = k
            .checked_mul(t_len)
            .and_then(|size| driver_offsets[i].checked_add(size))
            .ok_or_else(|| PyValueError::new_err("driver sizes overflow"))?;
        driver_offsets.push(next);
    }
    if driver_offsets[g] != drivers.len() {
        return Err(PyValueError::new_err(
            "drivers length does not match lengths x n_drivers",
        ));
    }

    let (kind, driver, partner, lag, partner_lag) = (
        kind.as_slice()?,
        driver.as_slice()?,
        partner.as_slice()?,
        lag.as_slice()?,
        partner_lag.as_slice()?,
    );
    let (coef, param, adstock) = (coef.as_slice()?, param.as_slice()?, adstock.as_slice()?);
    let mut terms = Vec::with_capacity(n_terms);
    for gi in 0..g {
        let k = nd[gi];
        for ti in to[gi] as usize..to[gi + 1] as usize {
            let in_range = |i: i64| i >= 0 && i < k;
            if !in_range(driver[ti]) || (partner[ti] >= 0 && !in_range(partner[ti])) {
                return Err(PyValueError::new_err(format!(
                    "term {ti} references a driver outside its group"
                )));
            }
            if lag[ti] < 0 || partner_lag[ti] < 0 {
                return Err(PyValueError::new_err("lags must be non-negative"));
            }
            if kind[ti] > groups::KIND_GATE {
                return Err(PyValueError::new_err(format!(
                    "unknown term kind {}",
                    kind[ti]
                )));
            }
            let is_interaction = kind[ti] == groups::KIND_PRODUCT || kind[ti] == groups::KIND_GATE;
            if is_interaction != (partner[ti] >= 0) {
                return Err(PyValueError::new_err(
                    "partner is required for, and only for, interactions",
                ));
            }
            terms.push(groups::Term {
                kind: kind[ti],
                driver: driver[ti] as usize,
                partner: (partner[ti] >= 0).then_some(partner[ti] as usize),
                lag: lag[ti] as usize,
                partner_lag: partner_lag[ti] as usize,
                coef: coef[ti],
                param: (!param[ti].is_nan()).then_some(param[ti]),
                adstock: (!adstock[ti].is_nan()).then_some(adstock[ti]),
            });
        }
    }

    let (persistence, target_ar, composition, share, log_std) = (
        persistence.as_slice()?,
        target_ar.as_slice()?,
        composition.as_slice()?,
        share.as_slice()?,
        log_std.as_slice()?,
    );
    let inputs: Vec<groups::ComposeInput> = (0..g)
        .map(|gi| groups::ComposeInput {
            base: &base[bo[gi] as usize..bo[gi + 1] as usize],
            drivers: &drivers[driver_offsets[gi]..driver_offsets[gi + 1]],
            n_drivers: nd[gi] as usize,
            terms: &terms[to[gi] as usize..to[gi + 1] as usize],
            persistence: persistence[gi],
            target_ar: target_ar[gi],
            composition: composition[gi],
            share: share[gi],
            log_std: log_std[gi],
        })
        .collect();

    let outputs = py
        .detach(|| groups::compose_batch(&inputs, burn_in, calibrate, n_workers))
        .map_err(PyValueError::new_err)?;

    let mut y = Vec::with_capacity(base.len());
    let mut effect = Vec::with_capacity(base.len());
    let mut cal_mean = Vec::new();
    let mut cal_std = Vec::new();
    let mut center = Vec::with_capacity(g);
    let mut scale = Vec::with_capacity(g);
    let mut calibration = Vec::with_capacity(g);
    let mut comp = Vec::with_capacity(g);
    let mut ok = Vec::with_capacity(g);
    let mut realized = Vec::with_capacity(g);
    for out in outputs {
        y.extend_from_slice(&out.y);
        effect.extend_from_slice(&out.effect);
        cal_mean.extend_from_slice(&out.cal_mean);
        cal_std.extend_from_slice(&out.cal_std);
        center.push(out.center);
        scale.push(out.scale);
        calibration.push(out.calibration);
        comp.push(out.composition);
        ok.push(u8::from(out.ok));
        realized.push(out.realized);
    }
    PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, y).into_any(),
            PyArray1::from_vec(py, effect).into_any(),
            PyArray1::from_vec(py, cal_mean).into_any(),
            PyArray1::from_vec(py, cal_std).into_any(),
            PyArray1::from_vec(py, center).into_any(),
            PyArray1::from_vec(py, scale).into_any(),
            PyArray1::from_vec(py, calibration).into_any(),
            PyArray1::from_vec(py, comp).into_any(),
            PyArray1::from_vec(py, ok).into_any(),
            PyArray1::from_vec(py, realized).into_any(),
        ],
    )
}

/// Column-wise AR(1) of a row-major ``(T, m)`` array with initial state ``init``.
#[pyfunction]
pub fn ar_columns<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    m: usize,
    phi: f64,
    init: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut v = x.as_slice()?.to_vec();
    let init = init.as_slice()?.to_vec();
    if m == 0 {
        if !v.is_empty() {
            return Err(PyValueError::new_err(
                "m must be positive for a non-empty array",
            ));
        }
    } else if v.len() % m != 0 || init.len() != m {
        return Err(PyValueError::new_err(
            "x must be (T, m) row-major and init must have m entries",
        ));
    }
    py.detach(|| groups::ar_columns(&mut v, m, phi, &init));
    Ok(PyArray1::from_vec(py, v))
}

/// VAR recursion of a row-major ``(T, n)`` drive with ``p`` row-major ``(n, n)`` lag matrices.
#[pyfunction]
pub fn var_filter<'py>(
    py: Python<'py>,
    drive: PyReadonlyArray1<'py, f64>,
    n: usize,
    lags: PyReadonlyArray1<'py, f64>,
    p: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let drive = drive.as_slice()?.to_vec();
    let lags = lags.as_slice()?.to_vec();
    let lag_size = n.checked_mul(n).and_then(|nn| nn.checked_mul(p));
    if n == 0 || drive.len() % n != 0 || lag_size != Some(lags.len()) {
        return Err(PyValueError::new_err(
            "drive must be (T, n) and lags (p, n, n)",
        ));
    }
    let y = py.detach(|| groups::var_filter(&drive, n, &lags, p));
    Ok(PyArray1::from_vec(py, y))
}

/// Register the `groups` submodule under `synforecast._lib`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "groups")?;
    m.add_function(wrap_pyfunction!(draw_series, &m)?)?;
    m.add_function(wrap_pyfunction!(compose_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(ar_columns, &m)?)?;
    m.add_function(wrap_pyfunction!(var_filter, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.groups", &m)?;
    Ok(())
}
