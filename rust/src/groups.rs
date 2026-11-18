//! Kernels behind `synforecast.groups`: pooled series draws, effect
//! composition, and the recursive filters of grouped tasks.
//!
//! Every kernel is causal: the output at time `t` reads inputs at or before
//! `t`, plus statistics of the first `burn_in` steps (the calibration window),
//! which the Python side discards.

use rayon::prelude::*;

use crate::batch::{dispatch_generator, get_or_create_pool, GEN_KERNEL_SYNTH};
use crate::generators::kernel_synth::kernel_synth_with_status;
use crate::group_params::{param_seed, randomize, Randomizer};
use crate::rng::SfRng;

/// Emitted values must stay finite and below this magnitude.
pub const MAX_ABS: f64 = 1e8;
/// Standard deviations below this on the calibration window are degenerate.
pub const MIN_STD: f64 = 1e-8;
/// Largest |log multiplier| of a multiplicative effect (ln MAX_ABS); beyond it the composition falls back to additive.
pub const MAX_LOG_MULTIPLIER: f64 = 18.420_680_743_952_367;

pub const KIND_LINEAR: u8 = 0;
pub const KIND_SATURATING: u8 = 1;
pub const KIND_QUADRATIC: u8 = 2;
pub const KIND_SIGNED_SQRT: u8 = 3;
pub const KIND_ABS_SQRT: u8 = 4;
pub const KIND_THRESHOLD: u8 = 5;
pub const KIND_HINGE: u8 = 6;
pub const KIND_PRODUCT: u8 = 7;
pub const KIND_GATE: u8 = 8;

pub const CAL_CALIBRATED: i32 = 0;
pub const CAL_NULL: i32 = 1;
pub const CAL_NOMINAL_EFFECT: i32 = 2;
pub const CAL_NOMINAL_BASE: i32 = 3;
pub const CAL_FIXED: i32 = 4;

pub const COMP_ADDITIVE: i32 = 0;
pub const COMP_MULTIPLICATIVE: i32 = 1;

fn run<T: Send>(n_workers: usize, work: impl FnOnce() -> T + Send) -> Result<T, String> {
    if n_workers > 0 {
        Ok(get_or_create_pool(n_workers)?.install(work))
    } else {
        Ok(work())
    }
}

/// Split `flat` into consecutive mutable chunks of the given lengths.
fn chunks_mut<'a>(mut flat: &'a mut [f64], lengths: &[usize]) -> Vec<&'a mut [f64]> {
    let mut out = Vec::with_capacity(lengths.len());
    for &len in lengths {
        let (head, tail) = flat.split_at_mut(len);
        out.push(head);
        flat = tail;
    }
    out
}

/// One generator configuration: a batch type id with its parameters, and an
/// optional [`Randomizer`] that re-samples constructor-randomized parameters
/// per draw.
pub struct SeriesSpec {
    pub gen_type: i32,
    pub scalar_params: Vec<f64>,
    pub array_params: Vec<Vec<f64>>,
    pub randomizer: Option<Randomizer>,
}

/// Outcome of one draw request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawStatus {
    Ok,
    /// The generator returned its white-noise fallback (KernelSynth).
    Fallback,
    /// The generator rejected its parameters or panicked; values are zeros.
    Error(String),
}

/// The parameters one draw uses: the template, re-sampled when randomized.
pub fn draw_params(spec: &SeriesSpec, seed: u64) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let mut sp = spec.scalar_params.clone();
    let mut ap = spec.array_params.clone();
    if let Some(r) = &spec.randomizer {
        let mut rng = SfRng::new(param_seed(seed));
        randomize(r, &mut sp, &mut ap, &mut rng)?;
    }
    Ok((sp, ap))
}

fn draw_one(spec: &SeriesSpec, out: &mut [f64], seed: u64) -> DrawStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let owned;
        let (sp, ap) = if spec.randomizer.is_some() {
            owned = draw_params(spec, seed)?;
            (&owned.0, &owned.1)
        } else {
            (&spec.scalar_params, &spec.array_params)
        };
        if spec.gen_type == GEN_KERNEL_SYNTH {
            if ap.len() < 2 {
                return Err("kernel_synth: expected >= 2 array params".to_string());
            }
            kernel_synth_with_status(out, sp, &ap[0], &ap[1], seed)
        } else {
            dispatch_generator(spec.gen_type, out, sp, ap, seed).map(|()| false)
        }
    }));
    match result {
        Ok(Ok(false)) => DrawStatus::Ok,
        Ok(Ok(true)) => DrawStatus::Fallback,
        Ok(Err(msg)) => DrawStatus::Error(msg),
        Err(_) => DrawStatus::Error("generator panicked".to_string()),
    }
}

/// Draw one series per request in parallel into one flat buffer.
///
/// Request `i` draws `lengths[i]` values from `specs[spec_index[i]]` with
/// `seeds[i]`. Values are concatenated in request order; every request gets
/// its own [`DrawStatus`], so one failing draw never aborts the others.
/// Generator panics are caught and reported as errors.
pub fn draw_series(
    specs: &[SeriesSpec],
    spec_index: &[u32],
    lengths: &[usize],
    seeds: &[u64],
    n_workers: usize,
) -> Result<(Vec<f64>, Vec<DrawStatus>), String> {
    let n = lengths.len();
    if spec_index.len() != n || seeds.len() != n {
        return Err("spec_index, lengths and seeds must have equal lengths".into());
    }
    if let Some(&bad) = spec_index.iter().find(|&&i| i as usize >= specs.len()) {
        return Err(format!("spec index {bad} out of range"));
    }
    let total = lengths
        .iter()
        .try_fold(0usize, |acc, &l| acc.checked_add(l))
        .ok_or("total length overflows")?;
    let mut flat: Vec<f64> = Vec::new();
    flat.try_reserve_exact(total)
        .map_err(|e| format!("cannot allocate {total} values: {e}"))?;
    flat.resize(total, 0.0);
    let chunks = chunks_mut(&mut flat, lengths);
    let statuses: Vec<DrawStatus> = run(n_workers, || {
        chunks
            .into_par_iter()
            .zip(spec_index.par_iter())
            .zip(seeds.par_iter())
            .map(|((out, &si), &seed)| draw_one(&specs[si as usize], out, seed))
            .collect()
    })?;
    Ok((flat, statuses))
}

/// `y[t] = phi * y[t-1] + x[t]` in place, with `y[-1] = init`.
#[inline]
pub fn ar_filter(x: &mut [f64], phi: f64, init: f64) {
    let mut prev = init;
    for v in x.iter_mut() {
        prev = phi * prev + *v;
        *v = prev;
    }
}

/// [`ar_filter`] started at the steady state of `x[0]`.
#[inline]
pub fn ar_filter_steady(x: &mut [f64], phi: f64) {
    if phi == 0.0 || x.is_empty() {
        return;
    }
    let init = x[0] / (1.0 - phi);
    ar_filter(x, phi, init);
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

/// Population standard deviation (two-pass, as numpy's `std`).
fn std(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
}

fn var(x: &[f64]) -> f64 {
    let s = std(x);
    s * s
}

fn ok(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite() && v.abs() < MAX_ABS)
}

/// Whether every applied log multiplier `scale * e` is within [-MAX_LOG_MULTIPLIER, MAX_LOG_MULTIPLIER].
fn multiplier_ok(effect: &[f64], scale: f64) -> bool {
    effect
        .iter()
        .all(|e| (scale * e).abs() <= MAX_LOG_MULTIPLIER)
}

/// `x[t - lag]`, holding the first value before the start (no wrap-around).
#[inline]
fn lagged(x: &[f64], t: usize, lag: usize) -> f64 {
    x[t.saturating_sub(lag)]
}

/// One effect term (see `synforecast.groups.Term`).
#[derive(Clone, Copy)]
pub struct Term {
    pub kind: u8,
    pub driver: usize,
    pub partner: Option<usize>,
    pub lag: usize,
    pub partner_lag: usize,
    pub coef: f64,
    pub param: Option<f64>,
    pub adstock: Option<f64>,
}

fn edge(kind: u8, x: f64, param: Option<f64>) -> f64 {
    match kind {
        KIND_LINEAR => x,
        KIND_SATURATING => {
            let s = param.unwrap_or(1.0);
            s * (x / s).tanh()
        }
        KIND_QUADRATIC => x * x,
        KIND_SIGNED_SQRT => x.signum() * x.abs().sqrt(),
        KIND_ABS_SQRT => x.abs().sqrt(),
        KIND_THRESHOLD => f64::from(u8::from(x > param.unwrap_or(0.0))),
        KIND_HINGE => (x - param.unwrap_or(0.0)).max(0.0),
        _ => f64::NAN,
    }
}

/// Values of one term over the whole path, drivers given row-major `(K, T)`.
fn term_values(term: &Term, z: &[f64], t_len: usize) -> Vec<f64> {
    let row = |i: usize| &z[i * t_len..(i + 1) * t_len];
    let x = row(term.driver);
    let mut out: Vec<f64> = match (term.kind, term.partner) {
        (KIND_PRODUCT, Some(p)) => {
            let w = row(p);
            (0..t_len)
                .map(|t| lagged(x, t, term.lag) * lagged(w, t, term.partner_lag))
                .collect()
        }
        (KIND_GATE, Some(p)) => {
            let w = row(p);
            let tau = term.param.unwrap_or(0.0);
            (0..t_len)
                .map(|t| {
                    let open = lagged(w, t, term.partner_lag) > tau;
                    if open {
                        lagged(x, t, term.lag)
                    } else {
                        0.0
                    }
                })
                .collect()
        }
        _ => (0..t_len)
            .map(|t| edge(term.kind, lagged(x, t, term.lag), term.param))
            .collect(),
    };
    out.iter_mut().for_each(|v| *v *= term.coef);
    if let Some(decay) = term.adstock {
        ar_filter_steady(&mut out, decay);
        out.iter_mut().for_each(|v| *v *= 1.0 - decay);
    }
    out
}

/// Inputs of one group's composition.
pub struct ComposeInput<'a> {
    /// Target base, length `T` (fixed regressions include intercept and noise).
    pub base: &'a [f64],
    /// Raw drivers, row-major `(K, T)`.
    pub drivers: &'a [f64],
    pub n_drivers: usize,
    pub terms: &'a [Term],
    pub persistence: f64,
    pub target_ar: f64,
    pub composition: i32,
    pub share: f64,
    pub log_std: f64,
}

/// Outputs of one group's composition.
pub struct ComposeOutput {
    pub y: Vec<f64>,
    /// Applied effect: `scale * (effect - center)`, additive or log-space.
    pub effect: Vec<f64>,
    pub cal_mean: Vec<f64>,
    pub cal_std: Vec<f64>,
    pub center: f64,
    pub scale: f64,
    pub calibration: i32,
    pub composition: i32,
    pub ok: bool,
    pub realized: f64,
}

/// Compose one group's target from its base and drivers.
///
/// With `calibrate`, drivers are standardized with calibration-window
/// statistics, the summed effect is centered on that window and scaled to the
/// requested strength (see `GroupPrior`); a failing multiplicative
/// composition (a non-finite or too large target, or a multiplier outside
/// [1/MAX_ABS, MAX_ABS] on the emitted window) falls back to additive. Without `calibrate` (fixed
/// regressions), terms act on raw drivers with their literal coefficients.
pub fn compose(input: &ComposeInput, burn_in: usize, calibrate: bool) -> ComposeOutput {
    let t_len = input.base.len();
    let b = burn_in.min(t_len);
    let k = input.n_drivers;

    let mut cal_mean = vec![0.0; k];
    let mut cal_std = vec![1.0; k];
    let z: Vec<f64> = if calibrate {
        let mut z = input.drivers.to_vec();
        for i in 0..k {
            let row = &mut z[i * t_len..(i + 1) * t_len];
            let m = mean(&row[..b]);
            let mut s = std(&row[..b]);
            if s < MIN_STD {
                s = 1.0;
            }
            cal_mean[i] = m;
            cal_std[i] = s;
            row.iter_mut().for_each(|v| *v = (*v - m) / s);
        }
        z
    } else {
        input.drivers.to_vec()
    };

    let mut effect = vec![0.0; t_len];
    for term in input.terms {
        let v = term_values(term, &z, t_len);
        effect.iter_mut().zip(&v).for_each(|(e, x)| *e += x);
    }
    ar_filter_steady(&mut effect, input.persistence);

    let (center, effect_std, base_calibration) = if calibrate {
        let center = mean(&effect[..b]);
        effect.iter_mut().for_each(|v| *v -= center);
        let mut effect_std = std(&effect[..b]);
        let mut calibration = CAL_CALIBRATED;
        if input.terms.is_empty() {
            calibration = CAL_NULL;
            effect_std = 1.0;
        } else if effect_std < MIN_STD {
            calibration = CAL_NOMINAL_EFFECT;
            effect_std = 1.0;
        }
        (center, effect_std, calibration)
    } else {
        (0.0, 1.0, CAL_FIXED)
    };

    // Returns (target, scale, calibration) for one composition.
    let compose_with = |composition: i32| -> (Vec<f64>, f64, i32) {
        let mut calibration = base_calibration;
        let scale = if !calibrate {
            1.0
        } else if composition == COMP_MULTIPLICATIVE {
            input.log_std / effect_std
        } else {
            let mut base_std = std(&input.base[..b]);
            if base_std < MIN_STD {
                if calibration == CAL_CALIBRATED {
                    calibration = CAL_NOMINAL_BASE;
                }
                base_std = 1.0;
            }
            (input.share / (1.0 - input.share)).sqrt() * base_std / effect_std
        };
        let mut y: Vec<f64> = if composition == COMP_MULTIPLICATIVE {
            input
                .base
                .iter()
                .zip(&effect)
                .map(|(b, e)| b * (scale * e).exp())
                .collect()
        } else {
            input
                .base
                .iter()
                .zip(&effect)
                .map(|(b, e)| b + scale * e)
                .collect()
        };
        ar_filter_steady(&mut y, input.target_ar);
        (y, scale, calibration)
    };

    let mut composition = input.composition;
    let (mut y, mut scale, mut calibration) = compose_with(composition);
    let mut good =
        ok(&y[b..]) && (composition != COMP_MULTIPLICATIVE || multiplier_ok(&effect[b..], scale));
    if !good && composition == COMP_MULTIPLICATIVE {
        composition = COMP_ADDITIVE;
        (y, scale, calibration) = compose_with(composition);
        good = ok(&y[b..]);
    }

    let applied: Vec<f64> = effect.iter().map(|e| scale * e).collect();
    let realized = if composition == COMP_MULTIPLICATIVE {
        std(&applied[b..])
    } else {
        let ve = var(&applied[b..]);
        let vb = var(&input.base[b..]);
        if ve + vb > 0.0 {
            ve / (ve + vb)
        } else {
            0.0
        }
    };
    ComposeOutput {
        y,
        effect: applied,
        cal_mean,
        cal_std,
        center,
        scale,
        calibration,
        composition,
        ok: good,
        realized,
    }
}

/// Compose many groups in parallel.
pub fn compose_batch(
    inputs: &[ComposeInput],
    burn_in: usize,
    calibrate: bool,
    n_workers: usize,
) -> Result<Vec<ComposeOutput>, String> {
    run(n_workers, || {
        inputs
            .par_iter()
            .map(|input| compose(input, burn_in, calibrate))
            .collect()
    })
}

/// Column-wise AR(1) of a row-major `(T, M)` array: `y[t] = phi * y[t-1] + x[t]`
/// with `y[-1] = init`.
pub fn ar_columns(x: &mut [f64], m: usize, phi: f64, init: &[f64]) {
    if m == 0 || x.is_empty() {
        return;
    }
    let mut prev = init.to_vec();
    for row in x.chunks_mut(m) {
        for (v, p) in row.iter_mut().zip(prev.iter_mut()) {
            *p = phi * *p + *v;
            *v = *p;
        }
    }
}

/// VAR recursion: `y[t] = drive[t] + sum_l A_l y[t-1-l]` for `t >= p`, with
/// `y[t] = drive[t]` for `t < p`. `drive` is row-major `(T, N)`, `lags` is
/// `(p, N, N)` row-major.
pub fn var_filter(drive: &[f64], n: usize, lags: &[f64], p: usize) -> Vec<f64> {
    let t_len = drive.len().checked_div(n).unwrap_or(0);
    let mut y = drive.to_vec();
    for t in p..t_len {
        for l in 0..p {
            let a = &lags[l * n * n..(l + 1) * n * n];
            let src = (t - 1 - l) * n;
            for i in 0..n {
                let mut acc = 0.0;
                let row = &a[i * n..(i + 1) * n];
                for j in 0..n {
                    acc += row[j] * y[src + j];
                }
                y[t * n + i] += acc;
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(kind: u8) -> Term {
        Term {
            kind,
            driver: 0,
            partner: None,
            lag: 0,
            partner_lag: 0,
            coef: 1.0,
            param: None,
            adstock: None,
        }
    }

    #[test]
    fn draw_errors_are_per_request() {
        let specs = [
            SeriesSpec {
                gen_type: -42,
                scalar_params: vec![],
                array_params: vec![],
                randomizer: None,
            },
            SeriesSpec {
                gen_type: crate::batch::GEN_RANDOM_WALK,
                scalar_params: vec![0.0, 1.0, 0.0, 0.0, 0.0],
                array_params: vec![],
                randomizer: None,
            },
        ];
        let (values, statuses) = draw_series(&specs, &[0, 1], &[3, 4], &[1, 2], 1).unwrap();
        assert_eq!(values.len(), 7);
        assert!(matches!(statuses[0], DrawStatus::Error(_)));
        assert_eq!(statuses[1], DrawStatus::Ok);
    }

    #[test]
    fn overflowing_lengths_are_rejected() {
        let specs = [SeriesSpec {
            gen_type: crate::batch::GEN_RANDOM_WALK,
            scalar_params: vec![0.0, 1.0, 0.0, 0.0, 0.0],
            array_params: vec![],
            randomizer: None,
        }];
        assert!(draw_series(&specs, &[0, 0], &[usize::MAX, 2], &[1, 2], 1).is_err());
        assert!(draw_series(&specs, &[0], &[usize::MAX / 4], &[1], 1).is_err());
    }

    #[test]
    fn ar_filter_steady_state_keeps_constants() {
        let mut x = vec![2.0; 50];
        ar_filter_steady(&mut x, 0.5);
        assert!(x.iter().all(|v| (v - 4.0).abs() < 1e-12));
    }

    #[test]
    fn adstock_has_unit_gain() {
        let t = Term {
            adstock: Some(0.7),
            ..term(KIND_LINEAR)
        };
        let v = term_values(&t, &[1.0; 30], 30);
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-12));
    }

    #[test]
    fn lags_hold_the_first_value() {
        let z: Vec<f64> = (0..10).map(f64::from).collect();
        let t = Term {
            lag: 3,
            ..term(KIND_LINEAR)
        };
        let v = term_values(&t, &z, 10);
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn compose_is_prefix_invariant() {
        let t_len = 200;
        let base: Vec<f64> = (0..t_len).map(|i| ((i as f64) * 0.3).sin() + 2.0).collect();
        let drivers: Vec<f64> = (0..2 * t_len).map(|i| ((i as f64) * 0.17).cos()).collect();
        let terms = [
            Term {
                lag: 4,
                adstock: Some(0.5),
                ..term(KIND_SATURATING)
            },
            Term {
                kind: KIND_GATE,
                partner: Some(1),
                partner_lag: 2,
                param: Some(0.1),
                ..term(KIND_LINEAR)
            },
        ];
        let make = |base: &[f64], drivers: &[f64], composition| {
            compose(
                &ComposeInput {
                    base,
                    drivers,
                    n_drivers: 2,
                    terms: &terms,
                    persistence: 0.4,
                    target_ar: 0.3,
                    composition,
                    share: 0.6,
                    log_std: 0.2,
                },
                50,
                true,
            )
        };
        for composition in [COMP_ADDITIVE, COMP_MULTIPLICATIVE] {
            let a = make(&base, &drivers, composition);
            let mut base2 = base.clone();
            let mut drivers2 = drivers.clone();
            for i in 120..t_len {
                base2[i] += 5.0;
                drivers2[i] -= 3.0;
                drivers2[t_len + i] += 3.0;
            }
            let b2 = make(&base2, &drivers2, composition);
            assert_eq!(a.y[..120], b2.y[..120]);
            assert_ne!(a.y[120..], b2.y[120..]);
        }
    }

    #[test]
    fn multiplier_outside_bound_falls_back_to_additive() {
        let t_len = 200;
        let burn_in = 50;
        let base: Vec<f64> = (0..t_len).map(|i| ((i as f64) * 0.3).sin() + 2.0).collect();
        let make = |level: f64| {
            let drivers: Vec<f64> = (0..t_len)
                .map(|i| {
                    if i < burn_in {
                        0.1 * ((i as f64) * 0.9).sin()
                    } else {
                        level
                    }
                })
                .collect();
            let terms = [term(KIND_LINEAR)];
            compose(
                &ComposeInput {
                    base: &base,
                    drivers: &drivers,
                    n_drivers: 1,
                    terms: &terms,
                    persistence: 0.0,
                    target_ar: 0.0,
                    composition: COMP_MULTIPLICATIVE,
                    share: 0.6,
                    log_std: 0.2,
                },
                burn_in,
                true,
            )
        };
        // A small drift keeps |log multiplier| inside the bound: multiplicative, every value positive.
        let kept = make(-0.1);
        assert_eq!(kept.composition, COMP_MULTIPLICATIVE);
        assert!(kept.ok && kept.y[burn_in..].iter().all(|v| *v > 0.0));
        // A large drift drives the multiplier toward zero (finite, below MAX_ABS): previously kept, now additive.
        let collapsed = make(-10.0);
        assert_eq!(collapsed.composition, COMP_ADDITIVE);
        assert!(collapsed.ok);
    }

    #[test]
    fn var_filter_matches_manual_recursion() {
        let drive = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let lags = vec![0.5, 0.0, 0.0, 0.25];
        let y = var_filter(&drive, 2, &lags, 1);
        assert_eq!(y, vec![1.0, 0.0, 0.5, 1.0, 0.75, 0.75]);
    }

    #[test]
    fn ar_columns_filters_each_column() {
        let mut x = vec![1.0, 2.0, 1.0, 2.0];
        ar_columns(&mut x, 2, 0.5, &[0.0, 0.0]);
        assert_eq!(x, vec![1.0, 2.0, 1.5, 3.0]);
    }
}
