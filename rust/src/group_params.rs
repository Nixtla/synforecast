//! Native per-draw parameter sampling for generators that randomize their
//! parameters in the constructor (SARIMA, ETS, GARCH, INAR, StateSpace).
//!
//! `synforecast.groups` resolves such a generator once into a template of
//! batch parameters plus a [`Randomizer`] describing which slots the
//! constructor randomizes and how. Each draw then re-samples those slots
//! natively from its own seed instead of rebuilding the Python generator.
//! The distributions follow the Python constructors; the random streams are
//! SynForecast's native ones.

use nalgebra::DMatrix;

use crate::rng::SfRng;

pub const RAND_SARIMA: u8 = 1;
pub const RAND_ETS: u8 = 2;
pub const RAND_GARCH: u8 = 3;
pub const RAND_INAR: u8 = 4;
pub const RAND_STATE_SPACE: u8 = 5;

/// Which constructor-randomized slots to re-sample, with their configuration.
///
/// Layouts (`config`, `fixed`):
/// - SARIMA: `[p, q, P, Q, s, rand_ar, rand_ma, rand_sar, rand_sma, burn_in]`
///   with `burn_in` NaN when computed; `fixed = [ar, ma, sar, sma]`.
/// - ETS: `[seasonal_type, m]` (1 additive, 2 multiplicative).
/// - GARCH: `[p, q, rand_alpha, rand_beta]`; `fixed = [alpha, beta]`.
/// - INAR: `[p]`.
/// - StateSpace: `[state_dim]`.
#[derive(Clone, Debug)]
pub struct Randomizer {
    pub kind: u8,
    pub config: Vec<f64>,
    pub fixed: Vec<Vec<f64>>,
}

/// Uniform on `[a, b)`, also for `a >= b` (like NumPy's `uniform`).
#[inline]
fn uniform(rng: &mut SfRng, a: f64, b: f64) -> f64 {
    a + (b - a) * rng.uniform01()
}

fn uniforms(rng: &mut SfRng, n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n).map(|_| uniform(rng, a, b)).collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Stable AR coefficients from uniform partial autocorrelations
/// (Durbin-Levinson), as `SARIMAGenerator._generate_stable_ar_params`.
pub fn stable_ar(rng: &mut SfRng, order: usize, bound: f64) -> Vec<f64> {
    let pacf = uniforms(rng, order, -bound, bound);
    let mut ar = vec![0.0; order];
    if order == 0 {
        return ar;
    }
    ar[0] = pacf[0];
    for k in 1..order {
        let prev = ar[..k].to_vec();
        ar[k] = pacf[k];
        for j in 0..k {
            ar[j] = prev[j] - pacf[k] * prev[k - 1 - j];
        }
    }
    ar
}

fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// Full AR lag coefficients of `phi(B) Phi(B^s)` (positive convention).
pub fn full_ar(ar: &[f64], sar: &[f64], s: usize) -> Vec<f64> {
    let mut poly = vec![1.0];
    poly.extend(ar.iter().map(|v| -v));
    let mut seasonal = vec![0.0; sar.len() * s + 1];
    seasonal[0] = 1.0;
    for (i, v) in sar.iter().enumerate() {
        seasonal[(i + 1) * s] = -v;
    }
    convolve(&poly, &seasonal)[1..].iter().map(|v| -v).collect()
}

/// Full MA lag coefficients of `theta(B) Theta(B^s)`.
pub fn full_ma(ma: &[f64], sma: &[f64], s: usize) -> Vec<f64> {
    let mut poly = vec![1.0];
    poly.extend_from_slice(ma);
    let mut seasonal = vec![0.0; sma.len() * s + 1];
    seasonal[0] = 1.0;
    for (i, v) in sma.iter().enumerate() {
        seasonal[(i + 1) * s] = *v;
    }
    convolve(&poly, &seasonal)[1..].to_vec()
}

/// Burn-in of `SARIMAGenerator._compute_burn_in`.
pub fn sarima_burn_in(ar_full: &[f64], ma_full: &[f64]) -> f64 {
    let max_lag = ar_full.len().max(ma_full.len()) as f64;
    let multiplier = if ar_full.is_empty() {
        2.0
    } else {
        let persistence: f64 = ar_full.iter().sum();
        if persistence > 0.9 {
            (10.0 / (1.0 - persistence + 0.01)).trunc()
        } else {
            5.0
        }
    };
    (max_lag * multiplier).max(100.0)
}

fn count(config: &[f64], i: usize) -> Result<usize, String> {
    let v = *config
        .get(i)
        .ok_or_else(|| format!("randomizer config needs index {i}"))?;
    if !(v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v < 1e6) {
        return Err(format!("randomizer config entry {i} must be a small count"));
    }
    Ok(v as usize)
}

fn flag(config: &[f64], i: usize) -> Result<bool, String> {
    Ok(*config
        .get(i)
        .ok_or_else(|| format!("randomizer config needs index {i}"))?
        != 0.0)
}

fn fixed(r: &Randomizer, i: usize) -> Result<&[f64], String> {
    r.fixed
        .get(i)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("randomizer needs fixed array {i}"))
}

fn slot(ap: &mut [Vec<f64>], i: usize) -> Result<&mut Vec<f64>, String> {
    ap.get_mut(i)
        .ok_or_else(|| format!("template needs array parameter {i}"))
}

/// Re-sample the randomized slots of a template in place.
pub fn randomize(
    r: &Randomizer,
    sp: &mut [f64],
    ap: &mut [Vec<f64>],
    rng: &mut SfRng,
) -> Result<(), String> {
    let c = &r.config;
    match r.kind {
        RAND_SARIMA => {
            let (p, q, big_p, big_q, s) = (
                count(c, 0)?,
                count(c, 1)?,
                count(c, 2)?,
                count(c, 3)?,
                count(c, 4)?.max(1),
            );
            let bound = if big_p > 0 || big_q > 0 { 0.4 } else { 0.6 };
            let ar = if flag(c, 5)? && p > 0 {
                stable_ar(rng, p, bound)
            } else {
                fixed(r, 0)?.to_vec()
            };
            let ma = if flag(c, 6)? && q > 0 {
                uniforms(rng, q, -0.5, 0.5)
            } else {
                fixed(r, 1)?.to_vec()
            };
            let sar = if flag(c, 7)? && big_p > 0 {
                stable_ar(rng, big_p, bound)
            } else {
                fixed(r, 2)?.to_vec()
            };
            let sma = if flag(c, 8)? && big_q > 0 {
                uniforms(rng, big_q, -0.5, 0.5)
            } else {
                fixed(r, 3)?.to_vec()
            };
            let ar_full = full_ar(&ar, &sar, s);
            let ma_full = full_ma(&ma, &sma, s);
            let burn_in = *c.get(9).ok_or("randomizer config needs index 9")?;
            let burn_in = if burn_in.is_nan() {
                sarima_burn_in(&ar_full, &ma_full)
            } else {
                burn_in
            };
            *sp.get_mut(6).ok_or("sarima template needs 9 scalars")? = burn_in;
            *slot(ap, 0)? = ar_full;
            *slot(ap, 1)? = ma_full;
        }
        RAND_ETS => {
            let kind = count(c, 0)?;
            let m = count(c, 1)?;
            if m == 0 {
                return Err("ets: seasonal period must be positive".into());
            }
            let seasonal = match kind {
                1 => {
                    let raw = uniforms(rng, m, -5.0, 5.0);
                    let mu = mean(&raw);
                    raw.iter().map(|v| v - mu).collect()
                }
                2 => {
                    let raw = uniforms(rng, m, 0.8, 1.2);
                    let mu = mean(&raw);
                    raw.iter().map(|v| v / mu).collect()
                }
                _ => return Err(format!("ets: unknown seasonal type {kind}")),
            };
            *slot(ap, 0)? = seasonal;
        }
        RAND_GARCH => {
            let (p, q) = (count(c, 0)?, count(c, 1)?);
            let alpha = if flag(c, 2)? {
                uniforms(rng, q, 0.05, 0.15)
            } else {
                fixed(r, 0)?.to_vec()
            };
            let beta = if flag(c, 3)? {
                if p == 0 {
                    Vec::new()
                } else {
                    let remaining = 0.95 - alpha.iter().sum::<f64>();
                    let high = remaining / p as f64;
                    let low = 0.1_f64.min(high / 2.0);
                    uniforms(rng, p, low, high)
                }
            } else {
                fixed(r, 1)?.to_vec()
            };
            if alpha.iter().sum::<f64>() + beta.iter().sum::<f64>() >= 1.0 {
                return Err("garch: sampled alpha + beta is not stationary".into());
            }
            *slot(ap, 0)? = alpha;
            *slot(ap, 1)? = beta;
        }
        RAND_INAR => {
            let p = count(c, 0)?;
            if p == 0 {
                return Err("inar: p must be positive".into());
            }
            *slot(ap, 0)? = uniforms(rng, p, 0.1, 0.8 / p as f64);
        }
        RAND_STATE_SPACE => {
            let d = count(c, 0)?;
            if d == 0 {
                return Err("state_space: state_dim must be positive".into());
            }
            let mut f = uniforms(rng, d * d, -0.3, 0.3);
            let radius = DMatrix::from_row_slice(d, d, &f)
                .complex_eigenvalues()
                .iter()
                .map(|z| z.norm())
                .fold(0.0_f64, f64::max);
            if radius >= 1.0 {
                f.iter_mut().for_each(|v| *v *= 0.9 / radius);
            }
            *slot(ap, 0)? = f;
        }
        other => return Err(format!("unknown randomizer kind {other}")),
    }
    Ok(())
}

/// Seed of a draw's parameter stream, distinct from its kernel seed.
#[inline]
pub fn param_seed(seed: u64) -> u64 {
    (seed ^ 0xD1B5_4A32_D192_ED03).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_ar_matches_known_recursion() {
        // Two lags: phi_2 = pacf_2, phi_1 = pacf_1 - pacf_2 * pacf_1
        let mut rng = SfRng::new(3);
        let ar = stable_ar(&mut rng, 2, 0.6);
        let mut rng = SfRng::new(3);
        let pacf = uniforms(&mut rng, 2, -0.6, 0.6);
        assert!((ar[1] - pacf[1]).abs() < 1e-15);
        assert!((ar[0] - (pacf[0] - pacf[1] * pacf[0])).abs() < 1e-15);
    }

    #[test]
    fn seasonal_polynomials_expand() {
        // (1 - 0.5B)(1 - 0.3B^2) = 1 - 0.5B - 0.3B^2 + 0.15B^3
        assert_eq!(full_ar(&[0.5], &[0.3], 2), vec![0.5, 0.3, -0.15]);
        // (1 + 0.4B)(1 + 0.2B^2) = 1 + 0.4B + 0.2B^2 + 0.08B^3
        let ma = full_ma(&[0.4], &[0.2], 2);
        assert!((ma[2] - 0.08).abs() < 1e-15 && ma.len() == 3);
    }

    #[test]
    fn ets_seasonal_states_are_normalized() {
        let mut ap = vec![vec![]];
        let r = Randomizer {
            kind: RAND_ETS,
            config: vec![1.0, 12.0],
            fixed: vec![],
        };
        randomize(&r, &mut [], &mut ap, &mut SfRng::new(1)).unwrap();
        assert_eq!(ap[0].len(), 12);
        assert!(mean(&ap[0]).abs() < 1e-12);
        let r = Randomizer {
            config: vec![2.0, 7.0],
            ..r
        };
        randomize(&r, &mut [], &mut ap, &mut SfRng::new(1)).unwrap();
        assert!((mean(&ap[0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn state_space_transitions_are_stable() {
        let r = Randomizer {
            kind: RAND_STATE_SPACE,
            config: vec![6.0],
            fixed: vec![],
        };
        for seed in 0..50 {
            let mut ap = vec![vec![]];
            randomize(&r, &mut [], &mut ap, &mut SfRng::new(seed)).unwrap();
            let radius = DMatrix::from_row_slice(6, 6, &ap[0])
                .complex_eigenvalues()
                .iter()
                .map(|z| z.norm())
                .fold(0.0_f64, f64::max);
            assert!(radius < 1.0);
        }
    }

    #[test]
    fn bad_configurations_are_errors() {
        let r = Randomizer {
            kind: RAND_INAR,
            config: vec![f64::NAN],
            fixed: vec![],
        };
        assert!(randomize(&r, &mut [], &mut [vec![]], &mut SfRng::new(0)).is_err());
        let r = Randomizer {
            kind: 99,
            config: vec![],
            fixed: vec![],
        };
        assert!(randomize(&r, &mut [], &mut [], &mut SfRng::new(0)).is_err());
    }
}
