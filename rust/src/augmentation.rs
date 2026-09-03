//! Native algorithms shared by augmentation and feature-targeted generation.

use std::f64::consts::PI;

use crate::{fft, rng::SfRng};

pub type Decomposition = (Vec<f64>, Vec<f64>, Vec<f64>);

fn validate_values(values: &[f64]) -> Result<(), String> {
    if values.len() < 3 {
        return Err("values must contain at least 3 observations".to_string());
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err("values must contain only finite observations".to_string());
    }
    Ok(())
}

/// Return square-root DTW distance and an optimal alignment path.
pub fn dtw_alignment(
    a: &[f64],
    b: &[f64],
    band: Option<usize>,
) -> Result<(f64, Vec<(usize, usize)>), String> {
    if a.is_empty() || b.is_empty() {
        return Err("DTW inputs must be non-empty one-dimensional arrays".to_string());
    }
    if !a.iter().chain(b).all(|value| value.is_finite()) {
        return Err("DTW inputs must be finite".to_string());
    }
    let (n, m) = (a.len(), b.len());
    let width = band.unwrap_or(n.max(m)).max(n.abs_diff(m));
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;
    let mut parents = vec![u8::MAX; n * m];
    for i in 1..=n {
        let mut current = vec![f64::INFINITY; m + 1];
        let lower = 1.max(i.saturating_sub(width));
        let upper = m.min(i.saturating_add(width));
        for j in lower..=upper {
            let mut best = previous[j - 1];
            let mut direction = 0;
            if previous[j] < best {
                best = previous[j];
                direction = 1;
            }
            if current[j - 1] < best {
                best = current[j - 1];
                direction = 2;
            }
            current[j] = (a[i - 1] - b[j - 1]).powi(2) + best;
            parents[(i - 1) * m + j - 1] = direction;
        }
        previous = current;
    }
    if !previous[m].is_finite() {
        return Err("DTW alignment is infeasible for the requested band".to_string());
    }

    let (mut i, mut j) = (n, m);
    let mut path = Vec::with_capacity(n + m);
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        match parents[(i - 1) * m + j - 1] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            2 => j -= 1,
            _ => return Err("DTW alignment is infeasible for the requested band".to_string()),
        }
    }
    path.reverse();
    Ok((previous[m].sqrt(), path))
}

/// Compute a weighted DBA barycenter restricted to reference length.
pub fn dba_barycenter(
    reference: &[f64],
    neighbors: &[Vec<f64>],
    weights: &[f64],
    n_iterations: usize,
    band: Option<usize>,
) -> Result<Vec<f64>, String> {
    let n_series = neighbors.len() + 1;
    if weights.len() != n_series
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err("weights must be non-negative and match all input series".to_string());
    }
    if n_iterations < 1 {
        return Err("n_iterations must be >= 1".to_string());
    }
    if reference.is_empty()
        || reference.iter().any(|value| !value.is_finite())
        || neighbors
            .iter()
            .any(|series| series.is_empty() || series.iter().any(|value| !value.is_finite()))
    {
        return Err("DBA inputs must be non-empty and finite".to_string());
    }

    let mut barycenter = reference.to_vec();
    for _ in 0..n_iterations {
        let mut sums = vec![0.0; barycenter.len()];
        let mut totals = vec![0.0; barycenter.len()];
        for (series_index, values) in std::iter::once(reference)
            .chain(neighbors.iter().map(Vec::as_slice))
            .enumerate()
        {
            let (_, path) = dtw_alignment(&barycenter, values, band)?;
            let weight = weights[series_index];
            for (barycenter_index, values_index) in path {
                sums[barycenter_index] += weight * values[values_index];
                totals[barycenter_index] += weight;
            }
        }
        for index in 0..barycenter.len() {
            if totals[index] > 0.0 {
                barycenter[index] = sums[index] / totals[index];
            }
        }
    }
    Ok(barycenter)
}

fn moving_average(values: &[f64], period: Option<usize>) -> Vec<f64> {
    let n = values.len();
    let mut weights = match period {
        None => {
            let mut window = (n / 10) | 1;
            window = window.max(3);
            if window > n {
                window = if n % 2 == 1 { n } else { n - 1 };
            }
            vec![1.0 / window as f64; window]
        }
        Some(period) if period % 2 == 1 => vec![1.0 / period as f64; period],
        Some(period) => {
            let mut result = vec![1.0 / period as f64; period + 1];
            result[0] *= 0.5;
            result[period] *= 0.5;
            result
        }
    };
    if weights.len() > n {
        let window = if n % 2 == 1 { n } else { n - 1 };
        weights = vec![1.0 / window as f64; window];
    }
    let width = weights.len();
    let mut computed = Vec::with_capacity(n - width + 1);
    for start in 0..=n - width {
        computed.push(
            values[start..start + width]
                .iter()
                .zip(&weights)
                .map(|(value, weight)| value * weight)
                .sum(),
        );
    }
    let left = (width - 1) / 2;
    let right = n - computed.len() - left;
    let mut trend = Vec::with_capacity(n);
    trend.extend(std::iter::repeat_n(computed[0], left));
    trend.extend_from_slice(&computed);
    trend.extend(std::iter::repeat_n(*computed.last().unwrap(), right));
    trend
}

/// Classical moving-average decomposition used by the Python feature helpers.
pub fn classical_decompose(values: &[f64], period: Option<usize>) -> Result<Decomposition, String> {
    validate_values(values)?;
    if period == Some(0) || period == Some(1) {
        return Err("period must be >= 2 when provided".to_string());
    }
    let usable_period = period.filter(|period| *period <= values.len() / 2);
    let trend = moving_average(values, usable_period);
    let mut seasonal = vec![0.0; values.len()];
    if let Some(period) = usable_period {
        let mut phase_means = vec![0.0; period];
        for (phase, phase_mean) in phase_means.iter_mut().enumerate() {
            let mut total = 0.0;
            let mut count = 0;
            for index in (phase..values.len()).step_by(period) {
                total += values[index] - trend[index];
                count += 1;
            }
            *phase_mean = total / count as f64;
        }
        let phase_mean = phase_means.iter().sum::<f64>() / period as f64;
        for value in &mut phase_means {
            *value -= phase_mean;
        }
        for (index, value) in seasonal.iter_mut().enumerate() {
            *value = phase_means[index % period];
        }
    }
    let remainder = values
        .iter()
        .zip(&trend)
        .zip(&seasonal)
        .map(|((&value, &trend), &seasonal)| value - trend - seasonal)
        .collect();
    Ok((trend, seasonal, remainder))
}

fn variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

fn spectral_entropy(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let centered: Vec<f64> = values.iter().map(|value| value - mean).collect();
    if variance(&centered) <= f64::EPSILON {
        return 0.0;
    }
    let n_bins = values.len() / 2;
    if n_bins <= 1 {
        return 0.0;
    }
    let power: Vec<f64> = if values.len().is_power_of_two() {
        fft::rfft(&centered)[1..=n_bins]
            .iter()
            .map(|value| value.norm_sqr())
            .collect()
    } else {
        (1..=n_bins)
            .map(|bin| {
                let frequency = -2.0 * PI * bin as f64 / values.len() as f64;
                let (real, imaginary) = centered.iter().enumerate().fold(
                    (0.0, 0.0),
                    |(real, imaginary), (index, &value)| {
                        let angle = frequency * index as f64;
                        (real + value * angle.cos(), imaginary + value * angle.sin())
                    },
                );
                real * real + imaginary * imaginary
            })
            .collect()
    };
    let total = power.iter().sum::<f64>();
    if total <= f64::EPSILON {
        return 0.0;
    }
    let entropy = power
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| {
            let probability = value / total;
            -probability * probability.ln()
        })
        .sum::<f64>();
    (entropy / (n_bins as f64).ln()).clamp(0.0, 1.0)
}

/// Return spectral entropy, trend strength, seasonal strength, and ACF(1).
pub fn compute_features(
    values: &[f64],
    period: Option<usize>,
) -> Result<(f64, f64, f64, f64), String> {
    validate_values(values)?;
    if period == Some(0) || period == Some(1) {
        return Err("period must be >= 2 when provided".to_string());
    }
    let (trend, seasonal, remainder) = classical_decompose(values, period)?;
    let trend_plus_remainder: Vec<f64> = trend
        .iter()
        .zip(&remainder)
        .map(|(trend, remainder)| trend + remainder)
        .collect();
    let trend_denominator = variance(&trend_plus_remainder);
    let trend_strength = if trend_denominator <= f64::EPSILON {
        0.0
    } else {
        (1.0 - variance(&remainder) / trend_denominator).clamp(0.0, 1.0)
    };
    let seasonal_strength = if period.is_none()
        || period.is_some_and(|period| period > values.len() / 2)
        || seasonal.iter().all(|value| *value == 0.0)
    {
        0.0
    } else {
        let seasonal_plus_remainder: Vec<f64> = seasonal
            .iter()
            .zip(&remainder)
            .map(|(seasonal, remainder)| seasonal + remainder)
            .collect();
        let denominator = variance(&seasonal_plus_remainder);
        if denominator <= f64::EPSILON {
            0.0
        } else {
            (1.0 - variance(&remainder) / denominator).clamp(0.0, 1.0)
        }
    };
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = variance(values);
    let acf1 = if var == 0.0 {
        0.0
    } else {
        values[..values.len() - 1]
            .iter()
            .zip(&values[1..])
            .map(|(left, right)| (left - mean) * (right - mean))
            .sum::<f64>()
            / values.len() as f64
            / var
    };
    Ok((
        spectral_entropy(values),
        trend_strength,
        seasonal_strength,
        acf1,
    ))
}

/// Bootstrap decomposition remainders using randomly selected moving blocks.
pub fn moving_block_bootstrap(
    values: &[f64],
    period: Option<usize>,
    block_size: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let (trend, seasonal, remainder) = classical_decompose(values, period)?;
    if block_size < 2 || block_size > values.len() {
        return Err("block_size must be in [2, len(values)]".to_string());
    }
    let mut rng = SfRng::new(seed);
    let n_blocks = values.len() / block_size + 2;
    let mut sampled = Vec::with_capacity(n_blocks * block_size);
    for _ in 0..n_blocks {
        let start = rng.integers(0, (values.len() - block_size + 1) as i32) as usize;
        sampled.extend_from_slice(&remainder[start..start + block_size]);
    }
    let offset = rng.integers(0, block_size as i32) as usize;
    Ok(trend
        .iter()
        .zip(&seasonal)
        .zip(&sampled[offset..offset + values.len()])
        .map(|((&trend, &seasonal), &remainder)| trend + seasonal + remainder)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtw_identity_and_path() {
        let values = [1.0, 2.0, 3.0];
        let (distance, path) = dtw_alignment(&values, &values, Some(0)).unwrap();
        assert_eq!(distance, 0.0);
        assert_eq!(path, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn decomposition_reconstructs_input() {
        let values: Vec<f64> = (0..48)
            .map(|index| index as f64 / 10.0 + (2.0 * PI * index as f64 / 12.0).sin())
            .collect();
        let (trend, seasonal, remainder) = classical_decompose(&values, Some(12)).unwrap();
        for index in 0..values.len() {
            assert!(
                (values[index] - trend[index] - seasonal[index] - remainder[index]).abs() < 1e-12
            );
        }
    }

    #[test]
    fn dba_and_mbb_are_deterministic() {
        let reference = vec![0.0, 1.0, 2.0, 1.0];
        let neighbors = vec![vec![0.0, 1.2, 1.8, 1.0]];
        let a = dba_barycenter(&reference, &neighbors, &[1.0, 1.0], 2, Some(1)).unwrap();
        let b = dba_barycenter(&reference, &neighbors, &[1.0, 1.0], 2, Some(1)).unwrap();
        assert_eq!(a, b);

        let values: Vec<f64> = (0..24).map(|index| index as f64).collect();
        assert_eq!(
            moving_block_bootstrap(&values, Some(4), 4, 9).unwrap(),
            moving_block_bootstrap(&values, Some(4), 4, 9).unwrap()
        );
    }
}
