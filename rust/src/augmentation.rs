//! Native algorithms shared by augmentation and feature-targeted generation.

use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{fft, rng::SfRng};

pub type Decomposition = (Vec<f64>, Vec<f64>, Vec<f64>);

#[derive(PartialEq)]
struct Neighbor {
    distance: f64,
    index: usize,
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn validate_values(values: &[f64]) -> Result<(), String> {
    if values.len() < 3 {
        return Err("values must contain at least 3 observations".to_string());
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err("values must contain only finite observations".to_string());
    }
    Ok(())
}

fn validate_dtw_inputs(a: &[f64], b: &[f64]) -> Result<(), String> {
    if a.is_empty() || b.is_empty() {
        return Err("DTW inputs must be non-empty one-dimensional arrays".to_string());
    }
    if !a.iter().chain(b).all(|value| value.is_finite()) {
        return Err("DTW inputs must be finite".to_string());
    }
    Ok(())
}

fn band_width(n: usize, m: usize, band: Option<usize>) -> usize {
    band.unwrap_or(n.max(m)).max(n.abs_diff(m))
}

/// Banded DTW dynamic program over two rolling rows.
///
/// Returns the accumulated squared cost. When `parents` is provided it must
/// hold `n * min(2 * width + 1, m)` cells; row `i` stores the columns
/// `lower..=upper` of its band contiguously, starting at column `lower`.
fn dtw_core(a: &[f64], b: &[f64], width: usize, mut parents: Option<&mut [u8]>) -> f64 {
    let (n, m) = (a.len(), b.len());
    let row_width = (2 * width + 1).min(m);
    let mut previous = vec![f64::INFINITY; m + 1];
    let mut current = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;
    for i in 1..=n {
        let lower = 1.max(i.saturating_sub(width));
        let upper = m.min(i.saturating_add(width));
        current[lower - 1] = f64::INFINITY;
        if upper < m {
            current[upper + 1] = f64::INFINITY;
        }
        for j in lower..=upper {
            let mut best = previous[j - 1];
            let mut direction = 0u8;
            if previous[j] < best {
                best = previous[j];
                direction = 1;
            }
            if current[j - 1] < best {
                best = current[j - 1];
                direction = 2;
            }
            current[j] = (a[i - 1] - b[j - 1]).powi(2) + best;
            if let Some(parents) = parents.as_deref_mut() {
                parents[(i - 1) * row_width + (j - lower)] = direction;
            }
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[m]
}

/// Return square-root DTW distance without materializing an alignment path.
pub fn dtw_distance(a: &[f64], b: &[f64], band: Option<usize>) -> Result<f64, String> {
    validate_dtw_inputs(a, b)?;
    let cost = dtw_core(a, b, band_width(a.len(), b.len(), band), None);
    if !cost.is_finite() {
        return Err("DTW alignment is infeasible for the requested band".to_string());
    }
    Ok(cost.sqrt())
}

/// Return square-root DTW distance and an optimal alignment path.
pub fn dtw_alignment(
    a: &[f64],
    b: &[f64],
    band: Option<usize>,
) -> Result<(f64, Vec<(usize, usize)>), String> {
    validate_dtw_inputs(a, b)?;
    let (n, m) = (a.len(), b.len());
    let width = band_width(n, m, band);
    let row_width = (2 * width + 1).min(m);
    let mut parents = vec![u8::MAX; n * row_width];
    let cost = dtw_core(a, b, width, Some(&mut parents));
    if !cost.is_finite() {
        return Err("DTW alignment is infeasible for the requested band".to_string());
    }

    let infeasible = || "DTW alignment is infeasible for the requested band".to_string();
    let (mut i, mut j) = (n, m);
    let mut path = Vec::with_capacity(n + m);
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let lower = 1.max(i.saturating_sub(width));
        if j < lower || j > m.min(i.saturating_add(width)) {
            return Err(infeasible());
        }
        match parents[(i - 1) * row_width + (j - lower)] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            2 => j -= 1,
            _ => return Err(infeasible()),
        }
    }
    path.reverse();
    Ok((cost.sqrt(), path))
}

/// Return the symmetric matrix of banded DTW distances between all series.
///
/// The band for each pair is `max(ceil(window_fraction * max_len), |len_a -
/// len_b| + 1)`. Pairs are evaluated in parallel and the result is a
/// row-major `len(series) x len(series)` matrix with a zero diagonal.
pub fn pairwise_dtw_distances(
    series: &[Vec<f64>],
    window_fraction: f64,
) -> Result<Vec<f64>, String> {
    if !(window_fraction > 0.0 && window_fraction <= 1.0) {
        return Err("window_fraction must satisfy 0 < value <= 1".to_string());
    }
    if series
        .iter()
        .any(|values| values.is_empty() || values.iter().any(|value| !value.is_finite()))
    {
        return Err("DTW inputs must be non-empty and finite".to_string());
    }
    let n = series.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let distances: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let (a, b) = (&series[i], &series[j]);
            let longest = a.len().max(b.len());
            let band = ((window_fraction * longest as f64).ceil() as usize)
                .max(a.len().abs_diff(b.len()) + 1);
            dtw_core(a, b, band_width(a.len(), b.len(), Some(band)), None).sqrt()
        })
        .collect();
    let mut matrix = vec![0.0; n * n];
    for (&(i, j), &distance) in pairs.iter().zip(&distances) {
        matrix[i * n + j] = distance;
        matrix[j * n + i] = distance;
    }
    Ok(matrix)
}

/// Retain only the k nearest neighbors per series, breaking ties by input index.
/// Each unordered pair is evaluated once, in bounded parallel chunks. Storage
/// is O(n * k) plus one chunk, rather than a full pair list and distance matrix.
pub fn nearest_dtw_neighbors(
    series: &[Vec<f64>],
    window_fraction: f64,
    n_neighbors: usize,
) -> Result<Vec<Vec<(usize, f64)>>, String> {
    if !(window_fraction > 0.0 && window_fraction <= 1.0) {
        return Err("window_fraction must satisfy 0 < value <= 1".to_string());
    }
    if n_neighbors == 0 {
        return Err("n_neighbors must be >= 1".to_string());
    }
    for values in series {
        validate_dtw_inputs(values, values)?;
    }
    let k = n_neighbors.min(series.len().saturating_sub(1));
    let mut nearest: Vec<BinaryHeap<Neighbor>> = (0..series.len())
        .map(|_| BinaryHeap::with_capacity(k))
        .collect();
    const CHUNK_SIZE: usize = 4096;
    let mut pairs = Vec::with_capacity(CHUNK_SIZE);
    let mut evaluate = |pairs: &[(usize, usize)]| -> Result<(), String> {
        let distances = pairs
            .par_iter()
            .map(|&(i, j)| {
                let (a, b) = (&series[i], &series[j]);
                let band = ((window_fraction * a.len().max(b.len()) as f64).ceil() as usize)
                    .max(a.len().abs_diff(b.len()) + 1);
                dtw_distance(a, b, Some(band))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (&(i, j), distance) in pairs.iter().zip(distances) {
            for (source, index) in [(i, j), (j, i)] {
                let heap = &mut nearest[source];
                let neighbor = Neighbor { distance, index };
                if heap.len() < k {
                    heap.push(neighbor);
                } else if let Some(mut worst) = heap.peek_mut() {
                    if neighbor < *worst {
                        *worst = neighbor;
                    }
                }
            }
        }
        Ok(())
    };
    for i in 0..series.len() {
        for j in i + 1..series.len() {
            pairs.push((i, j));
            if pairs.len() == CHUNK_SIZE {
                evaluate(&pairs)?;
                pairs.clear();
            }
        }
    }
    evaluate(&pairs)?;
    Ok(nearest
        .into_iter()
        .map(|heap| {
            heap.into_sorted_vec()
                .into_iter()
                .map(|neighbor| (neighbor.index, neighbor.distance))
                .collect()
        })
        .collect())
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
    let (mut width, mut half_endpoints) = match period {
        None => {
            let mut window = (n / 10) | 1;
            window = window.max(3);
            if window > n {
                window = if n % 2 == 1 { n } else { n - 1 };
            }
            (window, false)
        }
        Some(period) if period % 2 == 1 => (period, false),
        Some(period) => (period + 1, true),
    };
    if width > n {
        width = if n % 2 == 1 { n } else { n - 1 };
        half_endpoints = false;
    }
    // Center before accumulating and compensate rounding error as the window
    // slides. Even periods retain the classical half-weighted endpoints.
    let baseline = values[0];
    let mut sum = 0.0;
    let mut correction = 0.0;
    let add = |sum: &mut f64, correction: &mut f64, value: f64| {
        let adjusted = value - *correction;
        let next = *sum + adjusted;
        *correction = (next - *sum) - adjusted;
        *sum = next;
    };
    for value in &values[..width] {
        add(&mut sum, &mut correction, value - baseline);
    }
    let denominator = if half_endpoints { width - 1 } else { width } as f64;
    let mut computed = Vec::with_capacity(n - width + 1);
    for start in 0..=n - width {
        if start > 0 {
            add(&mut sum, &mut correction, -(values[start - 1] - baseline));
            add(
                &mut sum,
                &mut correction,
                values[start + width - 1] - baseline,
            );
        }
        let endpoints = if half_endpoints {
            0.5 * (values[start] - baseline) + 0.5 * (values[start + width - 1] - baseline)
        } else {
            0.0
        };
        computed.push(baseline + (sum - endpoints) / denominator);
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

fn spectral_entropy(values: &[f64]) -> Result<f64, String> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut centered: Vec<f64> = values.iter().map(|value| value - mean).collect();
    if variance(&centered) <= f64::EPSILON {
        return Ok(0.0);
    }
    let n_bins = values.len() / 2;
    if n_bins <= 1 {
        return Ok(0.0);
    }
    let spectrum = fft::rfft_half(&mut centered)?;
    let bins = &spectrum[1..=n_bins];
    let total = bins.iter().map(|value| value.norm_sqr()).sum::<f64>();
    if total <= f64::EPSILON {
        return Ok(0.0);
    }
    let entropy = bins
        .iter()
        .filter_map(|value| {
            let power = value.norm_sqr();
            if power > 0.0 {
                let probability = power / total;
                Some(-probability * probability.ln())
            } else {
                None
            }
        })
        .sum::<f64>();
    Ok((entropy / (n_bins as f64).ln()).clamp(0.0, 1.0))
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
        spectral_entropy(values)?,
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
    let decomposition = classical_decompose(values, period)?;
    if block_size < 2 || block_size > values.len() {
        return Err("block_size must be in [2, len(values)]".to_string());
    }
    Ok(bootstrap_remainder(&decomposition, block_size, seed))
}

/// Decompose once and independently resample the remainder for each seed.
pub fn moving_block_bootstrap_many(
    values: &[f64],
    period: Option<usize>,
    block_size: usize,
    seeds: &[u64],
) -> Result<Vec<Vec<f64>>, String> {
    let decomposition = classical_decompose(values, period)?;
    if block_size < 2 || block_size > values.len() {
        return Err("block_size must be in [2, len(values)]".to_string());
    }
    Ok(seeds
        .iter()
        .map(|&seed| bootstrap_remainder(&decomposition, block_size, seed))
        .collect())
}

fn bootstrap_remainder(decomposition: &Decomposition, block_size: usize, seed: u64) -> Vec<f64> {
    let (trend, seasonal, remainder) = decomposition;
    let length = remainder.len();
    let mut rng = SfRng::new(seed);
    let n_blocks = length / block_size + 2;
    let mut sampled = Vec::with_capacity(n_blocks * block_size);
    for _ in 0..n_blocks {
        let start = rng.integers(0, (length - block_size + 1) as i32) as usize;
        sampled.extend_from_slice(&remainder[start..start + block_size]);
    }
    let offset = rng.integers(0, block_size as i32) as usize;
    trend
        .iter()
        .zip(seasonal)
        .zip(&sampled[offset..offset + length])
        .map(|((&trend, &seasonal), &remainder)| trend + seasonal + remainder)
        .collect()
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
    fn banded_dtw_matches_unbanded_when_band_covers_everything() {
        let a: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).sin()).collect();
        let b: Vec<f64> = (0..37).map(|i| (i as f64 * 0.31 + 0.2).sin()).collect();
        let (full, path_full) = dtw_alignment(&a, &b, None).unwrap();
        let (wide, path_wide) = dtw_alignment(&a, &b, Some(100)).unwrap();
        assert_eq!(full, wide);
        assert_eq!(path_full, path_wide);
        assert_eq!(dtw_distance(&a, &b, None).unwrap(), full);
        let (narrow, _) = dtw_alignment(&a, &b, Some(4)).unwrap();
        assert!(narrow >= full);
        assert_eq!(dtw_distance(&a, &b, Some(4)).unwrap(), narrow);
    }

    #[test]
    fn pairwise_matrix_matches_single_distances() {
        let series = vec![
            (0..30)
                .map(|i| (i as f64 * 0.2).sin())
                .collect::<Vec<f64>>(),
            (0..25).map(|i| (i as f64 * 0.25).cos()).collect(),
            (0..30).map(|i| i as f64 * 0.01).collect(),
        ];
        let matrix = pairwise_dtw_distances(&series, 0.1).unwrap();
        for i in 0..3 {
            assert_eq!(matrix[i * 3 + i], 0.0);
            for j in (i + 1)..3 {
                let longest = series[i].len().max(series[j].len());
                let band = ((0.1 * longest as f64).ceil() as usize)
                    .max(series[i].len().abs_diff(series[j].len()) + 1);
                let expected = dtw_distance(&series[i], &series[j], Some(band)).unwrap();
                assert_eq!(matrix[i * 3 + j], expected);
                assert_eq!(matrix[j * 3 + i], expected);
            }
        }
    }

    #[test]
    fn decomposition_reconstructs_input() {
        use std::f64::consts::PI;
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
