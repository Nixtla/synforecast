//! Sequential recurrences of the Simons-parity mechanistic simulators (epidemic, rainfall-runoff, thermal).
//!
//! Random inputs are drawn in Python; these loops only advance the state, in f64 and in the same operation order as
//! the reference Python loops. `py_max`/`py_min` follow Python's `max`/`min` (first argument unless the second is
//! strictly larger/smaller), so NaN propagates the same way.

#[inline]
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

#[inline]
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline]
fn flow(value: f64, p: f64, z: f64) -> f64 {
    let mean = value * p;
    let spread = py_max(mean * (1.0 - p), 0.0).sqrt();
    py_min(py_max(mean + spread * z, 0.0), value)
}

/// SEIRS compartments with normal-approximated binomial flows; returns (incidence E->I, prevalence I) after burn-in.
pub fn epidemic(
    x0: [f64; 4],
    beta: &[f64],
    p_const: [f64; 3],
    p_birth: f64,
    pop: f64,
    imports: &[f64],
    z: &[f64],
    burn: usize,
    length: usize,
    inc: &mut [f64],
    prev: &mut [f64],
) {
    let [mut s, mut e, mut i, mut r] = x0;
    let [p_e, p_i, p_w] = p_const;
    let keep = 1.0 - p_birth;
    let born = p_birth * pop;
    for step in 0..burn + length {
        let p_s = 1.0 - (-beta[step] * i).exp();
        let zs = &z[4 * step..4 * step + 4];
        let f_s = flow(s, p_s, zs[0]);
        let f_e = flow(e, p_e, zs[1]);
        let f_i = flow(i, p_i, zs[2]);
        let f_r = flow(r, p_w, zs[3]);
        let (s1, e1, i1, r1) = (s - f_s + f_r, e - f_e + f_s, i - f_i + f_e, r - f_r + f_i);
        s = s1 * keep + born;
        e = e1 * keep + imports[step];
        i = i1 * keep;
        r = r1 * keep;
        if step >= burn {
            inc[step - burn] = f_e;
            prev[step - burn] = i;
        }
    }
}

/// Bucket rainfall-runoff model; returns (streamflow, soil plus snow storage) after burn-in.
pub fn rainfall_runoff(
    soil0: f64,
    params: [f64; 7],
    occ: &[f64],
    p01_t: &[f64],
    amount: &[f64],
    snow_frac: &[f64],
    melt_rate: &[f64],
    pet: &[f64],
    burn: usize,
    length: usize,
    flow_out: &mut [f64],
    moist_out: &mut [f64],
) {
    let [smax, k, a, alpha, fg, g, p11] = params;
    let mut soil = soil0;
    let mut snow = 0.0;
    let mut route = 0.0;
    let mut ground = 0.0;
    let mut wet = false;
    for step in 0..burn + length {
        wet = if wet {
            occ[step] < p11
        } else {
            occ[step] < p01_t[step]
        };
        let rain = if wet { amount[step] } else { 0.0 };
        let sf = snow_frac[step];
        let melt = py_min(snow, melt_rate[step]);
        snow = snow + rain * sf - melt;
        soil = soil + rain * (1.0 - sf) + melt;
        soil = py_max(soil - pet[step] * py_min(soil / smax, 1.0), 0.0);
        let over = py_max(soil - smax, 0.0);
        soil -= over;
        let q = py_min(k * smax * (soil / smax).powf(a), soil);
        soil -= q;
        ground = ground + fg * q - g * ground;
        route += alpha * ((1.0 - fg) * q + over - route);
        if step >= burn {
            flow_out[step - burn] = route + g * ground;
            moist_out[step - burn] = soil + snow;
        }
    }
}

/// Two-node thermal network with hysteresis or proportional cooling; returns (temperature, energy) after burn-in.
pub fn thermal(
    t1_0: f64,
    params: [f64; 7],
    proportional: bool,
    ambient: &[f64],
    heat: &[f64],
    burn: usize,
    length: usize,
    temp_out: &mut [f64],
    energy_out: &mut [f64],
) {
    let [r12, r1a, r2a, cap, setpoint, band, power] = params;
    let mut t1 = t1_0;
    let mut t2 = t1_0;
    let mut on = false;
    for step in 0..burn + length {
        on = (t1 > setpoint + band) || (on && t1 > setpoint - band);
        let cool = if proportional {
            power * py_min(py_max((t1 - setpoint + band) / (2.0 * band), 0.0), 1.0)
        } else if on {
            power
        } else {
            0.0
        };
        let amb = ambient[step];
        let h = heat[step];
        let next1 = t1 + r12 * (t2 - t1) + r1a * (amb - t1) + h - cool;
        let next2 = t2 + r12 * cap * (t1 - t2) + r2a * (amb - t2);
        t1 = next1;
        t2 = next2;
        if step >= burn {
            temp_out[step - burn] = t1;
            energy_out[step - burn] = cool + h;
        }
    }
}
