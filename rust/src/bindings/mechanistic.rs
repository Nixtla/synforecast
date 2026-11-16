use crate::generators::mechanistic as gen;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Pair = (Py<PyArray1<f64>>, Py<PyArray1<f64>>);

fn check(name: &str, len: usize, need: usize) -> PyResult<()> {
    if len < need {
        return Err(PyValueError::new_err(format!(
            "{name} needs at least {need} values, got {len}"
        )));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (x0, beta, p_const, p_birth, pop, imports, z, burn, length))]
pub fn epidemic(
    py: Python<'_>,
    x0: [f64; 4],
    beta: PyReadonlyArray1<'_, f64>,
    p_const: [f64; 3],
    p_birth: f64,
    pop: f64,
    imports: PyReadonlyArray1<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    burn: usize,
    length: usize,
) -> PyResult<Pair> {
    let total = burn + length;
    let (beta, imports, z) = (beta.as_slice()?, imports.as_slice()?, z.as_slice()?);
    check("beta", beta.len(), total)?;
    check("imports", imports.len(), total)?;
    check("z", z.len(), 4 * total)?;
    let mut inc = vec![0.0; length];
    let mut prev = vec![0.0; length];
    py.detach(|| {
        gen::epidemic(
            x0, beta, p_const, p_birth, pop, imports, z, burn, length, &mut inc, &mut prev,
        )
    });
    Ok((
        PyArray1::from_vec(py, inc).into(),
        PyArray1::from_vec(py, prev).into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (soil0, params, occ, p01_t, amount, snow_frac, melt_rate, pet, burn, length))]
pub fn rainfall_runoff(
    py: Python<'_>,
    soil0: f64,
    params: [f64; 7],
    occ: PyReadonlyArray1<'_, f64>,
    p01_t: PyReadonlyArray1<'_, f64>,
    amount: PyReadonlyArray1<'_, f64>,
    snow_frac: PyReadonlyArray1<'_, f64>,
    melt_rate: PyReadonlyArray1<'_, f64>,
    pet: PyReadonlyArray1<'_, f64>,
    burn: usize,
    length: usize,
) -> PyResult<Pair> {
    let total = burn + length;
    let series = [
        ("occ", occ.as_slice()?),
        ("p01_t", p01_t.as_slice()?),
        ("amount", amount.as_slice()?),
        ("snow_frac", snow_frac.as_slice()?),
        ("melt_rate", melt_rate.as_slice()?),
        ("pet", pet.as_slice()?),
    ];
    for (name, values) in series.iter() {
        check(name, values.len(), total)?;
    }
    let mut flow = vec![0.0; length];
    let mut moist = vec![0.0; length];
    py.detach(|| {
        gen::rainfall_runoff(
            soil0,
            params,
            series[0].1,
            series[1].1,
            series[2].1,
            series[3].1,
            series[4].1,
            series[5].1,
            burn,
            length,
            &mut flow,
            &mut moist,
        )
    });
    Ok((
        PyArray1::from_vec(py, flow).into(),
        PyArray1::from_vec(py, moist).into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (t1_0, params, proportional, ambient, heat, burn, length))]
pub fn thermal(
    py: Python<'_>,
    t1_0: f64,
    params: [f64; 7],
    proportional: bool,
    ambient: PyReadonlyArray1<'_, f64>,
    heat: PyReadonlyArray1<'_, f64>,
    burn: usize,
    length: usize,
) -> PyResult<Pair> {
    let total = burn + length;
    let (ambient, heat) = (ambient.as_slice()?, heat.as_slice()?);
    check("ambient", ambient.len(), total)?;
    check("heat", heat.len(), total)?;
    let mut temp = vec![0.0; length];
    let mut energy = vec![0.0; length];
    py.detach(|| {
        gen::thermal(
            t1_0,
            params,
            proportional,
            ambient,
            heat,
            burn,
            length,
            &mut temp,
            &mut energy,
        )
    });
    Ok((
        PyArray1::from_vec(py, temp).into(),
        PyArray1::from_vec(py, energy).into(),
    ))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "mechanistic")?;
    m.add_function(wrap_pyfunction!(epidemic, &m)?)?;
    m.add_function(wrap_pyfunction!(rainfall_runoff, &m)?)?;
    m.add_function(wrap_pyfunction!(thermal, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.mechanistic", &m)?;
    Ok(())
}
