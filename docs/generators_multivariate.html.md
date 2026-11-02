---
title: Multivariate Generators
description: Copula, VAR, VARX, and Gaussian Process generators for correlated time series
---

::: synforecast.generators.copula.CopulaGenerator
    options:
      members:
        - generate_single_series
        - generate

::: synforecast.generators.var.VARGenerator
    options:
      members:
        - generate_single_series
        - generate

::: synforecast.generators.varx.VARXGenerator
    options:
      members:
        - generate_single_series
        - generate
        - truth

::: synforecast.generators.varx.VARXTruth
    options:
      members:
        - forecast

::: synforecast.generators.gaussian_process.GaussianProcessGenerator
    options:
      members:
        - generate_single_series
        - get_model_info
