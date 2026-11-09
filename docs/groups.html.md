---
title: Covariate Groups
description: Grouped targets and covariates with known structure: regression priors, coupled channels, VARX panels and covariate-extraction probes
---

::: synforecast.groups.GroupPrior
    options:
      members:
        - sample
        - replay

::: synforecast.groups.RegressionSpec

::: synforecast.groups.FixedRegressionSpec

::: synforecast.groups.CoupledSpec

::: synforecast.groups.VARXSpec

::: synforecast.groups.VARXTruth
    options:
      members:
        - from_recipe
        - split
        - forecast
        - forecast_group

::: synforecast.groups.Term

::: synforecast.groups.RoleSpec

::: synforecast.groups.GeneratorSpec

::: synforecast.groups.GroupBatch
    options:
      members:
        - to_long
        - shuffle_covariates
        - diagnostics

::: synforecast.groups.Group

::: synforecast.groups.probe_suite
