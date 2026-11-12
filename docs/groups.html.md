---
title: Covariate Groups
description: Grouped targets and covariates with known structure: regression priors, coupled channels, VARX panels and covariate-extraction probes
---

::: synforecast.groups.GroupPrior
    options:
      members:
        - sample
        - replay
        - draw_channel
        - fingerprint

::: synforecast.groups.RegressionSpec

::: synforecast.groups.FixedRegressionSpec

::: synforecast.groups.CoupledSpec

::: synforecast.groups.VARXSpec

::: synforecast.groups.GroupOverlaySpec

::: synforecast.groups.ObservationSpec
    options:
      members:
        - plan

::: synforecast.groups.VARXTruth
    options:
      members:
        - from_recipe
        - split
        - forecast
        - forecast_group

::: synforecast.groups.Term

::: synforecast.groups.GeneratorSpec

::: synforecast.groups.default_target_pool

::: synforecast.groups.default_driver_pool

::: synforecast.groups.GroupBatch
    options:
      members:
        - to_padded
        - to_long
        - shuffle_covariates
        - observe
        - diagnostics

::: synforecast.groups.Group
    options:
      members:
        - observe
        - target_indices
        - target_index
        - covariate_indices
        - length

::: synforecast.groups.probe_suite
