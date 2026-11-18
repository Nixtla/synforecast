---
title: Pretraining Generators
description: TSI, TCM, KernelSynth, MAR, Simons-parity, TempoPFN SDE and SarSim generators for foundation-model pretraining
---

::: synforecast.generators.tsi.TSIGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.tcm.TCMGenerator
    options:
      members:
        - generate_single_series
        - generate

::: synforecast.generators.kernel_synth.KernelSynthGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.mar.MARGenerator
    options:
      members:
        - generate_single_series
        - tune_to_features
        - tuning_diagnostics

::: synforecast.generators.cauker.CauKerGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.polyrhythm.PolyrhythmGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.amfm_wave.AMFMWaveGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.forecast_pfn.ForecastPFNGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.regime_ou.RegimeOUGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.mechanistic.EpidemicGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.mechanistic.FluidQueueGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.mechanistic.RainfallRunoffGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.mechanistic.ThermalGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.tempo_sde.TempoSDEGenerator
    options:
      members:
        - generate_single_series
        - generate_block

::: synforecast.generators.sarsim.SarSimGenerator
    options:
      members:
        - generate_single_series
        - generate_block
