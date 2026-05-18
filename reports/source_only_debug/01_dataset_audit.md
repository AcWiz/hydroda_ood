# Check 1: Dataset Audit

Data path: /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc


## Split: source_train
- Active regions: US-R2|US-R3|US-R4|US-R5|US-R6
- Total samples: 16277, Checked: 10
- Time range: 2015-04-01 to 2015-07-21
- Unique months in checked: 1, Unique seasons: 1
- loss_mask fraction: 21.0114%
- metric_mask fraction: 21.0114%
- loss/metric ratio: 1.000 (should be ~1 if consistent)
- forecast_surface RMSE (loss-valid): 0.010450
- forecast_rootzone RMSE (loss-valid): 0.001075
- increment_surface mean (loss-valid): -0.000535
- increment_surface std: 0.010437
- increment_surface p50: 0.000000
- increment_surface p99: 0.033803
- abs(inc_surface) mean: 0.003950
- abs(inc_surface) p99: 0.044953
- increment_rootzone mean (loss-valid): -0.000007
- increment_rootzone std: 0.001075
- abs(inc_rootzone) mean: 0.000293
- forecast RMSE surface: 0.010450
- forecast RMSE rootzone: 0.001075
- N loss-valid pixels: 344,250
- N metric-valid pixels: 344,250
- Surface inc mean by season: MAM=-0.000535

## Split: source_val
- Active regions: US-R2|US-R3|US-R4|US-R5|US-R6
- Total samples: 2892, Checked: 10
- Time range: 2021-01-01 to 2021-05-09
- Unique months in checked: 1, Unique seasons: 1
- loss_mask fraction: 21.0114%
- metric_mask fraction: 21.0114%
- loss/metric ratio: 1.000 (should be ~1 if consistent)
- forecast_surface RMSE (loss-valid): 0.004879
- forecast_rootzone RMSE (loss-valid): 0.000450
- increment_surface mean (loss-valid): 0.000370
- increment_surface std: 0.004865
- increment_surface p50: 0.000000
- increment_surface p99: 0.016735
- abs(inc_surface) mean: 0.000979
- abs(inc_surface) p99: 0.021376
- increment_rootzone mean (loss-valid): 0.000039
- increment_rootzone std: 0.000449
- abs(inc_rootzone) mean: 0.000091
- forecast RMSE surface: 0.004879
- forecast RMSE rootzone: 0.000450
- N loss-valid pixels: 344,250
- N metric-valid pixels: 344,250
- Surface inc mean by season: DJF=0.000370

## Split: target_support
- Active regions: US-R1
- Total samples: 0, Checked: 0
- loss_mask fraction: nan%
- metric_mask fraction: nan%
- loss/metric ratio: nan (should be ~1 if consistent)
- forecast_surface RMSE (loss-valid): nan
- forecast_rootzone RMSE (loss-valid): nan
- increment_surface mean (loss-valid): nan
- increment_surface std: nan
- increment_surface p50: nan
- increment_surface p99: nan
- abs(inc_surface) mean: nan
- abs(inc_surface) p99: nan
- increment_rootzone mean (loss-valid): nan
- increment_rootzone std: nan
- abs(inc_rootzone) mean: nan
- forecast RMSE surface: nan
- forecast RMSE rootzone: nan
- N loss-valid pixels: 0
- N metric-valid pixels: 0

## Split: target_query
- Active regions: US-R1
- Total samples: 3359, Checked: 10
- Time range: 2023-01-01 to 2023-02-25
- Unique months in checked: 1, Unique seasons: 1
- loss_mask fraction: 2.7374%
- metric_mask fraction: 2.7374%
- loss/metric ratio: 1.000 (should be ~1 if consistent)
- forecast_surface RMSE (loss-valid): 0.005801
- forecast_rootzone RMSE (loss-valid): 0.001099
- increment_surface mean (loss-valid): 0.001005
- increment_surface std: 0.005713
- increment_surface p50: 0.000000
- increment_surface p99: 0.029863
- abs(inc_surface) mean: 0.001537
- abs(inc_surface) p99: 0.029954
- increment_rootzone mean (loss-valid): 0.000198
- increment_rootzone std: 0.001081
- abs(inc_rootzone) mean: 0.000236
- forecast RMSE surface: 0.005801
- forecast RMSE rootzone: 0.001099
- N loss-valid pixels: 44,850
- N metric-valid pixels: 44,850
- Surface inc mean by season: DJF=0.001005