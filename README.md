# mecfib

Adaptive multi-energy hybrid ensemble for short-term forecasting of heating, cooling, domestic hot water (DHW) and electricity in buildings.

Code, data and run configurations for the article *"Adaptive multi-energy hybrid ensemble for short-term forecasting of heating, cooling, DHW, and electricity in buildings"* (D. Karpenko, submitted to *Energy Reports*).

The forecaster issues, at 00:00 local time of each day, hourly forecasts for the next 24 hours of every energy channel of a site. Two base learners, a gradient-boosted tree model (XGBoost, applied recursively) and an LSTM-CNN, are trained per channel (a Temporal Fusion Transformer is optional and not used in the main results). Their forecasts are combined with three members that contain no model of their own: a zero forecast, an affine recalibration of the GBDT forecast (intercept and slope fitted by least squares on the last 42 days, per forecast hour) and a Kalman-filter recalibration of the GBDT forecast. ML-Poly online aggregation updates the weights of the five members before every forecast, for each channel and forecast hour. The blend is then mixed with the weekly naive forecast by a causal anchor, and channels that are inactive are set to zero. Periodic retraining with a hold-out guard is available as an option.

## Contents

| Path | Content |
|---|---|
| `forecasting/consumption/` | Training (`train_pipeline.py`, `gbdt/`, `lstm_cnn/`, `transformer/`) and day-ahead forecasting with optional retraining (`adaptive/`) |
| `forecasting/data/<dataset>/` | Preparation scripts and settings of the four datasets; `processed/consumption/` holds the hourly training and test files used in the article |
| `forecasting/data/correct_inputs.py` | Corrections applied to the processed files after preparation (see Data) |
| `ensemble_replay.py` | Final ensemble (zero, affine and Kalman members, ML-Poly, causal anchor, gates) and its accuracy, computed from the member forecasts stored by a forecasting run |
| `reproduce.py` | Re-runs any configuration of the article from `configs/` |
| `configs/` | Parameters of every training and forecasting run of the article; `configs/index.csv` lists them by dataset, seed and configuration |
| `configs/preliminary/` | Runs of the preliminary experiments of the article (optional TFT member, weather stress test), made with the code and data files of release v1.1 |
| `results/` | Results of the article in compact form (see below) |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

A GPU is recommended for training the LSTM-CNN; everything else runs on a CPU. All commands are run from the repository root.

## Data

Four public datasets are used. The hourly files in `forecasting/data/<dataset>/processed/consumption/` were derived from them with the `prepare.py` and `split_train_test.py` scripts of each dataset folder, complemented with weather variables from the Open-Meteo archive (https://open-meteo.com), and then corrected with `forecasting/data/correct_inputs.py`:

- the working-day indicator `is_business` is 0 on weekends and on the public holidays of the site;
- the air-quality columns are set to 0 (they are missing in all KSRP training years and in parts of HEEW) and are not read by the learners;
- the timestamps of Building 59, published in UTC, are converted to local standard time (UTC−8), so that the forecast day starts at local midnight; its WiFi count is aligned with the loads.

The files in this repository are already corrected; each `processed/consumption/` folder holds a `CORRECTED` marker, and the script refuses to correct the same files twice.

| Folder | Site | Source |
|---|---|---|
| `KSRP_Campus` | district-energy campus, Kitakyushu, Japan | https://doi.org/10.1038/s41597-024-04244-6 |
| `HEEW` | university campus, Tempe, Arizona, USA | https://doi.org/10.1038/s41597-025-06010-8 |
| `Building_59` | office building, Berkeley, California, USA | https://doi.org/10.1038/s41597-022-01257-x |
| `NEST_UMAR` | residential unit, Dübendorf, Switzerland | https://doi.org/10.1038/s41597-024-03292-2 |

Please cite the original datasets when using these files; their licences apply.

## Reproducing the results

Each result of the article comes from three steps. `configs/index.csv` gives the run identifiers.

1. **Train the base learners** of a dataset and seed:

   ```
   python reproduce.py configs/<training run id>
   ```

2. **Run the day-ahead forecasts** over the test data with a configuration that uses these models (column `training_run` of the index):

   ```
   python reproduce.py configs/<forecasting run id>
   ```

   The run writes the hourly forecasts of every member to `runs/<run id>/predictions_non_adaptive.csv.gz` (frozen models) and `predictions_adaptive.csv.gz` (retrained models).

3. **Compute the final ensemble and its accuracy** from these member forecasts:

   ```
   python ensemble_replay.py runs/<forecasting run id> --out metrics.csv
   ```

   The output gives, per channel, the MAE, RMSE, MASE and bias of the weekly naive forecast, GBDT alone, the affine and the Kalman recalibrations used alone, GBDT with the three correction members, and the ensemble of the article. `--all` adds the rolling-weight rules and GBDT with the zero member (slower).

`python reproduce.py <config> --dry_run` prints the command without running it. The modules can also be called directly, for example `python -m forecasting.consumption.adaptive.run --help`.

The forecasting runs of the article are "GBDT + LSTM-CNN, scheduled retraining with guard" (seeds 42–46 on Building 59 and NEST UMAR, 42–44 on KSRP and HEEW), which store both the frozen and the retrained member forecasts, and "GBDT + LSTM-CNN, scheduled retraining without guard" (seed 42). Training is seeded, but results can differ slightly between hardware and library versions.

## Results

`results/` holds the numbers behind the tables of the article, aggregated over seeds, for frozen models and for retraining with and without the guard:

- `metrics_by_dataset.csv`: MAE (mean of the channel MAEs), its standard deviation over seeds, RMSE, the mean load-scaled daily loss and the mean loss of the worst 10 % of days, for each rule of `ensemble_replay.py`.
- `metrics_by_channel.csv`: the same by energy channel, with the bias and the mean load.
- `comparisons.csv`: paired comparisons between rules and between frozen and retrained models (relative difference of the load-scaled daily loss, 95 % interval from a 7-day block bootstrap, Diebold–Mariano p-value).

Rules: `weekly naive`; `gbdt` (GBDT alone); `affine` and `kalman` (the recalibrations of GBDT used alone); `gbdt+zero`; `gbdt+corrections` (GBDT with the zero, affine and Kalman members); `gbdt+lstm` (GBDT and LSTM-CNN, rolling weights); `rolling4` (LSTM-CNN, GBDT, zero and affine members, rolling weights); `ensemble` (the configuration of the article). All rules include the causal anchor and the gates.

## Citation

See `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). Data: the licences of the original datasets apply.
