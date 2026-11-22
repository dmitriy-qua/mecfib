# mecfib

Adaptive multi-energy hybrid ensemble for short-term forecasting of heating, cooling, domestic hot water (DHW) and electricity in buildings.

Code, data and run configurations for the article *"Adaptive multi-energy hybrid ensemble for short-term forecasting of heating, cooling, DHW, and electricity in buildings"* (D. Karpenko, submitted to *Energy Reports*).

The forecaster issues, at 00:00 of each day, hourly forecasts for the next 24 hours of every energy channel of a site. Two base learners, a gradient-boosted tree model (XGBoost) and an LSTM-CNN, are trained per channel (a Temporal Fusion Transformer is optional). Their forecasts are combined with a zero forecast by weights that are re-chosen for every day and forecast hour from the errors of the previous 42 days, then blended with the weekly naive forecast by a causal anchor. Periodic retraining with a hold-out guard is available as an option.

## Contents

| Path | Content |
|---|---|
| `forecasting/consumption/` | Training (`train_pipeline.py`, `gbdt/`, `lstm_cnn/`, `transformer/`) and day-ahead forecasting with optional retraining (`adaptive/`) |
| `forecasting/data/<dataset>/` | Preparation scripts and settings of the four datasets; `processed/consumption/` holds the hourly training and test files used in the article |
| `ensemble_replay.py` | Final ensemble (rolling weights with the zero member, causal anchor, gates) and its accuracy from the member forecasts stored by a forecasting run |
| `reproduce.py` | Re-runs any configuration of the article from `configs/` |
| `configs/` | Parameters of every training and forecasting run of the article; `configs/index.csv` lists them by dataset, seed and configuration |
| `results/` | Results of the article in compact form (see below) |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

A GPU is recommended for training the LSTM-CNN; everything else runs on a CPU. All commands are run from the repository root.

## Data

Four public datasets are used. The hourly files in `forecasting/data/<dataset>/processed/consumption/` were derived from them with the `prepare.py` and `split_train_test.py` scripts of each dataset folder, and complemented with weather and air-quality variables from the Open-Meteo archive (https://open-meteo.com).

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

   The run writes the hourly forecasts of every member to `runs/<run id>/predictions_non_adaptive.csv.gz` (frozen models) and, for configurations with retraining, `predictions_adaptive.csv.gz`.

3. **Compute the final ensemble and its accuracy** from these member forecasts:

   ```
   python ensemble_replay.py runs/<forecasting run id> --out metrics.csv
   ```

`python reproduce.py <config> --dry_run` prints the command without running it. The modules can also be called directly, for example `python -m forecasting.consumption.adaptive.run --help`.

The main configurations are "GBDT + LSTM-CNN, static weights, frozen" (frozen models, seeds 42–46) and "GBDT + LSTM-CNN, rolling weights, scheduled retraining with guard" (frozen and retrained forecasts from the same run). Training is seeded, but results can differ slightly between hardware and library versions.

## Results

`results/` holds the numbers behind the tables of the article, aggregated over seeds:

- `metrics_by_dataset.csv`: MAE, RMSE and scaled daily loss of the weekly naive, GBDT alone, GBDT with the zero member, GBDT with the LSTM-CNN, and the full ensemble, for frozen and retrained models.
- `metrics_by_channel.csv`: the same by energy channel, with the bias and the mean load.
- `comparisons.csv`: paired comparisons between member sets and between frozen and retrained models (relative difference, 95 % block-bootstrap interval, Diebold–Mariano p-value).

Overall MAE of a dataset is the mean of its channel MAEs. The scaled daily loss divides each channel's absolute error by its mean load before averaging over channels.

## Citation

See `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). Data: the licences of the original datasets apply.
