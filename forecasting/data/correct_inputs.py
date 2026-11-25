"""Corrections of the processed consumption files (train and test) of the four datasets.

Run once, after ``prepare.py`` and ``split_train_test.py`` of each dataset folder have written
``<dataset>/processed/consumption/data_hourly_{train,test}.csv``:

    python forecasting/data/correct_inputs.py

The files in this repository are already corrected (each folder holds a ``CORRECTED`` marker,
and the script refuses to run twice on the same files). The corrections are:

1. is_business: 1 on Monday to Friday unless the day is a public holiday of the site
   (Japan; Arizona, US; California, US; canton of Zurich, Switzerland), else 0.
2. Air-quality columns (pm25, pm10, o3, no2, so2, co) set to 0: they are missing in all KSRP
   training years and in parts of HEEW; the learners do not read them (feature lists in configs/).
3. Building 59: the published timestamps are in UTC; they are converted to local standard time
   (UTC-8), the calendar columns are recomputed, the WiFi count (recorded in local time) is
   re-aligned by timestamp, and the train/test boundary is put at local midnight of the
   original boundary date.

The uncorrected files are kept in ``processed/consumption/_uncorrected/``.
"""
import os
import shutil

import holidays
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = {'KSRP_Campus': ('JP', None), 'HEEW': ('US', 'AZ'), 'Building_59': ('US', 'CA'), 'NEST_UMAR': ('CH', 'ZH')}
AQ = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']


def calendar(df, t):
    df['hour'] = t.dt.hour; df['month'] = t.dt.month; df['day_of_week'] = t.dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * df.hour / 24); df['hour_cos'] = np.cos(2 * np.pi * df.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.month / 12); df['month_cos'] = np.cos(2 * np.pi * df.month / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df.day_of_week / 7); df['dow_cos'] = np.cos(2 * np.pi * df.day_of_week / 7)


def correct(site, country, subdiv):
    d = os.path.join(ROOT, site, 'processed', 'consumption')
    if os.path.exists(os.path.join(d, 'CORRECTED')):
        print(site, 'already corrected, skipped')
        return
    bak = os.path.join(d, '_uncorrected'); os.makedirs(bak, exist_ok=True)
    for part in ('train', 'test'):
        shutil.copy2(os.path.join(d, f'data_hourly_{part}.csv'), os.path.join(bak, f'data_hourly_{part}.csv'))
    tr = pd.read_csv(os.path.join(bak, 'data_hourly_train.csv')); te = pd.read_csv(os.path.join(bak, 'data_hourly_test.csv'))
    cols = list(tr.columns)
    df = pd.concat([tr, te], ignore_index=True)
    t_old = pd.to_datetime(df['date'])
    boundary = pd.to_datetime(te['date'].iloc[0])
    if site == 'Building_59':
        heat_by_date = df.groupby(t_old.dt.normalize()).is_heating_season.max()
        season_by_month = df.groupby('month').season.first()
        t = t_old - pd.Timedelta(hours=8)
        if 'wifi_device_count' in df:
            w = pd.Series(df['wifi_device_count'].to_numpy(), index=t_old)
            df['wifi_device_count'] = w.reindex(t).to_numpy()          # value recorded at local clock time t
            df['wifi_device_count'] = df['wifi_device_count'].ffill().bfill()
        df['date'] = t.dt.strftime('%Y-%m-%d %H:%M:%S')
        calendar(df, t)
        df['season'] = df.month.map(season_by_month).astype(int)
        df['is_heating_season'] = t.dt.normalize().map(heat_by_date).ffill().bfill().astype(int)
        boundary = boundary.normalize()
    else:
        t = t_old
    hol = holidays.country_holidays(country, subdiv=subdiv, years=range(t.dt.year.min(), t.dt.year.max() + 1))
    is_hol = t.dt.date.map(lambda x: x in hol).to_numpy()
    df['is_business'] = ((t.dt.weekday < 5).to_numpy() & ~is_hol).astype(int)
    for c in AQ:
        if c in df:
            df[c] = 0.0
    split = (t >= boundary).to_numpy()
    df.loc[~split, cols].to_csv(os.path.join(d, 'data_hourly_train.csv'), index=False)
    df.loc[split, cols].to_csv(os.path.join(d, 'data_hourly_test.csv'), index=False)
    open(os.path.join(d, 'CORRECTED'), 'w').write('corrected by forecasting/data/correct_inputs.py\n')
    print(site, 'train', int((~split).sum()), 'test', int(split.sum()), 'working-day share %.3f' % df.is_business.mean())


if __name__ == '__main__':
    for site, (country, subdiv) in SITES.items():
        correct(site, country, subdiv)
