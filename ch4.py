import pandas as pd

weather = pd.read_csv("C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/nyc_weather_2018.csv")
weather.head()

snow_data = weather.query('datatype == "SNOW" and value>0 and station.str.contains("US1NY")')
snow_data.head()

import sqlite3

with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/weather.db') as connection:
    snow_data_from_db = pd.read_sql(
        'SELECT * FROM weather WHERE datatype == "SNOW" AND value>0 and station LIKE "%US1NY%"',
        connection
    )

snow_data.reset_index().drop(columns='index').equals(snow_data_from_db)

station_info = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/weather_stations.csv')
station_info.head()

station_info.id.describe()
weather.station.describe()

station_info.shape[0],weather.shape[0]

def get_row_count(*dfs):
    return [df.shape[0] for df in dfs]
get_row_count(station_info, weather)

inner_join = weather.merge(station_info,
                           left_on='station',
                           right_on='id')
inner_join.sample(5, random_state=0)

weather.merge(station_info.rename(dict(id='station'),axis=1), on='station').sample(5, random_state=0)

left_join = station_info.merge(weather, left_on='id', right_on='station', how='left')
right_join = weather.merge(station_info, left_on='station', right_on='id', how='right')
right_join[right_join.datatype.isna()].head()

left_join.sort_index(axis=1).sort_values(['date', 'station'], ignore_index=True).equals(
    right_join.sort_index(axis=1).sort_values(['date', 'station'], ignore_index=True)
)

get_row_count(inner_join, left_join, right_join)

outer_join = weather.merge(
    station_info[station_info.id.str.contains('US1NY')],
    left_on='station', right_on='id', how='outer', indicator=True
)

pd.concat([
    outer_join.query(f'_merge == "{kind}"').sample(2, random_state=0)
    for kind in outer_join._merge.unique()
]).sort_index()

import sqlite3

with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/weather.db') as connection:
    inner_join_from_db = pd.read_sql(
        'SELECT * FROM weather JOIN stations ON weather.station == stations.id',
        connection
    )
inner_join_from_db.shape == inner_join.shape

dirty_data = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/dirty_data.csv', index_col='date'
).drop_duplicates().drop(columns='SNWD')
dirty_data.head()

valid_station = dirty_data.query('station != "?"').drop(columns=['WESF', 'station'])
station_with_wesf = dirty_data.query('station == "?"').drop(columns=['station','TOBS','TMIN','TMAX'])

valid_station.merge(
    station_with_wesf, how='left', left_index=True, right_index=True
).query('WESF>0').head()

valid_station.merge(
    station_with_wesf, how='left', left_index=True, right_index=True,
    suffixes=('', '_?')
).query('WESF > 0').head()

valid_station.join(station_with_wesf, how='left', rsuffix='_?').query('WESF > 0').head()

weather.set_index('station', inplace=True)
station_info.set_index('id', inplace=True)

weather.index.intersection(station_info.index)

weather.index.difference(station_info.index)
station_info.index.difference(weather.index)

ny_in_name = station_info[station_info.index.str.contains('US1NY')]

ny_in_name.index.difference(weather.index).shape[0]\
+ weather.index.difference(ny_in_name.index).shape[0]\
== weather.index.symmetric_difference(ny_in_name.index).shape[0]

weather.index.unique().union(station_info.index)

# 2-dataframe_operations.ipynb

import numpy as np
import pandas as pd

weather = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/nyc_weather_2018.csv', parse_dates=['date'])
weather.head()

fb = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/fb_2018.csv', index_col='date', parse_dates=True)
fb.head()

fb.assign(
    abs_z_score_volume=lambda x: \
    x.volume.sub(x.volume.mean()).div(x.volume.std()).abs()
).query('abs_z_score_volume > 3')

fb.assign(
    volume_pct_change=fb.volume.pct_change(),
    pct_change_rank=lambda x: \
    x.volume_pct_change.abs().rank(ascending=False)
).nsmallest(5, 'pct_change_rank')

fb['2018-01-11':'2018-01-12']
(fb>215).any()

(fb>215).all()

(fb.volume.value_counts()>1).sum()
(fb.volume.value_counts()>1).any()

volume_binned=pd.cut(fb.volume, bins=3, labels=['low','med', 'high'])
volume_binned.value_counts()

fb[volume_binned == 'high'].sort_values('volume', ascending=False)

fb['2018-07-25':'2018-07-26']

fb['2018-03-16':'2018-03-20']

volume_qbinned = pd.qcut(
    fb.volume, q=4, labels=['q1','q2','q3','q4']
)
volume_qbinned.value_counts()

from visual_aids.misc_viz import low_med_high_bins_viz

low_med_high_bins_viz(
    fb,'volume', ylabel='volume traded',
    title='Daily Volume Traded of Facebook Stock in 2018 (with bins)'
)

central_park_weather = weather\
    .query('station == "GHCND:USW00094728"')\
    .pivot(index='date', columns='datatype', values='value')

oct_weather_z_scores = central_park_weather\
    .loc['2018-10', ['TMIN', 'TMAX', 'PRCP']]\
    .apply(lambda x: x.sub(x.mean()).div(x.std()))
oct_weather_z_scores.describe().T

oct_weather_z_scores.query('PRCP > 3').PRCP

central_park_weather.loc['2018-10', 'PRCP'].describe()

fb.apply(
    lambda x: np.vectorize(lambda y: len(str(np.ceil(y))))(x)
).astype('int64').equals(
    fb.applymap(lambda x: len(str(np.ceil(x))))
)

import time

import numpy as np
import pandas as pd

np.random.seed(0)

vectorized_results = {}
iteritems_results = {}

for size in [10, 100, 1000, 10000, 100000, 500000, 1000000, 5000000, 10000000]:
    # set of numbers to use
    test = pd.Series(np.random.uniform(size=size))

    # time the vectorized operation
    start = time.time()
    x = test + 10
    end = time.time()
    vectorized_results[size] = end - start

    # time the operation with `iteritems()`
    start = time.time()
    x = []
    for i, v in test.iteritems():
        x.append(v + 10)
    x = pd.Series(x)
    end = time.time()
    iteritems_results[size] = end - start

results = pd.DataFrame(
    [pd.Series(vectorized_results, name='vectorized'), pd.Series(iteritems_results, name='iteritems')]
).T

# plotting
ax = results.plot(title='Time Complexity', color=['blue', 'red'], legend=False)

# formatting
ax.set(xlabel='item size (rows)', ylabel='time (s)')
ax.text(0.5e7, iteritems_results[0.5e7] * .9, 'iteritems()', rotation=34, color='red', fontsize=12, ha='center',
        va='bottom')
ax.text(0.5e7, vectorized_results[0.5e7], 'vectorized', color='blue', fontsize=12, ha='center', va='bottom')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

central_park_weather.loc['2018-10'].assign(
    rolling_PRCP=lambda x: x.PRCP.rolling('3D').sum()
)[['PRCP', 'rolling_PRCP']].head(7).T

central_park_weather.loc['2018-10'].rolling('3D').mean().head(7).iloc[:,:6]

central_park_weather['2018-10-01':'2018-10-07'].rolling('3D').agg(
    {'TMAX':'max', 'TMIN':'min', 'AWND':'mean', 'PRCP':'sum'}
).join(
    central_park_weather[['TMAX','TMIN','AWND','PRCP']],
    lsuffix='_rolling'
).sort_index(axis=1)

central_park_weather.loc['2018-06'].assign(
    TOTAL_PRCP=lambda x: x.PRCP.cumsum(),
    AVG_PRCP=lambda x: x.PRCP.expanding().mean()
).head(10)[['PRCP','TOTAL_PRCP','AVG_PRCP']].T

central_park_weather\
    ['2018-10-01':'2018-10-07'].expanding().agg({
    'TMAX': np.max, 'TMIN':np.min,
    'AWND': np.mean, 'PRCP':np.sum
}).join(
    central_park_weather[['TMAX', 'TMIN', 'AWND', 'PRCP']],
    lsuffix='_expanding'
).sort_index(axis=1)

central_park_weather.assign(
AVG=lambda x: x.TMAX.rolling('30D').mean(),
EWMA=lambda x: x.TMAX.ewm(span=30).mean()
).loc['2018-09-29':'2018-10-08', ['TMAX', 'EWMA', 'AVG']].T

def get_info(df):
    return '%d rows, %d cols and max closing z-score: %d'
        % (*df.shape, df.close.max())

get_info(fb.loc['2018-Q1']\
         .apply(lambda x: (x-x.mean())/x.stf())

fb.loc['2018-Q1'].apply(lambda x: (x-x.mean())/x.std())\
    .pipe(get_info()

fb.pipe(pd.DataFrame.rolling, '20D').mean().equals(
    fb.rolling('20D').mean()
)



from window_calc import window_calc
window_calc??

window_calc(fb, pd.DataFrame.expanding, np.median).head()

window_calc(fb, pd.DataFrame.ewm, 'mean', span=3).head()

window_calc(
    central_park_weather.loc['2018-10'],
    pd.DataFrame.rolling,
    {'TMAX':'max', 'TMIN':'min',
     'AWND':'mean','PRCP':'sum'},
    '3D'
).head()